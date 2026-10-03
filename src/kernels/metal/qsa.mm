// src/kernels/metal/qsa.mm - the port of src/kernels/cuda/qsa.cu's host half (K15).  kv_append_step /
// kv_gather_step (the FP16 KV steps) live in kv_q8.mm on this backend, as they do in kv_q8.cu's sibling on
// CUDA - only the wrappers that upload the step state are here.
//
// THE INDEXER'S DOUBLE ARITHMETIC runs as the qsa_ double-float layer in qsa.metal (two f32s, exact
// two-products, compensated sums, a Newton reciprocal square root): the spare key's BIT-EXACT contract and
// the pooled keys' 1e-6 need ~2^-47, which a plain f32 sum misses by 22 bits.  The gate's double sigmoid
// became f32 precise::exp (its tolerance is the fp16 store's own).
//
// qsa_attend's threadgroup stage is FIXED at 4096 + 32 floats (MSL has no dynamic threadgroup size; the CUDA
// kernel's dynamic sizing is a graph-capture requirement that the fixed stage satisfies identically, since
// the kernel still reads the real n_ids from `step`).  max_ids above the capacity is refused loudly - the
// geometry's widest selection is 2051 and the parity test's dense fixture 2100, so 4096 is 2x headroom.
//
// The native indexer bridge that used to ride this file is GONE: native_qsa_indexer.cu is ported
// (native_qsa_indexer.mm), and qsa_parity links the real entry points.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope.hpp"

#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
constexpr int QSA_ATTEND_CAP = 4096;    // qsa.metal's fixed score/weight stage, in ids

void fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    std::exit(1);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa: %s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// The SYNCHRONOUS entry points must check their own wait (the sticky error from a kernel that faulted
/// becomes visible only at the NEXT call's `cudaGetLastError()`, which then names the wrong kernel).
void check_sync(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa: %s (synchronise): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// Geometry validation, verbatim: a wrong `head_dim % 4` would corrupt the gather's uint2 copy and a wrong
/// `n_head % n_head_kv` would produce a plausible attention with the wrong key.
void validate(const QsaShapes& s, const char* who) {
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1) {
        std::fprintf(stderr, "qsa: %s: geometry is not set up\n", who);
        std::exit(1);
    }
    if (s.n_head % s.n_head_kv != 0) {
        std::fprintf(stderr, "qsa: %s: n_head %lld is not a multiple of n_head_kv %lld\n", who,
                     (long long) s.n_head, (long long) s.n_head_kv);
        std::exit(1);
    }
    if (s.head_dim % 4 != 0) {
        std::fprintf(stderr, "qsa: %s: head_dim %lld must be a multiple of 4 (the gather copies uint2)\n", who,
                     (long long) s.head_dim);
        std::exit(1);
    }
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) {
        std::fprintf(stderr, "qsa: %s: n_rot %lld must be even and <= head_dim %lld and idx_dim %lld\n", who,
                     (long long) s.n_rot, (long long) s.head_dim, (long long) s.idx_dim);
        std::exit(1);
    }
    if (s.idx_n_head > 32) {
        std::fprintf(stderr, "qsa: %s: idx_n_head %lld > 32 (one warp per indexer head)\n", who,
                     (long long) s.idx_n_head);
        std::exit(1);
    }
}

inline unsigned grid_for(long long n, int threads) {
    return (unsigned) ((n + threads - 1) / threads);
}

/// The step state, uploaded (per-PROCESS, reused - the upload and the kernel that reads it are on the same
/// stream and the host is the only writer), exactly as on CUDA: this is what lets the host-scalar entry
/// points keep their signatures while the kernels read the device values.
int32_t* step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        if (cudaMalloc(&d_step, qsa_step_bytes()) != cudaSuccess) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\n");
            std::exit(1);
        }
    }
    return d_step;
}

void step_upload_raw(const int32_t* h_step) {
    int32_t* d = step_scratch();
    if (cudaMemcpy(d, h_step, qsa_step_bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "qsa: step upload: cudaMemcpy failed\n");
        std::exit(1);
    }
}

/// From a POSITION: fills all four entries (a caller that also knows `n_kv` is CHECKED against it).
const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv]) {
        std::fprintf(stderr, "qsa: step upload: n_kv %lld disagrees with pos+1 = %d\n", (long long) n_kv_hint,
                     h[kStepNKv]);
        std::exit(1);
    }
    step_upload_raw(h);
    return step_scratch();
}

/// From a WIDTH: only that entry is meaningful, the others are set consistently rather than left stale.
const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}

}  // namespace

// ================= THE CAPTURABLE ENTRY POINTS =================

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    // THE GRID IS `max_blocks`, A CONSTANT - the kernel returns for `b > n_bid`.
    const int threads = 32 * (int) s.idx_n_head;
    metal::Launch k("qsa_index_kernel", (unsigned) max_blocks, 1, 1, (unsigned) threads, 1, 1, 0, stream);
    k.buf(pooled).buf(q_idx).buf(bias).scalar((int) s.idx_n_head).scalar((int) s.idx_dim)
     .scalar((long) s.idx_block).buf(step).buf(cell_scores);
    k.done();
    check_launch("qsa_index");
    if (stream == nullptr) check_sync("qsa_index");
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    metal::Launch k("topk_kernel", 1, 1, 1, 256, 1, 1, 0, stream);
    k.buf(cell_scores).buf(step).buf(ids);
    k.done();
    check_launch("topk_512");
    if (stream == nullptr) check_sync("topk_512");
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights,
                     void* stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    if (max_ids > QSA_ATTEND_CAP) {
        std::fprintf(stderr, "qsa: qsa_attend: max_ids %lld over the fixed threadgroup stage of %d ids\n",
                     (long long) max_ids, QSA_ATTEND_CAP);
        std::exit(1);
    }
    metal::Launch k("qsa_attend_kernel", (unsigned) s.n_head, 1, 1, (unsigned) s.head_dim, 1, 1, 0, stream);
    k.buf(q).buf(k_scratch).buf(v_scratch).buf(step)
     .scalar((int) s.n_head).scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.head_dim)
     .buf(attn).buf(weights);
    k.done();
    check_launch("qsa_attend");
    if (stream == nullptr) check_sync("qsa_attend");
}

// ================= host entry points =================

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    // ONE PLACE COMPUTES ALL FOUR (n_kv, n_bid and width are DERIVED from pos)
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    if (pos < 0) { validate(s, "kv_append"); fail("kv_append: pos < 0"); }
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream);
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    // **ONE LAUNCH, NO HOST BRANCH ON THE POSITION** - the completion rotation is inside the kernel, so the
    // arguments are identical for every token and a captured graph replays correctly.
    metal::Launch k("indexer_key_append_kernel", 1, 1, 1, (unsigned) s.idx_dim, 1, 1, 0, stream);
    k.buf(raw).buf(pos_dev).scalar((int) pos_base).buf(w_k_norm).scalar(eps)
     .buf(b.tail).buf(b.dead).buf(b.pooled).buf(b.block_pos)
     .scalar((int) s.idx_dim).scalar((int) s.idx_block).scalar((int) s.n_rot)
     .buf(cos_tab).buf(sin_tab).buf(mrope_table());
    k.done();
    check_launch("indexer_key_append");
    if (stream == nullptr) check_sync("indexer_key_append");
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (n_bid < 0 || n_kv <= 0) fail("qsa_index: n_bid < 0 or n_kv <= 0");
    if ((n_kv / s.idx_block) != n_bid) {
        std::fprintf(stderr, "qsa: qsa_index: n_bid %lld is not n_kv %lld / r %lld\n", (long long) n_bid,
                     (long long) n_kv, (long long) s.idx_block);
        std::exit(1);
    }
    // THIS TOKEN'S counts as the CAPACITIES: correct for a direct launch, WRONG in a graph (the layer passes
    // a constant `max_blocks` from its state instead - see the note in qsa.hpp).
    qsa_index_step(pooled, q_idx, bias, s, step_upload(n_kv - 1, n_kv, s), n_bid + 1, cell_scores, stream);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream) {
    validate(s, "topk_512");
    if (n_kv <= 0) return;
    if (n_kv > kTopkMaxCells) {
        std::fprintf(stderr, "qsa: topk_512: n_kv %lld > kTopkMaxCells %lld (one block; phase 3 replaces this)\n",
                     (long long) n_kv, (long long) kTopkMaxCells);
        std::exit(1);
    }
    topk_512_step(cell_scores, s, cap, step_upload(n_kv - 1, n_kv, s), ids, stream);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (n_ids <= 0) return;
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch,
                   v_scratch, stream);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    if (n_ids < 0) fail("qsa_attend: n_ids < 0");
    if (n_ids == 0) {
        // the empty selection goes through the kernel's own zero path; `max_ids` must still be positive
        qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(0, s),
                        qsa_selection_width(kTopkMaxCells, s), s, attn, weights, stream);
        return;
    }
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    metal::Launch k("qsa_gate_apply_f32_kernel", grid_for(n, 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(attn).buf(q_full).scalar((int) s.n_head).scalar((int) s.head_dim).buf(out);
    k.done();
    check_launch("qsa_gate_apply_f32");
    if (stream == nullptr) check_sync("qsa_gate_apply_f32");
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    metal::Launch k("qsa_gate_apply_kernel", grid_for(n, 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(attn).buf(q_full).scalar((int) s.n_head).scalar((int) s.head_dim).buf(out);
    k.done();
    check_launch("qsa_gate_apply");
    if (stream == nullptr) check_sync("qsa_gate_apply");
}

}  // namespace strata::kernels

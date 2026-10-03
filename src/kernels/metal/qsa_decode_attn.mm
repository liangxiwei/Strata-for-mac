// src/kernels/metal/qsa_decode_attn.mm - the port of src/kernels/cuda/qsa_decode_attn.cu's host half (K15).
// The validation and the scratch layout are the CUDA file's, verbatim; the launches flatten QsaAttnPools
// into one bound buffer per pointer (rule 9 - see qsa_decode_attn.metal's header).
#include "strata/kernels/qsa_decode_attn.hpp"

#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block

struct AttnMode {
    int kv_mode;
    const char* kernel;
};

/// The pool format from which pointers are non-null, exactly the CUDA file's reading.
AttnMode pick_mode(const QsaAttnPools& pools) {
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    switch (kv_mode) {
    case 3: return {3, "attn_chunk_kernel_m3"};
    case 2: return {2, "attn_chunk_kernel_m2"};
    case 1: return {1, "attn_chunk_kernel_m1"};
    default: return {0, "attn_chunk_kernel_m0"};
    }
}

void launch_chunk(const char* kernel, const float* q, const QsaAttnPools& p, const int32_t* ids,
                  const int32_t* step, int n_kv_heads, int page_size, float scale, float* part_acc,
                  float* part_m, float* part_l, int n_chunks, int cap, long long stride, unsigned gx,
                  unsigned gy, unsigned gz, void* stream) {
    metal::Launch k(kernel, gx, gy, gz, 256, 1, 1, 0, stream);
    k.buf(q)
     .buf(p.k_pool)
     .buf(p.v_pool)
     .buf(p.k_q)
     .buf(p.v_q)
     .buf(p.k_scale)
     .buf(p.v_scale)
     .buf(p.k_q4)
     .buf(p.v_q4)
     .buf(p.page_table)
     .buf(ids)
     .buf(step)
     .scalar(n_kv_heads)
     .scalar(page_size)
     .scalar(scale)
     .buf(part_acc)
     .buf(part_m)
     .buf(part_l)
     .scalar(n_chunks)
     .scalar((long) cap)
     .scalar((long) stride);
    k.done();
}

void launch_merge(const float* part_acc, const float* part_m, const float* part_l, int n_chunks, float* attn,
                  int n_head, long long stride, unsigned gy, void* stream) {
    metal::Launch m("attn_merge_kernel", (unsigned) n_head, gy, 1, HD, 1, 1, 0, stream);
    m.buf(part_acc).buf(part_m).buf(part_l).scalar(n_chunks).buf(attn).scalar(n_head).scalar((long) stride);
    m.done();
}

}  // namespace

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const AttnMode m = pick_mode(pools);
    // the single-query form is the batched one with z = 1 (CUDA's default cap/stride are 0 there, and the
    // kernel's gpos.z offsets are zero with one z slice, so the real values change nothing)
    launch_chunk(m.kernel, q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m,
                 part_l, n_chunks, (int) cap, (long long) qsa_decode_attn_scratch_floats(cap, s),
                 (unsigned) n_chunks, (unsigned) s.n_head_kv, 1, stream);
    launch_merge(part_acc, part_m, part_l, n_chunks, attn, (int) s.n_head,
                 (long long) qsa_decode_attn_scratch_floats(cap, s), 1, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn_batch: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const AttnMode m = pick_mode(pools);
    launch_chunk(m.kernel, q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m,
                 part_l, n_chunks, (int) cap, stride, (unsigned) n_chunks, (unsigned) s.n_head_kv,
                 (unsigned) n_q, stream);
    launch_merge(part_acc, part_m, part_l, n_chunks, attn, (int) s.n_head, stride, (unsigned) n_q, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels

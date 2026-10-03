// src/kernels/metal/verify_kernels.mm - the port of src/kernels/cuda/verify_kernels.cu's host half (K21).
// Same entry points, same validation and the same launch order; the launches are metal::Launch chains whose
// argument order IS each kernel's [[buffer(N)]] order.
//
// THREE BACKEND-SPECIFIC SHAPES (each documented at its site, all measured - docs/PORT_METAL/PROGRESS.md):
//   * RULE 9 (fetch_blobs): its CUDA kernel dereferences blob pointers loaded from device memory, which is
//     not dereferenceable on this GPU.  The entry syncs, reads the count and the pointer table back to the
//     host (the moe_grouped_s2 precedent) and stages each blob with the shim's stream-ordered
//     cudaMemcpyAsync - the sources are mapped host memory the runtime knows, so the copies are blits.  Like
//     moe_grouped_s2 this makes it not graph-capturable; only the P6 verify path calls it.
//   * RULE 10 (gdn_step_norm_multi): the kernel stages the per-thread recurrence state in a private device
//     buffer (verify_kernels.metal's file comment) - the launcher owns that buffer, laid out like `state`,
//     allocated once and never reallocated (a captured launch binds it; a realloc would dangle the tape).
//   * gpu_stamp: MSL has no GPU clock, so a stamp writes one strictly increasing DEVICE counter value per
//     executed stamp (a private 8-byte buffer, incremented on the device at execution time so captured
//     replays keep the sequence increasing).  The stamps feed STRATA_VERIFY_PROFILE only: on this backend a
//     stage difference counts STAMP LAUNCHES between stages, not nanoseconds - verify.cpp prints them with
//     its ns->ms arithmetic, so the printed "ms" numbers are launch counts / 1e6, not wall time.
//   * MEASURED (this machine, M2 Max), for the engine link (M5): the wait_flag_ge spins read a mapped host
//     flag correctly when its value was stored BEFORE the dispatch (or before a captured replay - both
//     verified by /tmp probes), but a CPU store made WHILE the kernel is already spinning was never observed
//     (a plain store: not in 8 s; a volatile + seq_cst + __sync_synchronize store: once after 4.1 s, then
//     never again in two 15 s runs).  The kernels are the faithful spin; the split-verify window's overlap
//     (release a spinning wait from the host) will not work on this backend as-is - and a killed spinner
//     WEDGES the whole GPU until a sleep cycle, so any M5 work here must raise flags before dispatch/replay.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

/// The recurrence's state staging (rule 10; see the file comment).  One buffer for the process, sized like
/// the largest layer's state - S * S * h_v floats, the (row, head, col) layout the kernel strides
/// (gdn_parity's st_dev is the same count) - never freed or reallocated; a captured launch binds it.
float* step_state_shadow(int h_v) {
    static float* shadow = nullptr;
    static size_t floats_have = 0;
    const size_t floats = (size_t) S * (size_t) S * (size_t) h_v;
    if (floats > floats_have) {
        if (shadow != nullptr) {   // a second, larger model: refuse rather than dangle a captured tape
            std::fprintf(stderr, "gdn_step_norm_multi: state staging buffer is too small for h_v = %d\n", h_v);
            std::exit(1);
        }
        const cudaError_t e = cudaMalloc((void**) &shadow, floats * sizeof(float));
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_step_norm_multi: no state staging buffer: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        floats_have = floats;
    }
    return shadow;
}

/// gpu_stamp's monotonic counter (see the file comment): incremented ON THE DEVICE at execution time, so
/// captured replays advance it exactly as executed stamps do.
unsigned long long* stamp_counter() {
    static unsigned long long* counter = nullptr;
    if (counter == nullptr) {
        const cudaError_t e = cudaMalloc((void**) &counter, sizeof(unsigned long long));
        if (e != cudaSuccess || cudaMemset(counter, 0, sizeof(unsigned long long)) != cudaSuccess) {
            std::fprintf(stderr, "gpu_stamp: no counter buffer\n");
            std::exit(1);
        }
    }
    return counter;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    // RULE 9 (the moe_grouped_s2 precedent): the pointer table and its count come back to the host - one
    // sync - and each blob stages through the shim's stream-ordered memcpy (a blit over the mapped host
    // source).  The .cu reads *n on the device so the entry could be captured; that is the one semantic
    // this restructure gives up, exactly as moe_grouped_s2 already did in the same verify window.
    cudaStreamSynchronize((cudaStream_t) stream);
    int32_t count = 0;
    if (cudaMemcpy(&count, n, sizeof(count), cudaMemcpyDeviceToHost) != cudaSuccess || count < 0 || count > cap) {
        std::fprintf(stderr, "fetch_blobs: *n = %d out of range (cap %d)\n", count, cap);
        std::exit(1);
    }
    std::vector<unsigned long long> ptrs((size_t) count, 0);
    if (count > 0 && cudaMemcpy(ptrs.data(), src, (size_t) count * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "fetch_blobs: cannot read the pointer table\n");
        std::exit(1);
    }
    for (int32_t k = 0; k < count; ++k) {
        const cudaError_t e = cudaMemcpyAsync(dst + (size_t) k * (size_t) blob_bytes, (const void*) ptrs[k],
                                              (size_t) blob_bytes, cudaMemcpyDeviceToDevice,
                                              (cudaStream_t) stream);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fetch_blobs: blob %d: %s\n", k, cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    metal::Launch k("rebase_ptrs_kernel", 1, 1, 1, 128, 1, 1, 0, stream);
    k.buf(ptr).buf(n).scalar((unsigned long long) (uintptr_t) base).scalar((long long) blob_bytes);
    k.done();
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    metal::Launch k("add_streams_broadcast_kernel", (unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok, 1,
                    256, 1, 1, 0, stream);
    k.buf(h).buf(e).buf(R).scalar((long long) n_embd).scalar((int) hc).scalar((unsigned) 256);
    k.done();
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    metal::Launch k("ident_hits_kernel", 1, 1, 1, 1024, 1, 1, 0, stream);
    k.buf(ids).scalar(n).buf(slot).buf(dst).buf(count);
    k.done();
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    metal::Launch k("mtp_select_kernel", 16, 1, 1, 256, 1, 1, 0, stream);
    k.buf(R_src).scalar((long long) R_stride).buf(ids).buf(row_dev).buf(R_dst).buf(tok_dst).buf(out)
     .scalar(j).buf(probs).buf(out_p);
    k.done();
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    // E = the widest element the row size divides into (16, 4 or 1 bytes): the .cu's template instantiation
    if (row_bytes % 16 == 0) {
        metal::Launch k16("gather_rows_16_kernel", 48 * 8, 1, 1, 256, 1, 1, 0, stream);
        k16.buf(src).scalar((long long) (row_bytes / 16)).buf(ids).scalar((long long) n).buf(dst);
        k16.done();
    } else if (row_bytes % 4 == 0) {
        metal::Launch k4("gather_rows_4_kernel", 48 * 8, 1, 1, 256, 1, 1, 0, stream);
        k4.buf(src).scalar((long long) (row_bytes / 4)).buf(ids).scalar((long long) n).buf(dst);
        k4.done();
    } else {
        metal::Launch k1("gather_rows_1_kernel", 48 * 8, 1, 1, 256, 1, 1, 0, stream);
        k1.buf(src).scalar((long long) row_bytes).buf(ids).scalar((long long) n).buf(dst);
        k1.done();
    }
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    metal::Launch k("map_ids_kernel", 1, 1, 1, 64, 1, 1, 0, stream);
    k.buf(ids).buf(table).scalar(n);
    k.done();
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    metal::Launch k("row_top_prob_kernel", (unsigned) n_rows, 1, 1, 1024, 1, 1, 0, stream);
    k.buf(logits).scalar(n_vocab).buf(ids).buf(probs).scalar((unsigned) 1024);
    k.done();
    check("row_top_prob");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    metal::Launch k("window_ids_kernel", 8, (unsigned) n, 1, 256, 1, 1, 0, stream);
    k.buf(steps).scalar(window).buf(ids).scalar((long long) ids_stride).scalar((unsigned) (8u * 256u));
    k.done();
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    metal::Launch k("dense_steps_kernel", 1, 1, 1, 64, 1, 1, 0, stream);
    k.buf(cells).scalar(n).buf(steps);
    k.done();
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch k("gdn_conv_l2_multi_kernel", (unsigned) (channels / S), (unsigned) n_tok, 1, S, 1, 1, 0,
                    stream);
    k.buf(history).buf(qkv).buf(conv_w).buf(h).scalar(channels).scalar(qk_heads).scalar(eps).scalar(t_begin);
    k.done();
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    metal::Launch k("gdn_conv_commit_kernel", (unsigned) ((channels + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(history).buf(qkv).scalar(channels).buf(n_keep);
    k.done();
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch k("gdn_ab_multi_kernel", (unsigned) ((2 * h_v + 7) / 8), 1, 1, 256, 1, 1, 0, stream);
    k.buf(x).buf(w_alpha).buf(w_beta).buf(dt).buf(ssm_a).buf(gate).buf(beta)
     .scalar(n_embd).scalar(h_v).scalar(n_tok);
    k.done();
    check("gdn_ab_multi");
}

// STRATA_METAL_GDN_TAIL=0 keeps the kernel that also stages the last token's state and copies it back (same bits)
const char* step_kernel() {
    static const char* name = [] {
        const char* v = std::getenv("STRATA_METAL_GDN_TAIL");
        return v != nullptr && std::atoi(v) == 0 ? "gdn_step_norm_multi_kernel" : "gdn_step_norm_multi_tail_kernel";
    }();
    return name;
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    metal::Launch k(step_kernel(), (unsigned) h_v, 1, 1, S, RG, 1, 0, stream);
    k.buf(state).buf(h).scalar(conv_channels).buf(gate).buf(beta).buf(z).buf(gamma).scalar(eps).buf(y)
     .scalar(h_k).scalar(h_v).scalar(n_tok).buf(n_keep).scalar(t_out_begin).buf(step_state_shadow(h_v));
    k.done();
    check("gdn_step_norm_multi");
}

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    metal::Launch kp("resident_plan_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    kp.buf(ids).scalar(n_entries).scalar(k).buf(res_layer).scalar(n_expert)
      .scalar((unsigned long long) (uintptr_t) cache_base).buf(slot_off).scalar((long long) blob)
      .buf(plan).scalar((long long) capx).buf(skip).scalar(ring);
    kp.done();
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    metal::Launch k("wait_flag_ge_or_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    k.buf(flag).scalar(value).buf(skip);
    k.done();
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    metal::Launch k("copy_i32_unless_kernel", 1, 1, 1, 128, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar((int) n).buf(skip).scalar(value).scalar((unsigned) 128);
    k.done();
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    metal::Launch k("copy_or_zero_kernel", (unsigned) blocks, 1, 1, 256, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar(n4).buf(skip).scalar(value);
    k.done();
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    metal::Launch k("wait_flag_ge_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    k.buf(flag).scalar(value);
    k.done();
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    metal::Launch k("embedding_gather_dev_kernel", (unsigned) ((n + 255) / 256), (unsigned) n_tok, 1, 256, 1, 1,
                    0, stream);
    k.buf(codes).buf(scales).buf(offsets).buf(tokens).scalar((long long) n)
     .scalar(code_bits).scalar(code_bias).scalar(group_elems)
     .scalar((unsigned long long) row_codes).scalar((unsigned long long) row_groups)
     .buf(out).scalar((unsigned) 256);
    k.done();
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    metal::Launch k("broadcast_streams_kernel", (unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok, 1,
                    256, 1, 1, 0, stream);
    k.buf(x).buf(R).scalar((long long) n_embd).scalar((int) hc).scalar((unsigned) 256);
    k.done();
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    metal::Launch k("copy_indexed_kernel", blocks, 1, 1, 256, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar((long long) stride).buf(index).scalar((long long) n);
    k.done();
    check("copy_indexed");
}

// the verify window's stage profiler: a strictly increasing device-counter value into buf[i] (see the file
// comment - MSL has no GPU clock; the differences count stamp launches, not nanoseconds)
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    metal::Launch k("gpu_stamp_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    k.buf(buf).scalar(i).buf(stamp_counter());
    k.done();
}

}  // namespace strata::kernels

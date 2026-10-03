// src/kernels/metal/kv_q4.mm - the port of src/kernels/cuda/kv_q4.cu's launchers.  Every pointer is a BOUND
// buffer argument (rule 9, docs/PORT_METAL/PROGRESS.md): each KvHostPools the CUDA kernels took by value is
// flattened into its eight pointers, bound in the struct's declaration order - k_pool, v_pool, k_q, v_q,
// k_scale, v_scale, k_q4, v_q4 - so the kernel's nullptr tests see exactly the C struct's fields.  The
// threadgroups mirror the CUDA blocks: 32 threads per 32-value block (the appends), 128 = one warp per row
// (fwht256, the gather).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"     // KvHostPools (the .cu includes it for the same definition)
#include "strata/kernels/qsa.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int Q4_THREADS = 32;               // one 32-value block per threadgroup
constexpr int Q4_WARP_GROUP = 4 * Q4_THREADS;    // four warps: one row (fwht) / one q4 block (gather) per warp

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// RULE 9: the eight KvHostPools pointers as eight bound buffers (never struct bytes); a null field binds nil.
static metal::Launch& host_pools(metal::Launch& k, const KvHostPools& p) {
    return k.buf(p.k_pool).buf(p.v_pool).buf(p.k_q).buf(p.v_q).buf(p.k_scale).buf(p.v_scale).buf(p.k_q4)
     .buf(p.v_q4);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    metal::Launch k("fwht256_kernel", (unsigned) num_blocks, 1, 1, Q4_WARP_GROUP, 1, 1, 0, stream);
    k.buf(src).buf(dst).scalar((long) n_rows).scalar(1.0f / 16.0f);
    k.done();
    check("fwht256 launch");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    need_256(s, "kv_append_q4");
    const KvHostPools pools = host ? *host : KvHostPools{};
    metal::Launch k("kv_append_q4_kernel", (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), 2,
                    Q4_THREADS, 1, 1, 0, stream);
    k.buf(k_q4).buf(v_q4).buf(page_table).buf(step).buf(kcur).buf(vcur)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size);
    host_pools(k, pools);
    k.done();
    check("kv_append_q4 launch");
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const unsigned grid[3] = {(unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0)};
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v) {
        metal::Launch k("kv_append_q4_batch_kernel", grid[0], grid[1], grid[2], Q4_THREADS, 1, 1, 0, stream);
        k.buf(k_q4).buf(v_q4).buf(page_table).buf(K).buf(V)
         .scalar((long) pos0)
         .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size).scalar(is_v);
        host_pools(k, h);
        host_pools(k, st);
        k.done();
    }
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    metal::Launch k("kv_gather_q4_kernel", num_blocks, 1, 1, Q4_WARP_GROUP, 1, 1, 0, stream);
    k.buf(k_q4).buf(v_q4).buf(page_table).buf(ids).buf(step)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size)
     .buf(k_scratch).buf(v_scratch);
    k.done();
    check("kv_gather_q4 launch");
}

}  // namespace strata::kernels

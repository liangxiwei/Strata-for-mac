// src/kernels/metal/kv_q8.mm - the port of src/kernels/cuda/kv_q8.cu's launchers (K14 part).
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"     // kv_append_step / kv_gather_step (their kernels live here until K15)
#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

}  // namespace

// RULE 9: the eight KvHostPools pointers travel as EIGHT bound buffer arguments, in the struct's declaration
// order - never as struct bytes (a pointer in setBytes data does not dereference on this GPU; a null field
// binds nil and the kernel's nullptr test sees exactly the C struct's field).
static metal::Launch& host_pools(metal::Launch& k, const KvHostPools& p) {
    return k.buf(p.k_pool).buf(p.v_pool).buf(p.k_q).buf(p.v_q).buf(p.k_scale).buf(p.v_scale).buf(p.k_q4)
     .buf(p.v_q4);
}

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const KvHostPools pools = host ? *host : KvHostPools{};
    metal::Launch k("kv_append_q8_kernel", (unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_Q8_GROUP), 2,
                    KV_Q8_GROUP, 1, 1, 0, stream);
    k.buf(k_q).buf(v_q).buf(k_scale).buf(v_scale).buf(page_table).buf(step).buf(kcur).buf(vcur)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size);
    host_pools(k, pools);
    k.done();
    check("kv_append_q8 launch");
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv * (s.head_dim / 4);
    metal::Launch k("kv_gather_q8_kernel", (unsigned) ((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(k_q).buf(v_q).buf(k_scale).buf(v_scale).buf(page_table).buf(ids).buf(step)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size)
     .buf(k_scratch).buf(v_scratch);
    k.done();
    check("kv_gather_q8 launch");
}


// ---- the FP16 KV steps (qsa.cu on CUDA; see the .metal file's note) -------------------------

namespace {
void validate_shapes(const QsaShapes& s, const char* what) {
    if (s.head_dim % 4 != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv: %s: bad shapes\n", what);
        std::exit(1);
    }
}
}  // namespace

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate_shapes(s, "kv_append");
    if (step == nullptr) {
        std::fprintf(stderr, "kv_append: step is null\n");
        std::exit(1);
    }
    const int64_t n = s.n_head_kv * s.head_dim;
    const KvHostPools pools = host ? *host : KvHostPools{};
    metal::Launch k("kv_append_kernel", (unsigned) ((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(k_pool).buf(v_pool).buf(page_table).buf(step).buf(kcur).buf(vcur)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size);
    host_pools(k, pools);
    k.done();
    check("kv_append");
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate_shapes(s, "kv_gather");
    if (step == nullptr) {
        std::fprintf(stderr, "kv_gather: step is null\n");
        std::exit(1);
    }
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv * (s.head_dim / 4);
    metal::Launch k("kv_gather_kernel", (unsigned) ((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(k_pool).buf(v_pool).buf(page_table).buf(ids).buf(step)
     .scalar((int) s.n_head_kv).scalar((int) s.head_dim).scalar((int) s.page_size)
     .buf(k_scratch).buf(v_scratch);
    k.done();
    check("kv_gather");
}

}  // namespace strata::kernels

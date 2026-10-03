// src/kernels/metal/qsa_prompt_attn.mm - the port of src/kernels/cuda/qsa_prompt_attn.cu's host half (K15).
// The dispatch contract is the CUDA file's (which pool layouts are accepted, the 65,535-query slabs); the
// kernel it launches is the f32 online-softmax rewrite qsa_prompt_attn.metal's header explains (the MMA
// originals are sm_75+/sm_80/gfx12 instructions over 38-54 KB of shared memory - neither exists here, and
// both exceed this GPU's 32,768 B threadgroup budget).  Q4_0 K (mode 2) is refused exactly as on CUDA;
// STRATA_PROMPT_ATTN_V1 / STRATA_QSA_WARP are the CUDA kernel-chooser arms and have no effect here (this
// backend has one prompt kernel).
#include "strata/kernels/qsa_prompt_attn.hpp"

#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256;    // head_dim
constexpr int G = 12;      // query heads per KV head

/// One slab of up to 65,535 queries (CUDA's gridDim.x limit; kept so the launch shapes match).
// STRATA_METAL_PROMPT_ATTN_REG=0 keeps prompt_attn_kernel_m*; prompt_attn_reg_m* compute the same bits
const char* reg_name(const char* kernel) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_METAL_PROMPT_ATTN_REG");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return kernel;
    if (std::strcmp(kernel, "prompt_attn_kernel_m0") == 0) return "prompt_attn_reg_m0";
    if (std::strcmp(kernel, "prompt_attn_kernel_m1") == 0) return "prompt_attn_reg_m1";
    if (std::strcmp(kernel, "prompt_attn_kernel_m3") == 0) return "prompt_attn_reg_m3";
    return kernel;
}

bool launch_slab(const char* kernel, const float* q, const QsaAttnPools& p, const int32_t* ids,
                 const int32_t* steps, int64_t cap, const QsaShapes& s, float* attn, int64_t n_q,
                 void* stream) {
    kernel = reg_name(kernel);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        metal::Launch k(kernel, (unsigned) nb, (unsigned) s.n_head_kv, 1, 256, 1, 1, 0, stream);
        k.buf(q + (size_t) q0 * s.n_head * HD)
         .buf(p.k_pool)
         .buf(p.v_pool)
         .buf(p.k_q)
         .buf(p.v_q)
         .buf(p.k_scale)
         .buf(p.v_scale)
         .buf(p.k_q4)
         .buf(p.v_q4)
         .buf(p.page_table)
         .buf(ids + (size_t) q0 * cap)
         .buf(steps + (size_t) q0 * kStepCount)
         .scalar((int) s.n_head_kv)
         .scalar((int) s.page_size)
         .scalar((long) cap)
         .buf(attn + (size_t) q0 * s.n_head * HD);
        k.done();
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    // no compute-capability probe here: the one Metal kernel has no sm_75/sm_80/gfx12 gate to respect
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids ||
        !steps || !pools.page_table)
        return false;
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        return launch_slab("prompt_attn_kernel_m3", q, pools, ids, steps, cap, s, attn, n_q, stream);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch_slab("prompt_attn_kernel_m1", q, pools, ids, steps, cap, s, attn, n_q, stream);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch_slab("prompt_attn_kernel_m0", q, pools, ids, steps, cap, s, attn, n_q, stream);
}

}  // namespace strata::kernels

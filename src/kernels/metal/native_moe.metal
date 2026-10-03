// src/kernels/metal/native_moe.metal - the port of src/kernels/cuda/native_moe.cu's `combine` kernel
// (the pinned ggml moe-weighted-reduction, see the .cu's license header).
//
// The arithmetic is the header contract verbatim: the FIRST product rounds to f32, every following product
// accumulates with fma in expert order, and the optional shared row is added once afterward.  nvcc contracts
// `sum += p * w` by default; this metallib builds with -ffp-contract=off (measured, docs/PORT_METAL/
// PROGRESS.md), so the fma is spelled - the chain is the CUDA one bit for bit.
//
// The token of a multi-token launch is blockIdx.y (rule 6: the group position is the data index); the column
// is blockIdx.x * blockDim.x + threadIdx.x, spelled as gpos.x * block + t with the block size a scalar
// argument (rule 7).  A null `shared` binds nil and the kernel tests its pointer (rule 4).
#include "strata_port.metalh"

kernel void nmoe_combine_kernel(constant const float* parts [[buffer(0)]],
                                constant const float* weights [[buffer(1)]],
                                constant const float* shared [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant const int& n_embd [[buffer(4)]],
                                constant const int& k [[buffer(5)]],
                                constant const uint& block [[buffer(6)]],
                                uint2 gpos [[threadgroup_position_in_grid]],   // .y = the token, 0 when single
                                uint t [[thread_index_in_threadgroup]]) {
    const ulong tk = gpos.y;
    const ulong col = (ulong) gpos.x * block + t;
    if (col >= (ulong) n_embd) return;
    constant const float* p = parts + tk * (ulong) k * (ulong) n_embd + col;
    constant const float* w = weights + tk * (ulong) k;
    constant const float* s = shared != nullptr ? shared + tk * (ulong) n_embd + col : nullptr;
    float sum = p[0] * w[0];
    for (int expert = 1; expert < k; ++expert)
        sum = fma(p[(ulong) expert * (ulong) n_embd], w[expert], sum);
    if (s != nullptr) sum += s[0];
    output[tk * (ulong) n_embd + col] = sum;
}

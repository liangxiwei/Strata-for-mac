// src/kernels/metal/native_qsa_score.metal - the port of src/kernels/cuda/native_qsa_score.cu's kernel
// (K18): the indexer's per-cell scores from the pooled keys - four heads' dots, ReLU PER HEAD, the
// documented ordered F32 head addition, the optional F32 block bias, the finite 1e9 bias on the incomplete
// tail, and the reference's +0 on full blocks.
//
// THE TENSOR-CORE PATH DOES NOT EXIST HERE.  The CUDA file's sm_80+ kernel is ldmatrix + mma.sync
// m16n8k8 over TF32 register bits; Apple GPUs have neither, and this port follows the CUDA file's OWN
// documented fallback for such a target (the gfx1100 scalar_score_kernel: "Keep the same entry point and
// score contract with an ordered scalar F32 dot for each indexer head") - the same stand-in qsa.metal's
// prompt-attention mma.sync became.  Each head's dot is the F32 FMA chain in ascending d, exactly the
// Turing fallback's arithmetic; only the rounding differs from the TF32 path (F32 inputs, not 10-bit).
#include "strata_port.metalh"

constant const int NQSS_D = 128;      // idx_dim
constant const int NQSS_HEADS = 4;    // idx_n_head

// step[kStepPos]=0, [kStepNKv]=1, [kStepNBid]=2, [kStepWidth]=3 (qsa.hpp's QsaStep index names)
kernel void nqss_score_kernel(constant const float* pooled [[buffer(0)]],
                              constant const float* query [[buffer(1)]],
                              constant const float* bias [[buffer(2)]],        // nil = no bias
                              constant const int* step [[buffer(3)]],
                              constant const int& max_cells [[buffer(4)]],
                              device float* cells [[buffer(5)]],
                              uint3 gpos [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
    const int n = step[1], full = step[2];       // kStepNKv, kStepNBid
    if (n < 1 || n > max_cells || step[0] != n - 1 || full != n / 4 ||
        step[3] != (n < 2051 ? n : 2051))
        return;
    const int row = (int) gpos.x;
    if (row > full) return;
    threadgroup float head_score[NQSS_HEADS];
    const int head = (int) t;
    if (head < NQSS_HEADS) {
        float dot = 0.0f;
        for (int d = 0; d < NQSS_D; ++d)
            dot = fma(pooled[(ulong) row * NQSS_D + d], query[(ulong) head * NQSS_D + d], dot);
        head_score[head] = dot > 0.0f ? dot : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head == 0) {
        float sum = 0.0f + head_score[0];
        sum = sum + head_score[1];
        sum = sum + head_score[2];
        sum = sum + head_score[3];
        if (bias != nullptr) sum = sum + bias[row];
        sum = sum + (row == full && n % 4 != 0 ? 1e9f : 0.0f);
        // the live causal mask is +0; invalid/padded cells are never exported
        sum = sum + 0.0f;
        for (int i = row * 4; i < n && i < (row + 1) * 4; ++i) cells[i] = sum;
    }
}

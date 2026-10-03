// src/kernels/metal/native_gdn.metal - the port of src/kernels/cuda/native_gdn.cu's kernel (K10's native
// sibling, wave 3): the pinned llama.cpp gated_delta_net recurrence, Strata state layout, one warp owning
// one (head, column) with its four state rows sharded across the lanes.
//
// The CUDA grid dim3(h_v, 1, S/4) x block dim3(32, 4) becomes MTL (h_v, S/4, 1) x 128 flat threads
// (docs/PORT_METAL/PROGRESS.md rules 6-7): blockIdx.x and blockIdx.z are DATA indices (head, column tile),
// so both are read from `uint3 [[threadgroup_position_in_grid]]` - the .cu's z axis collapses onto gpos.y.
// The block's (threadIdx.x, threadIdx.y) = (lane, row shard) is the flat thread's
// (thread_index_in_simdgroup, simdgroup_index_in_threadgroup), because CUDA linearizes dim3(32, 4) with x
// fastest - exactly how a 128-thread MSL group splits into four 32-wide simdgroups.
//
// Rule 10 (measured on this M2 Max, gdn's port): a register-heavy kernel can pin the pipeline's
// maxTotalThreadsPerThreadgroup below the dispatched group size and Metal DROPS the dispatch silently.
// The CUDA kernel's per-thread arrays are 3 x 4 floats with fully-unrolled constant-index loops - an order
// under the 32-float array that pinned fused_gdn's 512-thread kernel at 384 - and this dispatch is only
// 128 threads wide, so the arrays stay (the runtime aborts by name if a cap is ever exceeded; the probe and
// gdn_parity would show it immediately).
//
// No fp64 anywhere in the .cu - the pinned recurrence is all-float fast math - so unlike gdn.metal's norms
// there is no double accumulation to emulate; the only precision seam is expf -> metal::precise::exp
// (rule 5: the metallib builds -fno-fast-math, where the plain spelling does not exist).  The .cu's
// mul-then-add pairs compile to fused FMA under its --use_fast_math and stay UNfused here
// (-ffp-contract=off, the port's standing flags): one extra rounding per pair, ~1e-7 relative, measured by
// the /tmp double-reference probe (the fused_gdn precedent - these native_* files have no in-repo parity).
#include "strata_port.metalh"

constant const int NGDN_S = 128;         ///< the .cu's constexpr int S (state size, rows = cols)
constant const int NGDN_ROWS = 4;        ///< blockDim.y: state rows per warp shard (block dim3(32, 4))

// the .cu's warp_sum: five xor-shuffle levels over the 32 lanes
static inline float ngd_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += simd_shuffle_xor(value, (uint) offset);
    return value;
}

// ---- the pinned recurrence, one warp per (head, column): kv = state.k, delta = (v - exp(gate)*kv)*beta,
// state = exp(gate)*state + k*delta, out = state.q * scale.  Every lane owns the four state rows
// i = r*32 + lane of its column; the two warp sums reduce the shards into the column's scalars. ----
kernel void step(device float* state [[buffer(0)]],
                 constant const float* q [[buffer(1)]],
                 constant const float* k [[buffer(2)]],
                 constant const float* v [[buffer(3)]],
                 constant const float* gate [[buffer(4)]],
                 constant const float* beta [[buffer(5)]],
                 device float* output [[buffer(6)]],
                 constant const int& h_k [[buffer(7)]],
                 constant const int& h_v [[buffer(8)]],
                 constant const float& scale [[buffer(9)]],
                 uint3 gpos [[threadgroup_position_in_grid]],    // x: blockIdx.x (head), y: blockIdx.z (column tile)
                 uint lane [[thread_index_in_simdgroup]],        // threadIdx.x
                 uint sg [[simdgroup_index_in_threadgroup]]) {   // threadIdx.y, 0..3
    const int head = (int) gpos.x;
    const int col = (int) gpos.y * NGDN_ROWS + (int) sg;   // blockIdx.z * blockDim.y + threadIdx.y
    const int q_head = head % h_k;                         // MODULO head pairing (gdn.hpp's property 1)
    float s_shard[4], k_reg[4], q_reg[4];
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + (int) lane;
        s_shard[r] = state[((ulong) i * h_v + head) * NGDN_S + col];   // layout (S, h_v, S), j fastest
        k_reg[r] = k[(ulong) q_head * NGDN_S + i];
        q_reg[r] = q[(ulong) q_head * NGDN_S + i];
    }
    const float g_val = metal::precise::exp(gate[head]);
    float kv_shard = 0.0f;
    for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
    const float kv_col = ngd_warp_sum(kv_shard);
    const float delta_col = (v[(ulong) head * NGDN_S + col] - g_val * kv_col) * beta[head];
    float attn_partial = 0.0f;
    for (int r = 0; r < 4; ++r) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;       // decay BEFORE the update, in order
        attn_partial += s_shard[r] * q_reg[r];                        // readout uses the UPDATED state
    }
    const float attn_col = ngd_warp_sum(attn_partial);
    if (lane == 0u) output[(ulong) head * NGDN_S + col] = attn_col * scale;
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + (int) lane;
        state[((ulong) i * h_v + head) * NGDN_S + col] = s_shard[r];
    }
}

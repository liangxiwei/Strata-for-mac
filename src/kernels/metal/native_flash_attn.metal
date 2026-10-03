// src/kernels/metal/native_flash_attn.metal - the port of src/kernels/cuda/native_flash_attn.cu's `attend`
// kernel: llama.cpp's fattn-vec F16/F16 shape specialized to D=256, ncols=1, 128 threads (dim3(32,4), four
// warps), four float2 per load, four V columns per iteration, one online-softmax pass over the 256-cell
// padded window.
//
// Transcription rules used (all documented in docs/PORT_METAL/PROGRESS.md):
//   * the 32x4 block flattens with x fastest, so thread t's simdgroup IS its CUDA warp and the simd lane is
//     threadIdx.x; blockIdx.x (the head) is the group position (rule 6);
//   * `__shfl_xor_sync(..., mask, 8)` for mask < 8 and `warp_sum<8>` reduce the 8-lane subgroups exactly -
//     a full-simdgroup simd_shuffle_xor with the same small masks never crosses the 8-lane boundary, so the
//     width parameter disappears;
//   * `__syncwarp` is simdgroup_barrier (the tile slice each warp reads is the one it wrote), `__syncthreads`
//     is threadgroup_barrier (all 128 threads reach both);
//   * the CUDA file's OWN comment pins the arithmetic: the old-accumulator rescale contracts with the first
//     V addition (fma(rescale, old, round(V*w))), later columns are fma(V, w, acc) - those are explicit
//     intrinsics there and stay explicit fma() here; the dot accumulation `dot += a*q` and the online sum
//     `sum*rescale + score` are plain text nvcc contracts by default (fmad=true), so they are spelled fma()
//     to match the pinned binary; every other product stays a materialized multiply (the build's
//     -ffp-contract=off forbids cross-statement contraction);
//   * tile (4*4*256 = 16 KiB) plus the two 32-float arrays sit inside the measured 32 KiB threadgroup
//     budget - no device-memory restructure needed;
//   * `__half2float` is f32_from_f16, expf is metal::precise::exp, __int_as_float is as_type<float>, and
//     -FLT_MAX/2 is spelled as its literal.
#include "strata_port.metalh"

static inline float nfa_warp_sum32(float v) {
    for (uint off = 16; off; off >>= 1) v += simd_shuffle_xor(v, off);
    return v;
}
static inline float nfa_warp_max32(float v) {
    for (uint off = 16; off; off >>= 1) v = fmax(v, simd_shuffle_xor(v, off));
    return v;
}

// kStepWidth = 3, kStepNKv = 1, kStepPos = 0, kStepNBid = 2 (strata/kernels/qsa.hpp's QsaStep)
kernel void attend(constant const float* q [[buffer(0)]],
                   constant const ushort* k [[buffer(1)]],          // f16 bits
                   constant const ushort* v [[buffer(2)]],          // f16 bits
                   constant const int* step [[buffer(3)]],
                   constant const int& max_context [[buffer(4)]],
                   constant const int& padded_length [[buffer(5)]],
                   constant const float& scale [[buffer(6)]],
                   device float* out [[buffer(7)]],
                   device int* status [[buffer(8)]],
                   constant const ushort* mask [[buffer(9)]],       // nullable: binds nil (rule 4)
                   uint3 gpos [[threadgroup_position_in_grid]],     // blockIdx.x = the head
                   uint t [[thread_index_in_threadgroup]]) {
    const uint lane = t & 31u, warp = t >> 5;    // threadIdx.x / threadIdx.y of the 32x4 block
    const uint tid = t;
    const uint head = gpos.x, kv = head / 12;
    const int width = step[3], nkv = step[1];
    const bool valid = width >= 1 && width <= max_context && nkv == width &&
                       step[0] == width - 1 && step[2] == width / 4;
    if (head == 0 && tid == 0)
        *status = valid ? 0 /* kNativeFlashAttnSuccess */ : 1 /* kNativeFlashAttnUnsupportedStep */;
    if (!valid) {
        out[head * 256 + tid] = as_type<float>(0x7fc00000u);        // NaN
        out[head * 256 + tid + 128] = as_type<float>(0x7fc00000u);
        return;
    }
    float2 qreg[16];
    float2 vkq[16];
    for (int i = 0; i < 16; ++i) vkq[i] = float2(0.0f);             // CUDA's `= {}`
    threadgroup float tile[4 * 4 * 256];                            // 16 KiB, inside the 32 KiB budget
    threadgroup float max_shared[32], sum_shared[32];
    float maximum = -3.402823466e+38f / 2.0f, sum = 0.0f;
    for (uint i0 = 0; i0 < 128; i0 += 32) {
        const uint i = i0 + (lane % 8) * 4;
        for (int j = 0; j < 4; ++j) {
            const uint d = 2 * (i + (uint) j);
            qreg[i0 / 8 + (uint) j] = float2(q[head * 256 + d] * scale,
                                             q[head * 256 + d + 1] * scale);
        }
    }
    for (int base = 0; base < padded_length; base += 128) {
        float score = 0.0f, next_max = maximum;
        for (int row = 0; row < 8; ++row) {
            const int cell = base + (int) (warp * 32 + (lane & ~7u)) + row;
            float dot = 0.0f;
            for (uint i0 = 0; i0 < 128; i0 += 32) {
                const uint i = i0 + (lane % 8) * 4;
                for (int j = 0; j < 4; ++j) {
                    const uint d = 2 * (i + (uint) j);
                    const float a = cell < width ? f32_from_f16(k[(ulong) (cell * 2 + (int) kv) * 256 + d]) : 0.0f;
                    const float b = cell < width ? f32_from_f16(k[(ulong) (cell * 2 + (int) kv) * 256 + d + 1]) : 0.0f;
                    dot = fma(a, qreg[i0 / 8 + (uint) j].x, dot);   // nvcc contracts dot += a*q
                    dot = fma(b, qreg[i0 / 8 + (uint) j].y, dot);
                }
            }
            for (uint off = 4; off; off >>= 1)                      // warp_sum<8>: masks 4,2,1 stay in-lane-group
                dot += simd_shuffle_xor(dot, off);
            dot += cell < width ? (mask != nullptr ? f32_from_f16(mask[cell]) : 0.0f)
                                : as_type<float>(0xff800000u);      // -inf
            next_max = fmax(next_max, dot + (3.0f * 0.6931f));
            if (lane % 8 == (uint) row) score = dot;
        }
        for (uint offset = 8; offset < 32; offset <<= 1)
            next_max = fmax(next_max, simd_shuffle_xor(next_max, offset));
        const float rescale = metal::precise::exp(maximum - next_max);
        maximum = next_max;
        score = metal::precise::exp(score - maximum);
        sum = fma(sum, rescale, score);                             // nvcc contracts sum*rescale + score
        tile[tid] = score;
        simdgroup_barrier(mem_flags::mem_threadgroup);              // __syncwarp: the slice is warp-local
        for (int k0 = 0; k0 < 32; k0 += 4) {
            const int local = (int) (warp * 32) + k0 + (int) (lane / 8), cell = base + local;
            const float weight = tile[local];
            for (uint i0 = 0; i0 < 128; i0 += 32) {
                const uint i = i0 + (lane % 8) * 4;
                for (int j = 0; j < 4; ++j) {
                    const uint d = 2 * (i + (uint) j);
                    const float a = cell < width ? f32_from_f16(v[(ulong) (cell * 2 + (int) kv) * 256 + d]) : 0.0f;
                    const float b = cell < width ? f32_from_f16(v[(ulong) (cell * 2 + (int) kv) * 256 + d + 1]) : 0.0f;
                    // the pinned sm120a binary's own order: the rescale contracts with the FIRST addition
                    // (fma(rescale, old, round(V*w))), later columns use fma(V, w, acc)
                    if (k0 == 0) {
                        vkq[i0 / 8 + (uint) j].x = fma(rescale, vkq[i0 / 8 + (uint) j].x, a * weight);
                        vkq[i0 / 8 + (uint) j].y = fma(rescale, vkq[i0 / 8 + (uint) j].y, b * weight);
                    } else {
                        vkq[i0 / 8 + (uint) j].x = fma(a, weight, vkq[i0 / 8 + (uint) j].x);
                        vkq[i0 / 8 + (uint) j].y = fma(b, weight, vkq[i0 / 8 + (uint) j].y);
                    }
                }
            }
        }
    }
    if (warp == 0) { max_shared[lane] = -3.402823466e+38f / 2.0f; sum_shared[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) max_shared[warp] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float global_max = nfa_warp_max32(max_shared[lane]);
    const float rescale = metal::precise::exp(maximum - global_max);
    for (int i = 0; i < 16; ++i) {
        vkq[i].x = vkq[i].x * rescale;                              // __fmul_rn: a materialized multiply
        vkq[i].y = vkq[i].y * rescale;
    }
    for (uint i0 = 0; i0 < 128; i0 += 32) {
        const uint start = warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
        for (int j = 0; j < 4; ++j) {
            tile[start + 2 * (uint) j] = vkq[i0 / 8 + (uint) j].x;
            tile[start + 2 * (uint) j + 1] = vkq[i0 / 8 + (uint) j].y;
        }
    }
    sum *= rescale;
    sum = nfa_warp_sum32(sum);
    if (lane == 0) sum_shared[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = nfa_warp_sum32(sum_shared[lane]);
    for (uint i0 = 0; i0 < 256; i0 += 128) {
        float result = 0.0f;
        for (int w = 0; w < 4; ++w) {
            for (int group = 0; group < 4; ++group)
                result += tile[(uint) w * 4 * 256 + (uint) group * 256 + i0 + tid];
        }
        out[head * 256 + i0 + tid] = result / sum;
    }
}

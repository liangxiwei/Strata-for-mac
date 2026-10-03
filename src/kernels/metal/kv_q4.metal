// src/kernels/metal/kv_q4.metal - the port of src/kernels/cuda/kv_q4.cu: Q4_0 KV storage with the
// Walsh-Hadamard rotation (PR #21).  The fwht256 is one WARP per row (8 values per lane in registers, the
// butterfly across the simdgroup); the appends are one 32-thread threadgroup per 32-value block of one KV
// head of K (z=0) or V (z=1); the gather is one warp per (id, head, block).
//
// RULE 9 (docs/PORT_METAL/PROGRESS.md): no pointer ever travels as setBytes bytes - a pointer stored in
// bytes data does not dereference on this GPU.  kv_stream.hpp's KvHostPools (eight device pointers) is
// FLATTENED into eight bound [[buffer(N)]] arguments in the struct's declaration order (k_pool, v_pool, k_q,
// v_q, k_scale, v_scale, k_q4, v_q4), exactly as kv_q8.metal does it; a null field binds nil and the kernel's
// nullptr test sees the struct's own field.  The batch append takes TWO pools (host + stage) - 16 buffers.
//
// block_q4_0 is 18 PACKED bytes (kv_q4.hpp), so a block's fp16 scale sits at an UNALIGNED offset: the scale
// is stored/read as its two bytes rather than through a uint16_t* cast (MSL gives no alignment guarantee).
#include "strata_port.metalh"

constant const int QK4_0 = 32;              // kv_q4.hpp
constant const int Q4_BLOCK_BYTES = 18;     // sizeof(kv_q4.hpp's #pragma-pack(1) block_q4_0): 2 + 16, no pad
constant const int Q4_BLOCKS_PER_HEAD = 8;  // head_dim 256 / 32 (need_256() pins head_dim = 256)
constant const int Q4_ROWS_PER_GROUP = 4;   // warps (rows) per fwht256/gather threadgroup

// include/strata/kernels/qsa.hpp's step slots (kStepPos = 0, kStepWidth = 3)
constant const int kStepPos = 0;
constant const int kStepWidth = 3;

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.  Orthonormal
// (scale 1/sqrt(256) = 1/16), so it is its own inverse.  The CUDA block is (32, rows_per_block); the MSL
// threadgroup is 128 threads and warp `sg` carries row `gpos * rows_per_group + sg` - the same warp-per-row
// mapping through threadgroup_position_in_grid + simdgroup_index_in_threadgroup (R6/R7's standing pattern).
kernel void fwht256_kernel(constant const float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant const long& n_rows [[buffer(2)]],
                           constant const float& scale [[buffer(3)]],
                           uint gpos [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;     // 8

    const long r = (long) gpos * Q4_ROWS_PER_GROUP + (long) sg;
    if (r >= n_rows) return;

    constant const float* row_src = src + (ulong) r * N;
    device float* row_dst = dst + (ulong) r * N;

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, (uint) h);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15).  Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one).  Ties in |x| resolve to the larger value in EVERY
// lane, so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two
// scales).  __float2int_rz is the C float->int cast: truncation toward zero, both here and in MSL.
static inline uint q4_group(float x, thread uint8_t& byte, uint lane) {
    float amax = metal::precise::fabs(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = simd_shuffle_xor(amax, (uint) o);
        const float v = simd_shuffle_xor(mval, (uint) o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int) (x * id + 8.5f);
    const uint qc = (uint) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint qhi = simd_shuffle_down(qc, 16u);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

// One block of one head-row: 8 blocks x 18 B per head; lane 0 writes the two scale bytes, lanes < 16 the
// packed code bytes.  Byte stores, never a uint16_t* - the packed block's scale is UNALIGNED.
static inline void q4_store(device uint8_t* pool, ulong row, int b, uint lane, uint d_bits, uint8_t byte) {
    device uint8_t* blk = pool + (row * (ulong) Q4_BLOCKS_PER_HEAD + (ulong) b) * (ulong) Q4_BLOCK_BYTES;
    if (lane == 0) {
        blk[0] = (uint8_t) (d_bits & 0xffu);
        blk[1] = (uint8_t) (d_bits >> 8);
    }
    if (lane < 16) blk[2 + lane] = byte;
}

// One block = one 32-value group of one KV head of K (z=0) or V (1); 32 threads. KV streaming: the VRAM
// page only if the block is resident (table >= 0), the host copy always (identity layout) when there is one.
kernel void kv_append_q4_kernel(device uint8_t* k_q4 [[buffer(0)]],
                                device uint8_t* v_q4 [[buffer(1)]],
                                constant const int* table [[buffer(2)]],
                                constant const int* step [[buffer(3)]],
                                constant const float* kcur [[buffer(4)]],
                                constant const float* vcur [[buffer(5)]],
                                constant const int& kv_heads [[buffer(6)]],
                                constant const int& head_dim [[buffer(7)]],
                                constant const int& page_size [[buffer(8)]],
                                device uint16_t* host_k_pool [[buffer(9)]],   // KvHostPools flattened (rule 9);
                                device uint16_t* host_v_pool [[buffer(10)]],  // only the q4 pair is read here
                                device int8_t* host_k_q [[buffer(11)]],
                                device int8_t* host_v_q [[buffer(12)]],
                                device uint16_t* host_k_scale [[buffer(13)]],
                                device uint16_t* host_v_scale [[buffer(14)]],
                                device uint8_t* host_k_q4 [[buffer(15)]],
                                device uint8_t* host_v_q4 [[buffer(16)]],
                                uint3 gid [[threadgroup_position_in_grid]],  // (head, block, K/V) are GROUP indices
                                uint t [[thread_index_in_threadgroup]]) {
    const long pos = (long) step[kStepPos];
    const int h = (int) gid.x, b = (int) gid.y;
    const bool is_v = gid.z == 1;
    const float x = (is_v ? vcur : kcur)[(ulong) h * head_dim + (ulong) b * QK4_0 + t];
    uint8_t byte;
    const uint d = q4_group(x, byte, t);
    const long page = (long) table[(ulong) pos / (ulong) page_size];
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, (ulong) ((page * kv_heads + h) * page_size + (pos % page_size)), b, t, d, byte);
    if (host_k_q4 != nullptr)
        q4_store(is_v ? host_v_q4 : host_k_q4,
                 (ulong) (((pos / page_size) * kv_heads + h) * page_size + (pos % page_size)), b, t, d, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when
// given.  Two flattened KvHostPools (host + stage) ride as 16 bound buffers - the struct-with-pointers idiom
// of the CUDA original does not survive rule 9.
kernel void kv_append_q4_batch_kernel(device uint8_t* k_q4 [[buffer(0)]],
                                      device uint8_t* v_q4 [[buffer(1)]],
                                      constant const int* table [[buffer(2)]],
                                      constant const float* K [[buffer(3)]],
                                      constant const float* V [[buffer(4)]],
                                      constant const long& pos0 [[buffer(5)]],
                                      constant const int& kv_heads [[buffer(6)]],
                                      constant const int& head_dim [[buffer(7)]],
                                      constant const int& page_size [[buffer(8)]],
                                      constant const int& is_v_grid [[buffer(9)]],
                                      device uint16_t* host_k_pool [[buffer(10)]],   // KvHostPools host, flattened
                                      device uint16_t* host_v_pool [[buffer(11)]],
                                      device int8_t* host_k_q [[buffer(12)]],
                                      device int8_t* host_v_q [[buffer(13)]],
                                      device uint16_t* host_k_scale [[buffer(14)]],
                                      device uint16_t* host_v_scale [[buffer(15)]],
                                      device uint8_t* host_k_q4 [[buffer(16)]],
                                      device uint8_t* host_v_q4 [[buffer(17)]],
                                      device uint16_t* stage_k_pool [[buffer(18)]],  // KvHostPools stage, flattened
                                      device uint16_t* stage_v_pool [[buffer(19)]],
                                      device int8_t* stage_k_q [[buffer(20)]],
                                      device int8_t* stage_v_q [[buffer(21)]],
                                      device uint16_t* stage_k_scale [[buffer(22)]],
                                      device uint16_t* stage_v_scale [[buffer(23)]],
                                      device uint8_t* stage_k_q4 [[buffer(24)]],
                                      device uint8_t* stage_v_q4 [[buffer(25)]],
                                      uint3 gpos [[threadgroup_position_in_grid]],  // (token, head, block)
                                      uint th [[thread_index_in_threadgroup]]) {
    const long t = (long) gpos.x;
    const long pos = pos0 + t;
    const int h = (int) gpos.y, b = (int) gpos.z;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[(ulong) t * (ulong) (kv_heads * head_dim) + (ulong) h * head_dim +
                                  (ulong) b * QK4_0 + th];
    uint8_t byte;
    const uint d = q4_group(x, byte, th);
    const long page = (long) table[(ulong) pos / (ulong) page_size];
    const ulong row_id = (ulong) (((pos / page_size) * kv_heads + h) * page_size + (pos % page_size));
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, (ulong) ((page * kv_heads + h) * page_size + (pos % page_size)), b, th, d, byte);
    if (host_k_q4 != nullptr) q4_store(is_v ? host_v_q4 : host_k_q4, row_id, b, th, d, byte);
    if (stage_k_q4 != nullptr) q4_store(is_v ? stage_v_q4 : stage_k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths).  One warp per
// (id, head, block): the CUDA block is (32, 4) and warp `y` carries block-index gpos * 4 + y.
kernel void kv_gather_q4_kernel(device const uint8_t* k_q4 [[buffer(0)]],
                                device const uint8_t* v_q4 [[buffer(1)]],
                                constant const int* table [[buffer(2)]],
                                constant const int* ids [[buffer(3)]],
                                constant const int* step [[buffer(4)]],
                                constant const int& kv_heads [[buffer(5)]],
                                constant const int& head_dim [[buffer(6)]],
                                constant const int& page_size [[buffer(7)]],
                                device uint* k_scratch [[buffer(8)]],       // fp16 patterns, ushort views below
                                device uint* v_scratch [[buffer(9)]],
                                uint gpos [[threadgroup_position_in_grid]],
                                uint t [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const long n_ids = (long) step[kStepWidth];
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * Q4_BLOCK_BYTES;              // 144
    const long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long blk_idx = (long) gpos * Q4_ROWS_PER_GROUP + (long) sg;
    if (blk_idx >= total_blocks) return;

    const long id = blk_idx / (kv_heads * (long) blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * (long) blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long page = (long) table[(ulong) cell / (ulong) page_size];
    const long row = (page * kv_heads + h) * page_size + (cell % page_size);

    device const uint8_t* k_blk = k_q4 + (ulong) row * bytes_per_head + (ulong) b * Q4_BLOCK_BYTES;
    device const uint8_t* v_blk = v_q4 + (ulong) row * bytes_per_head + (ulong) b * Q4_BLOCK_BYTES;
    const float kd = f32_from_f16((uint) k_blk[0] | ((uint) k_blk[1] << 8));  // the packed, UNALIGNED scale
    const float vd = f32_from_f16((uint) v_blk[0] | ((uint) v_blk[1] << 8));
    const int j = t < 16 ? (int) t : (int) t - 16;
    const uint8_t k_byte = k_blk[2 + j];
    const uint8_t v_byte = v_blk[2 + j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const long dst_offset = (id * kv_heads + h) * (long) head_dim + (b * QK4_0 + (long) t);
    reinterpret_cast<device uint16_t*>(k_scratch)[dst_offset] = (uint16_t) f16_from_f32((float) kq * kd);
    reinterpret_cast<device uint16_t*>(v_scratch)[dst_offset] = (uint16_t) f16_from_f32((float) vq * vd);
}

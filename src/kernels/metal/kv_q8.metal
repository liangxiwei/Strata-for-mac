// src/kernels/metal/kv_q8.metal - the port of src/kernels/cuda/kv_q8.cu (K14 part).  The append is one
// threadgroup per 64-value group of one KV head of K (z=0) or V (z=1), 64 threads, one value each; the
// gather is one thread per 4 consecutive values.
//
// RULE 9 (docs/PORT_METAL/PROGRESS.md): a pointer stored in setBytes data is NOT a usable device pointer on
// this GPU - the bytes arrive verbatim but the first dereference reads 0.0 and every write through it
// vanishes.  The original port rode kv_stream.hpp's KvHostPools (eight device pointers) through one setBytes
// argument; it only ever "worked" because kv_q8_parity passes host=null.  The struct is now FLATTENED: every
// pool pointer is its own bound [[buffer(N)]] argument, in the struct's declaration order (k_pool, v_pool,
// k_q, v_q, k_scale, v_scale, k_q4, v_q4), a null field binding nil (the kernels test the fields they own).
#include "strata_port.metalh"

constant const int KV_Q8_GROUP = 64;

// include/strata/kernels/qsa.hpp's step slots (kStepPos = 0, kStepWidth = 3)
constant const int kStepPos = 0;
constant const int kStepWidth = 3;

kernel void kv_append_q8_kernel(device int8_t* k_q [[buffer(0)]],
                                device int8_t* v_q [[buffer(1)]],
                                device uint* k_scale [[buffer(2)]],
                                device uint* v_scale [[buffer(3)]],
                                constant const int* table [[buffer(4)]],
                                constant const int* step [[buffer(5)]],
                                constant const float* kcur [[buffer(6)]],
                                constant const float* vcur [[buffer(7)]],
                                constant const int& kv_heads [[buffer(8)]],
                                constant const int& head_dim [[buffer(9)]],
                                constant const int& page_size [[buffer(10)]],
                                device uint* host_k_pool [[buffer(11)]],      // KvHostPools flattened (rule 9);
                                device uint* host_v_pool [[buffer(12)]],      // nil fields bind nil, and only the
                                device int8_t* host_k_q [[buffer(13)]],       // q8 fields are read here
                                device int8_t* host_v_q [[buffer(14)]],
                                device uint* host_k_scale [[buffer(15)]],
                                device uint* host_v_scale [[buffer(16)]],
                                device uint8_t* host_k_q4 [[buffer(17)]],
                                device uint8_t* host_v_q4 [[buffer(18)]],
                                uint3 gid [[threadgroup_position_in_grid]],   // (head, group, K/V) are GROUP indices
                                uint t [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const long pos = (long) step[kStepPos];
    const int h = (int) gid.x, g = (int) gid.y;
    const bool is_v = gid.z == 1;
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[(ulong) h * head_dim + (ulong) g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two simdgroups, then combine through shared memory in a fixed order
    float a = metal::precise::fabs(x);
    for (int o = 16; o > 0; o >>= 1) a = metal::precise::fmax(a, simd_shuffle_xor(a, o));
    threadgroup float warp_max[2];
    if (lane == 0) warp_max[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = metal::precise::fmax(warp_max[0], warp_max[1]);
    const uint sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);                      // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = (int) metal::precise::rint(x / sf);                // __float2int_rn
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long page = (long) table[(ulong) pos / (ulong) page_size];
    if (page >= 0) {
        const long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[(ulong) row * head_dim + (ulong) g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) reinterpret_cast<device uint16_t*>(is_v ? v_scale : k_scale)[(ulong) row * groups + g] =
            (uint16_t) sbits;
    }
    if (host_k_q != nullptr) {
        const long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host_v_q : host_k_q)[(ulong) row * head_dim + (ulong) g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) reinterpret_cast<device uint16_t*>(is_v ? host_v_scale : host_k_scale)
                        [(ulong) row * groups + g] = (uint16_t) sbits;
    }
}

kernel void kv_gather_q8_kernel(constant const int8_t* k_q [[buffer(0)]],
                                constant const int8_t* v_q [[buffer(1)]],
                                constant const uint* k_scale [[buffer(2)]],
                                constant const uint* v_scale [[buffer(3)]],
                                constant const int* table [[buffer(4)]],
                                constant const int* ids [[buffer(5)]],
                                constant const int* step [[buffer(6)]],
                                constant const int& kv_heads [[buffer(7)]],
                                constant const int& head_dim [[buffer(8)]],
                                constant const int& page_size [[buffer(9)]],
                                device uint* k_scratch [[buffer(10)]],     // fp16 patterns, ushort views
                                device uint* v_scratch [[buffer(11)]],
                                uint i_in [[thread_position_in_grid]]) {
    const long n_ids = (long) step[kStepWidth];
    const int per = head_dim / 4;
    const long total = n_ids * kv_heads * per;
    const long i = (long) i_in;
    if (i >= total) return;
    const long id = i / (kv_heads * (long) per);
    const int rem = (int) (i % (kv_heads * (long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long page = (long) table[(ulong) cell / (ulong) page_size];
    const long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    constant const uint16_t* ks16 = reinterpret_cast<constant const uint16_t*>(k_scale);
    constant const uint16_t* vs16 = reinterpret_cast<constant const uint16_t*>(v_scale);
    const float ks = f32_from_f16(ks16[(ulong) row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(vs16[(ulong) row * groups + d / KV_Q8_GROUP]);
    constant const int8_t* krow = k_q + (ulong) row * head_dim;
    constant const int8_t* vrow = v_q + (ulong) row * head_dim;
    device uint16_t* kout = reinterpret_cast<device uint16_t*>(k_scratch);
    device uint16_t* vout = reinterpret_cast<device uint16_t*>(v_scratch);
    const long dst = (id * kv_heads + h) * (long) per + q4;
    for (int e = 0; e < 4; ++e) {
        kout[dst * 4 + e] = (uint16_t) f16_from_f32((float) krow[d + e] * ks);
        vout[dst * 4 + e] = (uint16_t) f16_from_f32((float) vrow[d + e] * vs);
    }
}

// ---- the FP16 KV steps, ported from src/kernels/cuda/qsa.cu (their parity rides kv_q8_parity; they move
// into the qsa port's files when K15 lands, exactly as they sit in qsa.cu on CUDA).

kernel void kv_append_kernel(device uint* k_pool [[buffer(0)]],      // fp16 patterns
                             device uint* v_pool [[buffer(1)]],
                             constant const int* table [[buffer(2)]],
                             constant const int* step [[buffer(3)]],
                             constant const float* kcur [[buffer(4)]],
                             constant const float* vcur [[buffer(5)]],
                             constant const int& kv_heads [[buffer(6)]],
                             constant const int& head_dim [[buffer(7)]],
                             constant const int& page_size [[buffer(8)]],
                             device uint* host_k_pool [[buffer(9)]],    // KvHostPools flattened (rule 9); the
                             device uint* host_v_pool [[buffer(10)]],   // fp16 pair is the one read here
                             device int8_t* host_k_q [[buffer(11)]],
                             device int8_t* host_v_q [[buffer(12)]],
                             device uint* host_k_scale [[buffer(13)]],
                             device uint* host_v_scale [[buffer(14)]],
                             device uint8_t* host_k_q4 [[buffer(15)]],
                             device uint8_t* host_v_q4 [[buffer(16)]],
                             uint i [[thread_position_in_grid]]) {
    const long pos = (long) step[kStepPos];
    if (i >= (uint) (kv_heads * head_dim)) return;
    const int h = (int) i / head_dim, d = (int) i - h * head_dim;
    device uint16_t* kp = reinterpret_cast<device uint16_t*>(k_pool);
    device uint16_t* vp = reinterpret_cast<device uint16_t*>(v_pool);
    const long page = (long) table[(ulong) pos / (ulong) page_size];
    if (page >= 0) {
        const long row = (page * kv_heads + h) * page_size + (pos % page_size);
        kp[(ulong) row * head_dim + d] = (uint16_t) f16_from_f32(kcur[i]);
        vp[(ulong) row * head_dim + d] = (uint16_t) f16_from_f32(vcur[i]);
    }
    if (host_k_pool != nullptr) {
        const long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        reinterpret_cast<device uint16_t*>(host_k_pool)[(ulong) row * head_dim + d] =
            (uint16_t) f16_from_f32(kcur[i]);
        reinterpret_cast<device uint16_t*>(host_v_pool)[(ulong) row * head_dim + d] =
            (uint16_t) f16_from_f32(vcur[i]);
    }
}

kernel void kv_gather_kernel(constant const uint* k_pool [[buffer(0)]],
                             constant const uint* v_pool [[buffer(1)]],
                             constant const int* table [[buffer(2)]],
                             constant const int* ids [[buffer(3)]],
                             constant const int* step [[buffer(4)]],
                             constant const int& kv_heads [[buffer(5)]],
                             constant const int& head_dim [[buffer(6)]],
                             constant const int& page_size [[buffer(7)]],
                             device uint* k_scratch [[buffer(8)]],
                             device uint* v_scratch [[buffer(9)]],
                             uint i_in [[thread_position_in_grid]]) {
    const long n_ids = (long) step[kStepWidth];
    const int per = head_dim / 4;                       // 4 halfs per gather thread
    const long total = n_ids * kv_heads * per;
    const long i = (long) i_in;
    if (i >= total) return;
    const long id = i / (kv_heads * (long) per);
    const int rem = (int) (i % (kv_heads * (long) per));
    const int h = rem / per, q = rem - h * per;
    const int cell = ids[id];
    const long page = (long) table[(ulong) cell / (ulong) page_size];
    const long src = ((page * kv_heads + h) * page_size + (cell % page_size)) * (long) per + q;
    const long dst = (id * kv_heads + h) * (long) per + q;
    constant const uint16_t* ksrc16 = reinterpret_cast<constant const uint16_t*>(k_pool);
    constant const uint16_t* vsrc16 = reinterpret_cast<constant const uint16_t*>(v_pool);
    device uint16_t* kdst16 = reinterpret_cast<device uint16_t*>(k_scratch);
    device uint16_t* vdst16 = reinterpret_cast<device uint16_t*>(v_scratch);
    for (int e = 0; e < 4; ++e) {
        kdst16[dst * 4 + e] = ksrc16[src * 4 + e];
        vdst16[dst * 4 + e] = vsrc16[src * 4 + e];
    }
}

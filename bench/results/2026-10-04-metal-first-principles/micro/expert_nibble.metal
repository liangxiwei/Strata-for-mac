// A lossless cache of signed IQ2 grid values. Original scale and integer arithmetic stay intact.
#include "iq_kernels.metal"

struct NibbleBlock {
    half d;
    uchar ls[8];
    ushort pad;
    uchar qs[128];
};

inline uchar value_code(int v) {
    const int a = abs(v);
    return uchar((a == 8 ? 0 : a == 25 ? 1 : 2) | (v < 0 ? 4 : 0));
}
inline int code_value(uint v) {
    const int m = int(v & 3);
    const int a = 8 + 17 * m + int(m == 2);
    return v & 4 ? -a : a;
}

template<int TY>
inline void expand_nibbles(constant const uint8_t* w, device NibbleBlock* dst,
                           device atomic_uint* bad, uint block, uint segment) {
    const auto r = iqk_Split<TY>::load(w, int(block), int(segment * 2));
    const thread int8_t* bytes = reinterpret_cast<const thread int8_t*>(r.g);
    for (int i = 0; i < 16; ++i) {
        const int a = int(bytes[2 * i]), b = int(bytes[2 * i + 1]);
        if ((abs(a) != 8 && abs(a) != 25 && abs(a) != 43) || (abs(b) != 8 && abs(b) != 25 && abs(b) != 43))
            atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
        dst[block].qs[segment * 16 + i] = value_code(a) | (value_code(b) << 4);
    }
    if (segment == 0) { dst[block].d = half(r.dw); dst[block].pad = 0; }
    if constexpr (TY == 22) dst[block].ls[segment] = uchar(r.ls0 | (r.ls1 << 4));
    else dst[block].ls[segment] = uchar(r.ls);
}

#define EXP_NIBBLE(TY) \
kernel void expand_nibble_##TY(constant const uint8_t* w [[buffer(0)]], device NibbleBlock* dst [[buffer(1)]], \
                               device atomic_uint* bad [[buffer(2)]], constant uint& nb [[buffer(3)]], \
                               uint gid [[thread_position_in_grid]]) { \
    if (gid < nb * 8) expand_nibbles<TY>(w, dst, bad, gid / 8, gid % 8); }
EXP_NIBBLE(22)
EXP_NIBBLE(16)

template<int TY>
inline float nibble_dot(constant const uint8_t* wr, constant const block_q8_1* x, int nb, int lane) {
    constant const NibbleBlock* b = reinterpret_cast<constant const NibbleBlock*>(wr);
    float s = 0.0f;
    for (int kk = lane; kk < nb * 8; kk += 32) {
        const int kbx = kk / 8, segment = kk % 8;
        typename iqk_Split<TY>::W r;
        constant const ushort* q = reinterpret_cast<constant const ushort*>(b[kbx].qs + segment * 16);
        for (int j = 0; j < 8; ++j) {
            const uint v = q[j];
            uint packed = 0;
            for (int i = 0; i < 4; ++i) packed |= (uint(code_value((v >> (4 * i)) & 7)) & 255u) << (8 * i);
            r.g[j] = as_type<int>(packed);
        }
        r.dw = float(b[kbx].d);
        if constexpr (TY == 22) { r.ls0 = b[kbx].ls[segment] & 15; r.ls1 = b[kbx].ls[segment] >> 4; }
        else r.ls = int(b[kbx].ls[segment]);
        s += iqk_Split<TY>::apply(r, x + kbx * 8, segment * 2);
    }
    return iqk_warp_sum(s);
}

template<int TY>
inline void cached_resident(constant const uint8_t* arena, constant const ulong* offsets,
                             constant const int* ids, constant const int* residency,
                             constant const block_q8_1* xq, long n_embd, long n_ff, ulong row_bytes,
                             ulong weight_offset, ulong slot_bytes, int n_expert, int k, int has_offsets,
                             device float* out, device float* up, uint3 gp, uint tid) {
    const int e = int(gp.y), row = int(gp.x) * 8 + int(tid >> 5), lane = int(tid & 31);
    if (row >= 2 * n_ff) return;
    const int id = ids[e], slot = id >= 0 && id < n_expert ? residency[id] : -1;
    const bool is_up = row >= n_ff;
    const int r = is_up ? row - int(n_ff) : row;
    float value = 0;
    if (slot >= 0) {
        const ulong off = has_offsets ? offsets[slot] : ulong(slot) * slot_bytes;
        constant const uint8_t* wr = arena + off + (is_up ? weight_offset : 0) + ulong(r) * row_bytes;
        value = nibble_dot<TY>(wr, xq + size_t(e / k) * (n_embd / 32), int(n_embd / 256), lane);
    }
    if (lane == 0) (is_up ? up : out)[size_t(e) * n_ff + r] = value;
}

#define CACHED_NIBBLE(TY) \
kernel void cached_nibble_gu_##TY(IQK_RESIDENT_PARAMS, device float* up [[buffer(14)]], \
                                 uint3 gp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    cached_resident<TY>(arena, offsets, ids, residency, xq, n_embd, n_ff, row_bytes, weight_offset, slot_bytes, \
                         n_expert, k, has_offsets, out, up, gp, tid); }
CACHED_NIBBLE(22)
CACHED_NIBBLE(16)

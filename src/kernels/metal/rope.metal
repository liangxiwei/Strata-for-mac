// src/kernels/metal/rope.metal - the port of src/kernels/cuda/rope.cu's kernel (K7).  The frequency tables
// are built on the HOST in float64 (the CUDA file's own rule - reproducing libm's pow/cos on device is
// neither fast nor guaranteed to agree), so this file is a pure rotation.  One thread per row, as the CUDA
// file insists (its first version's per-pair threads raced the tail copy).
#include "strata_port.metalh"

// rope.hpp's rope_neox_pair, transcribed (the pairing convention is the thing a transcription gets wrong -
// the CUDA header says so - so this is copied from its two lines verbatim).
static inline void rope_neox_pair(float a, float b, float c, float s, thread float& oa, thread float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

// mrope.hpp's mrope_pos (CUDA-only there): sections {11, 11, 10, 0} collapse to pair % 3 for pairs 0..31.
static inline int mrope_pos(constant const int* tab, int pos, int pair) {
    return tab != nullptr ? tab[(ulong) pos * 3 + (uint) pair % 3] : pos;
}

kernel void rope_neox_kernel(constant const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant const long& rows [[buffer(2)]],
                             constant const int& head_dim [[buffer(3)]],
                             constant const int& n_rot [[buffer(4)]],
                             constant const float* cos_tab [[buffer(5)]],
                             constant const float* sin_tab [[buffer(6)]],
                             constant const int* pos [[buffer(7)]],
                             constant const int* mtab [[buffer(8)]],     // may bind null: no image path
                             uint r [[thread_position_in_grid]]) {
    if (r >= (uint) rows) return;
    const int half_n = n_rot / 2;   // not 'half': that is MSL's fp16 type name
    constant const float* xr = x + (ulong) r * head_dim;
    device float* orow = out + (ulong) r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
    for (int i = 0; i < half_n; ++i) {
        const ulong toff = (ulong) mrope_pos(mtab, pos[r], i) * half_n;
        float oa, ob;
        rope_neox_pair(xr[i], xr[half_n + i], cos_tab[toff + i], sin_tab[toff + i], oa, ob);
        orow[i] = oa;
        orow[half_n + i] = ob;
    }
}

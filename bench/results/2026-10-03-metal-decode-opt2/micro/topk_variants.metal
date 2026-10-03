// block_topk variants. The production kernel is included verbatim (qsa_select.metal). The variant keeps the
// selection's definition - the radix threshold, the equal-key budget handed out lowest block first, the
// output in ascending block order - and computes the same integers with parallel scans.
#include "qsa_select.metal"

kernel void block_topk_v2(constant const float* scores [[buffer(0)]],
                          constant const int* steps [[buffer(1)]],
                          constant const long& max_blocks [[buffer(2)]],
                          constant const long& cap [[buffer(3)]],
                          device int* ids [[buffer(4)]],
                          uint gpos [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_int hist[8][256];       // one histogram per simdgroup: less atomic contention
    threadgroup int tot[256];
    threadgroup int part[8];
    threadgroup int s_digit;
    threadgroup atomic_int s_max;
    const long qi = (long) gpos;
    constant const int* st = steps + (ulong) qi * 4;
    const long n_kv = (long) st[1], n_bid = (long) st[2], width = (long) st[3];
    device int* out = ids + (ulong) qi * cap;
    if (n_kv <= width) {
        for (long j = (long) t; j < n_kv; j += QS_TOPK_T) out[j] = (int) j;
        return;
    }
    constant const float* sc = scores + (ulong) qi * max_blocks;
    const long nb = n_bid + 1;
    uint prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = (int) t; i < 8 * 256; i += QS_TOPK_T)
            atomic_store_explicit(&hist[i / 256][i % 256], 0, memory_order_relaxed);
        if (t == 0) atomic_store_explicit(&s_max, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (long b = (long) t; b < nb; b += QS_TOPK_T) {      // a count does not depend on who counts
            const int w = qs_block_weight(b, n_bid, n_kv);
            if (w == 0) continue;
            const uint k = qs_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask))
                atomic_fetch_add_explicit(&hist[sg][(k >> shift) & 255], w, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // digit d = 255 - t: its count, and the inclusive sum of the counts of all digits >= d
        const int d = 255 - (int) t;
        int c = 0;
        for (int g = 0; g < 8; ++g) c += atomic_load_explicit(&hist[g][d], memory_order_relaxed);
        tot[d] = c;
        const int incl = simd_prefix_inclusive_sum(c);
        if (lane == 31) part[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int off = 0;
        for (int g = 0; g < (int) sg; ++g) off += part[g];
        const int ge = above + off + incl;                   // cells with key digit >= d (above the prefix)
        // the original walks d = 255 .. 1 and stops at the first d with cum + hist[d] >= width: the largest
        // such d, else 0 - and the cells strictly above it are ge - hist[d]
        if (d > 0 && ge >= width) atomic_fetch_max_explicit(&s_max, d, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int dsel = atomic_load_explicit(&s_max, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // recompute ge for the chosen digit: above + sum of tot[e] for e > dsel
        if (t == 0) s_digit = dsel;
        int gt_part = 0;
        if (d > dsel) gt_part = c;
        const int gsum = simd_sum(gt_part);
        if (lane == 0) part[sg] = gsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int add = 0;
        for (int g = 0; g < 8; ++g) add += part[g];
        prefix |= (uint) s_digit << shift;
        above += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix;
    const long eq_budget = width - above;
    // contiguous ranges for the emission, as in the original (ascending block order across threads)
    const long per = (nb + QS_TOPK_T - 1) / QS_TOPK_T;
    const long b0 = (long) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    int gt = 0, eq = 0;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    // exclusive prefix of eq over threads, then of the selected counts
    const int eq_incl = simd_prefix_inclusive_sum(eq);
    if (lane == 31) part[sg] = eq_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int eo = 0;
    for (int g = 0; g < (int) sg; ++g) eo += part[g];
    const long eq_before = eo + eq_incl - eq;
    long my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int sel_incl = simd_prefix_inclusive_sum(sel);
    if (lane == 31) part[sg] = sel_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int so = 0;
    for (int g = 0; g < (int) sg; ++g) so += part[g];
    long wpos = so + sel_incl - sel;
    long eq_left = my_eq;
    for (long b = b0; b < b1; ++b) {
        const int w = qs_block_weight(b, n_bid, n_kv);
        if (w == 0) continue;
        const uint k = qs_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int) (b * QS_R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int) (b * QS_R + c);
        }
    }
    (void) tot;
}

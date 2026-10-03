// src/kernels/metal/sampler.metal - the port of src/kernels/cuda/sampler.cu's kernels (K8): the sampler
// chain, penalties -> top_k -> top_p -> min_p -> temperature -> pick, in llama.cpp's order (issue #53 - the
// order is the whole content of that file).  Three sampled implementations pick the same token, bit for bit,
// exactly as the CUDA file arranges: the SPLIT top_k (default), the one-block kernel
// (STRATA_SAMPLER_ONE_BLOCK=1, and the fallback when the split cannot run), and sampler_kernel
// (STRATA_OLD_SAMPLER=1), engine 0.1.20's kernel kept as the reference.
//
// The port's three restructurings, each forced by this GPU and each value-preserving:
//   * THE PENALTY BITMAP LIVES IN DEVICE MEMORY, NOT THE THREADGROUP.  The CUDA kernels size a shared bitmap
//     of n_vocab bits as dynamic shared memory - 32,768 bytes at a 262,144-wide vocabulary, already the
//     whole of this GPU's measured threadgroup budget (32,768, docs/PORT_METAL/PROGRESS.md), before the
//     selection arrays the one-block kernel adds on top.  The .mm launcher stages it in a permanent device
//     buffer (a ring, so launches in flight never share one) and the kernel builds and reads it exactly as
//     before: zero, relaxed device atomics, a mem_device barrier, then the same O(1) membership test.  Only
//     sampler_split_part keeps its bitmap in the threadgroup - it covers one 4,096-logit block (512 bytes),
//     as in CUDA.
//   * THE SPLIT MERGE READS THE BLOCK LISTS WHERE THEY ARE.  The CUDA merge stages up to 64 x 64 int2
//     (32 KB) of lists in shared memory - the whole budget; coupled_merge stacks the selection and exp
//     scratch on it.  A merge of ordered lists reads heads and advances, so the port walks the candidate
//     buffer directly: the same values in the same order, and the threadgroup holds only the 64-entry
//     selection and exp scratch.
//   * THE TAIL'S DOUBLES DO NOT EXIST (no fp64 on this GPU).  The CUDA tail runs top_p's and the pick's
//     sums and cumulatives in double; here the exps are metal::precise::exp of the same f32 arguments, the
//     two ordered sums are the port's Neumaier-compensated f32 sums (strata_port.metalh - double's job,
//     the router's port set the pattern), and every comparison runs in f32.  The decisions the CUDA tail
//     took in double sit on margins orders wider than the ~1e-7 this carries; sampler_parity's fixtures 9,
//     16 and 17 pin the picks draw for draw.
// MSL notes: the merge's Sink lambda becomes a template helper with two fixed sinks (MSL has no lambda
// expressions); __shfl_down/xor_sync are simd_shuffle_down/xor (the down-chain's out-of-range lanes are
// undefined in MSL, harmless - lane 0's fold only reads in-range lanes); __umulhi is a 64-bit product shift;
// the coupled-draft helpers of include/strata/core/coupled_draft.hpp are transcribed below; the greedy
// kernel's dead `pmin` argument is dropped (the CUDA body voids it).
#include "strata_port.metalh"
#include <metal_atomic>

// SamplerParams, the .cu's by-value kernel argument, mirrored field for field - it holds no pointers, so it
// rides the launcher's setBytes whole (the port's rule 9 bans pointers in bytes, not scalars; the bytes
// arrive verbatim).  `ulong` is MSL's uint64_t; the layout is the C one on both sides of the seam.
struct SamplerParams {
    int top_k;
    float top_p;
    float min_p;
    float temperature;
    int min_keep;
    int penalty_last_n;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    ulong seed;
    ulong counter;
    bool greedy;
};

constant const int smp_kSelMax = 64;          // the widest top_k list, `sampler_kernel`'s KMAX
constant const int smp_kSplitPerLane = 32;    // logits per lane, in registers
constant const int smp_kSplitWarpSpan = 32 * smp_kSplitPerLane;    // 1,024 logits per warp
constant const int smp_kSplitWarps = 4;
constant const int smp_kSplitBlockSpan = smp_kSplitWarps * smp_kSplitWarpSpan;    // 4,096 logits per block
constant const int smp_kSplitThreads = smp_kSplitWarps * 32;      // 128

// ---- Philox 4x32-10, the counter-based generator the phase asks for (a batch can be sampled in any order;
// a run is reproducible).  `__umulhi` is the high half of a 32x32 multiply, spelled as a 64-bit shift - the
// same spelling the parity test's host transcription uses, so the draws are bit-identical. ----
static inline void smp_philox_round(thread uint& c0, thread uint& c1, thread uint& c2, thread uint& c3,
                                    uint k0, uint k1) {
    const uint hi0 = (uint) (((ulong) 0x9E3779B9u * c0) >> 32);
    const uint hi1 = (uint) (((ulong) 0xBB67AE85u * c2) >> 32);
    const uint lo0 = 0x9E3779B9u * c0;
    const uint lo1 = 0xBB67AE85u * c2;
    const uint n0 = hi1 ^ c1 ^ k0;
    const uint n1 = lo1;
    const uint n2 = hi0 ^ c3 ^ k1;
    const uint n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

static inline float smp_philox_uniform(ulong seed, ulong counter) {
    uint c0 = (uint) counter, c1 = (uint) (counter >> 32);
    uint c2 = (uint) seed, c3 = (uint) (seed >> 32);
    for (int i = 0; i < 10; ++i) smp_philox_round(c0, c1, c2, c3, (uint) i, 0u);
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// ---- the penalties, transcribed from `llama_sampler_penalties_apply`.  The repeat penalty MULTIPLIES for
// non-positive logits and DIVIDES for positive ones; the presence penalty is `float(count > 0)`, a boolean. ----
static inline int smp_history_count(constant const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i)
        if (h[i] == v) ++c;
    return c;
}

// by value: the coupled kernels hold a THREAD-local copy of the params (their counter is patched first), and
// MSL cannot bind a thread-space object to a constant-space reference
static inline float smp_apply_penalties(float logit, int count, SamplerParams p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

// The membership bitmap of one row, in the launcher's device scratch: zero, set the window's bits with
// relaxed device atomics, a mem_device barrier on each side.  `block` is the launch's thread count.
static inline void smp_build_bits(device uint* bits, constant const int* hrow, int hlen, int n_vocab, uint tid,
                                  uint block) {
    const int words = (n_vocab + 31) / 32;
    for (int w = (int) tid; w < words; w += (int) block) bits[w] = 0u;
    threadgroup_barrier(mem_flags::mem_device);
    for (int i = (int) tid; i < hlen; i += (int) block) {
        const int h = hrow[i];
        if (h >= 0 && h < n_vocab)   // an id outside the vocabulary is never a candidate
            atomic_fetch_or_explicit((device atomic_uint*) &bits[h >> 5], 1u << (h & 31), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
}

static inline int smp_hit_count(const device uint* bits, constant const int* hrow, int hlen, int v,
                                bool use_bits) {
    if (!use_bits || (bits[v >> 5] & (1u << (v & 31))) == 0u) return 0;
    return smp_history_count(hrow, hlen, v);
}

// ---- the selection order: (value, id) before (value', id') when value > value', or value == value' and
// id < id' - the serial scan's strict `>` keeps the first maximum it meets, the reductions resolve a tie to
// the lower id.  A strict total order (-0 and +0 compare equal and fall to the id). ----
static inline void smp_take_first(thread float& bv, thread int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// the first of the warp's 32 candidates, left in EVERY lane (an XOR butterfly over a strict total order is
// exact: two lanes never hold different candidates that compare equal)
static inline void smp_warp_first(thread float& bv, thread int& bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_xor(bv, (uint) off);
        const int oi = simd_shuffle_xor(bi, (uint) off);
        smp_take_first(bv, bi, ov, oi);
    }
}

// the warp's first via a shuffle-down chain (CUDA's __shfl_down_sync reduction): lane 0 holds the winner.
// Lanes near the end read out of range - undefined in MSL, harmless: lane 0's fold only reads in-range lanes.
static inline void smp_warp_argmax(thread float& bv, thread int& bi) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(bv, (uint) off);
        const int oi = simd_shuffle_down(bi, (uint) off);
        smp_take_first(bv, bi, ov, oi);
    }
}

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
static inline int smp_sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < smp_kSelMax) ? top_k : smp_kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// the coupled-draft helpers of include/strata/core/coupled_draft.hpp (plain C++ there; MSL cannot include it)
static inline int smp_hist_len(int penalty_last_n, int cap) {
    return penalty_last_n <= 0 ? 0 : (penalty_last_n < cap ? penalty_last_n : cap);
}
static inline int smp_hist_start(int cap, int j, int h) { return cap + j - h; }
static inline ulong smp_draft_counter(long cell) { return (ulong) (cell + 1); }

// **THE TAIL ON ONE WARP, WITH `sampler_kernel`'S ARITHMETIC** (top_p / min_p / temperature / one Philox
// draw).  `sel_ids` / `sel_logit` (threadgroup, k entries) are the top_k list in selection order; lane 0
// writes `out[t]`.  The lanes share the exps (one per entry, into `ex`) and the quotients, and lane 0 alone
// runs the two ORDERED sums and the two cumulative scans - the CUDA shape, with the doubles emulated (the
// file header's third note).  kProb (the coupled draft only): lane 0 also writes the pick's probability
// under the final distribution to `*prob_out`.
template <bool kProb, typename OutPtr, typename ProbPtr>
static inline void smp_tail_warp(threadgroup const int* sel_ids, threadgroup const float* sel_logit, int k,
                                 SamplerParams p, int t, OutPtr out, threadgroup float* ex, uint lane,
                                 ProbPtr prob_out) {
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = (int) lane; i < k; i += 32) ex[i] = metal::precise::exp(sel_logit[i] - mx);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        if (lane == 0) {
            KahanSum ks;
            for (int i = 0; i < k; ++i) ks.add(ex[i]);
            sum = ks.value();
        }
        sum = simd_shuffle(sum, 0u);
        simdgroup_barrier(mem_flags::mem_threadgroup);   // lane 0 has read every `ex` before it is overwritten
        for (int i = (int) lane; i < k; i += 32) ex[i] = ex[i] / sum;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        int cut = k;
        if (lane == 0) {
            KahanSum ks;
            for (int i = 0; i < k; ++i) {
                ks.add(ex[i]);
                if (ks.value() >= p.top_p) { cut = i + 1; break; }
            }
        }
        cut = simd_shuffle(cut, 0u);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        simdgroup_barrier(mem_flags::mem_threadgroup);   // `ex` is written again below
    }
    // min_p on top_p's survivors: in logit space the threshold is `sel_logit[0] + log(min_p)`; the head
    // itself always survives, so the count never reaches zero (0 disables)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + metal::precise::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw - the chain APPLIES the temperature after the truncation filters
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmax(smx, sel_logit[i] * inv_t);
    for (int i = (int) lane; i < n_keep; i += 32) ex[i] = metal::precise::exp(sel_logit[i] * inv_t - smx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    if (lane == 0) {
        KahanSum ks;
        for (int i = 0; i < n_keep; ++i) ks.add(ex[i]);
        sum = ks.value();
    }
    sum = simd_shuffle(sum, 0u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int) lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const float u = smp_philox_uniform(p.seed, p.counter + (ulong) t);
        KahanSum ks;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            ks.add(ex[i]);
            if (u < ks.value()) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if (kProb) *prob_out = n_keep > 0 ? ex[pi] : 1.0f;
    }
}

// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order
// and padded with sentinels - into their first `k`.  Lane owns lists `lane` and `lane + 32`; a round takes
// the first of all heads and advances the list it came from.  An id is in one list at most, so exactly one
// head matches.  CUDA passed a Sink lambda; MSL has none, so the two sinks the file had are branches: the
// part kernel writes the merged (id, value-bits) pair to a device buffer, the merge kernels write the
// selection arrays (sentinel ids clamped to 0, as their sinks did).  `lists` may live in the threadgroup
// (the part kernel's warp lists) or in device memory (the merge kernels read the candidate buffer directly -
// the file header's second note).
template <typename ListsPtr>
static inline void smp_warp_merge_lists(ListsPtr lists, int nl, int stride, int k, int n_vocab, uint lane,
                                        device int2* pair_dst, threadgroup int* sel_ids,
                                        threadgroup float* sel_logit) {
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = (int) lane + 32 * m;
        pos[m] = 0;
        hv[m] = as_type<float>(0xff800000u);
        hi[m] = n_vocab;
        if (L < nl) {
            const int2 c = lists[(size_t) L * (size_t) stride];
            hi[m] = c.x;
            hv[m] = as_type<float>(c.y);
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        smp_take_first(bv, bi, hv[1], hi[1]);
        smp_warp_first(bv, bi);
        if (pair_dst != nullptr) {
            if (lane == 0) pair_dst[i] = int2(bi, as_type<int>(bv));
        } else if (lane == 0) {
            sel_ids[i] = bi < n_vocab ? bi : 0;
            sel_logit[i] = bv;
        }
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const int2 c = lists[(size_t) ((int) lane + 32 * m) * (size_t) stride + (size_t) pos[m]];
                    hi[m] = c.x;
                    hv[m] = as_type<float>(c.y);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = as_type<float>(0xff800000u);
                }
            }
        }
    }
}

// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY** - the block reduction that keeps the
// serial scan's tie rule: every comparison resolves to the larger value and, on equality, the SMALLER index.
kernel void sampler_greedy_kernel(constant const float* logits [[buffer(0)]],
                                  constant const int& n_vocab [[buffer(1)]],
                                  constant const int* history [[buffer(2)]],
                                  constant const int& history_len [[buffer(3)]],
                                  constant const SamplerParams& p [[buffer(4)]],
                                  constant const int& plen [[buffer(5)]],
                                  constant const uint& block [[buffer(6)]],
                                  device uint* penal_bits [[buffer(7)]],
                                  device int* out [[buffer(8)]],
                                  uint3 gpos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = (int) gpos.x;
    constant const float* l = logits + (size_t) t * (size_t) n_vocab;
    constant const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    // the membership bitmap of THIS row, in the launcher's device scratch; the gate needs a NON-EMPTY WINDOW
    // (a stale history buffer with penalty_last_n == 0 must not be touched - PR #59)
    const int bits_words = (n_vocab + 31) / 32;
    device uint* bits_row = penal_bits != nullptr ? penal_bits + (size_t) t * (size_t) bits_words : nullptr;
    const device uint* bits = bits_row;
    const bool use_bits = bits_row != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) smp_build_bits(bits_row, hrow, hlen, n_vocab, tid, block);

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one
    float bv = as_type<float>(0xff800000u);   // -inf
    int best = n_vocab;
    for (int v = (int) tid; v < n_vocab; v += (int) block) {
        const float s = smp_apply_penalties(l[v], smp_hit_count(bits, hrow, hlen, v, use_bits), p);
        if (s > bv) { bv = s; best = v; }
    }
    smp_warp_argmax(bv, best);
    threadgroup float sv[32];
    threadgroup int si[32];
    if (lane == 0) { sv[sg] = bv; si[sg] = best; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const int nw = (int) ((block + 31u) >> 5);
        float wv = lane < (uint) nw ? sv[lane] : as_type<float>(0xff800000u);
        int wi = lane < (uint) nw ? si[lane] : n_vocab;
        smp_warp_argmax(wv, wi);
        // a tie between two -inf candidates leaves `wi == n_vocab`, and the serial version answered 0
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

// **THE OLD SAMPLED PATH, ONE BLOCK PER TOKEN** (engine 0.1.20's kernel, kept as the reference behind
// STRATA_OLD_SAMPLER=1): top_k as k argmax rounds with the `taken` sweep, then the tail computed by every
// thread redundantly - the CUDA shape; thread 0 writes.  The doubles of the tail are emulated in f32 (the
// file header's third note).
kernel void sampler_kernel(constant const float* logits [[buffer(0)]],
                           constant const int& n_vocab [[buffer(1)]],
                           constant const int& n_tokens [[buffer(2)]],
                           constant const int* history [[buffer(3)]],
                           constant const int& history_len [[buffer(4)]],
                           constant const SamplerParams& p [[buffer(5)]],
                           constant const uint& block [[buffer(6)]],
                           device uint* penal_bits [[buffer(7)]],
                           device int* out [[buffer(8)]],
                           uint3 gpos [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = (int) gpos.x;
    if (t >= n_tokens) return;
    constant const float* l = logits + (size_t) t * (size_t) n_vocab;
    // Temperature is computed here, but the chain still APPLIES it after the truncation filters
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    constant const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    const int bits_words = (n_vocab + 31) / 32;
    device uint* bits_row = penal_bits != nullptr ? penal_bits + (size_t) t * (size_t) bits_words : nullptr;
    const device uint* bits = bits_row;
    const bool use_bits = bits_row != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) smp_build_bits(bits_row, hrow, hlen, n_vocab, tid, block);

    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist, 64
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    // ---- top_k: k rounds of a block argmax over the not-yet-taken; `sel_*` holds the kept ids and their
    // raw logits in selection order (descending by value, ties to the lower index) ----
    threadgroup int sel_ids[64];
    threadgroup float sel_logit[64];
    threadgroup float sv[32];
    threadgroup int si[32];
    for (int i = 0; i < k; ++i) {
        float bv = as_type<float>(0xff800000u);
        int best = n_vocab;
        for (int v = (int) tid; v < n_vocab; v += (int) block) {
            bool taken = false;
            for (int j = 0; j < i; ++j)
                if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = smp_apply_penalties(l[v], smp_hit_count(bits, hrow, hlen, v, use_bits), p);
            if (s > bv) { bv = s; best = v; }
        }
        smp_warp_argmax(bv, best);
        if (lane == 0) { sv[sg] = bv; si[sg] = best; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const int nw = (int) ((block + 31u) >> 5);
            float wv = lane < (uint) nw ? sv[lane] : as_type<float>(0xff800000u);
            int wi = lane < (uint) nw ? si[lane] : n_vocab;
            smp_warp_argmax(wv, wi);
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- top_p over the top_k list, then min_p, then temperature and one Philox draw - llama.cpp's order.
    // Every thread computes the same chain redundantly over `sel_*` (the CUDA kernel's own shape), so they
    // agree on `pick` and thread 0 writes it ----
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        KahanSum ksum;
        for (int i = 0; i < k; ++i) ksum.add(metal::precise::exp(sel_logit[i] - mx));
        const float sum = ksum.value();
        KahanSum kcum;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            kcum.add(metal::precise::exp(sel_logit[i] - mx) / sum);
            if (kcum.value() >= p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + metal::precise::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53)
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmax(smx, sel_logit[i] * inv_t);
    KahanSum ksum2;
    for (int i = 0; i < n_keep; ++i) ksum2.add(metal::precise::exp(sel_logit[i] * inv_t - smx));
    const float sum = ksum2.value();
    const float u = smp_philox_uniform(p.seed, p.counter + (ulong) t);
    KahanSum kcum2;
    int pick = sel_ids[n_keep > 0 ? n_keep - 1 : 0];
    for (int i = 0; i < n_keep; ++i) {
        kcum2.add(metal::precise::exp(sel_logit[i] * inv_t - smx) / sum);
        if (u < kcum2.value()) { pick = sel_ids[i]; break; }
    }
    if (tid == 0) out[t] = pick;
}

// **THE ONE-BLOCK SAMPLED PATH: `sampler_kernel` WITHOUT THE `taken` SWEEP.**  Round i's candidates are the
// logits strictly AFTER round i-1's pick in the selection order, `s < prev_v || (s == prev_v && v > prev_i)`,
// which is exactly the set `taken` left.  O(k x n_vocab) per row, then warp 0 runs the tail.
kernel void sampler_one_block_kernel(constant const float* logits [[buffer(0)]],
                                     constant const int& n_vocab [[buffer(1)]],
                                     constant const int* history [[buffer(2)]],
                                     constant const int& history_len [[buffer(3)]],
                                     constant const SamplerParams& p [[buffer(4)]],
                                     constant const uint& block [[buffer(5)]],
                                     device uint* penal_bits [[buffer(6)]],
                                     device int* out [[buffer(7)]],
                                     uint3 gpos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = (int) gpos.x;
    constant const float* l = logits + (size_t) t * (size_t) n_vocab;
    constant const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    const int bits_words = (n_vocab + 31) / 32;
    device uint* bits_row = penal_bits != nullptr ? penal_bits + (size_t) t * (size_t) bits_words : nullptr;
    const device uint* bits = bits_row;
    const bool use_bits = bits_row != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) smp_build_bits(bits_row, hrow, hlen, n_vocab, tid, block);

    const int k = smp_sampled_k(p.top_k, n_vocab);
    threadgroup int sel_ids[smp_kSelMax];
    threadgroup float sel_logit[smp_kSelMax];
    threadgroup float ex[smp_kSelMax];
    threadgroup float sv[32];
    threadgroup int si[32];
    float prev_v = as_type<float>(0x7f800000u);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    for (int i = 0; i < k; ++i) {
        float bv = as_type<float>(0xff800000u);   // -inf
        int best = n_vocab;
        for (int v = (int) tid; v < n_vocab; v += (int) block) {
            const float s = smp_apply_penalties(l[v], smp_hit_count(bits, hrow, hlen, v, use_bits), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        smp_warp_argmax(bv, best);
        if (lane == 0) { sv[sg] = bv; si[sg] = best; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const int nw = (int) ((block + 31u) >> 5);
            float wv = lane < (uint) nw ? sv[lane] : as_type<float>(0xff800000u);
            int wi = lane < (uint) nw ? si[lane] : n_vocab;
            smp_warp_argmax(wv, wi);
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // an empty round leaves (-inf, 0): nothing comes after it, as nothing was left untaken
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (sg != 0) return;
    smp_tail_warp<false>(sel_ids, sel_logit, k, p, t, out, ex, lane, (device float*) nullptr);
}

// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK.**  Grid (blocks per row, rows), 128 threads.  Each
// warp loads its 1,024 penalised logits once into registers (lane + 32 j: every load is one coalesced
// 128-byte line) and runs k warp-argmax rounds with the threshold of the one-block kernel.  Warp 0 then
// merges the four warp lists into the block's list in `cand` (row-major: row t, block b, entry i at
// `(t * n_blocks + b) * k + i`, as (id, value bits)).  The bitmap covers THIS BLOCK'S 4,096 logits (512
// bytes) and stays in the threadgroup.
kernel void sampler_split_part_kernel(constant const float* logits [[buffer(0)]],
                                      constant const int& n_vocab [[buffer(1)]],
                                      constant const int* history [[buffer(2)]],
                                      constant const int& history_len [[buffer(3)]],
                                      constant const SamplerParams& p [[buffer(4)]],
                                      constant const int& k [[buffer(5)]],
                                      constant const int& n_blocks [[buffer(6)]],
                                      device int2* cand [[buffer(7)]],
                                      uint2 gpos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = (int) gpos.y;
    constant const float* l = logits + (size_t) t * (size_t) n_vocab;
    const int blo = (int) gpos.x * smp_kSplitBlockSpan;

    constant const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    threadgroup uint bits[smp_kSplitBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = (int) tid; w < smp_kSplitBlockSpan / 32; w += smp_kSplitThreads) bits[w] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int) tid; i < hlen; i += smp_kSplitThreads) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < smp_kSplitBlockSpan)
                atomic_fetch_or_explicit((threadgroup atomic_uint*) &bits[(h - blo) >> 5],
                                         1u << ((h - blo) & 31), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // This warp's logits, penalised: `smp_apply_penalties` with a zero count returns the logit unchanged.
    // Past the vocabulary: -inf, which no round picks.
    const int lo = blo + (int) sg * smp_kSplitWarpSpan;
    float s[smp_kSplitPerLane];
#pragma unroll
    for (int j = 0; j < smp_kSplitPerLane; ++j) {
        const int v = lo + 32 * j + (int) lane;
        s[j] = v < n_vocab ? l[v] : as_type<float>(0xff800000u);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < smp_kSplitPerLane; ++j) {
            const int v = lo + 32 * j + (int) lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))) != 0u)
                s[j] = smp_apply_penalties(s[j], smp_history_count(hrow, hlen, v), p);
        }
    }

    threadgroup int2 wl[smp_kSplitWarps][smp_kSelMax];
    float prev_v = as_type<float>(0x7f800000u);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first
        // in the order; `take_first` then orders the two
        float b0 = as_type<float>(0xff800000u), b1 = as_type<float>(0xff800000u);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < smp_kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + (int) lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        smp_take_first(b0, i0, b1, i1);
        smp_warp_first(b0, i0);
        if (i0 >= n_vocab) break;    // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wl[sg][i] = int2(i0, as_type<int>(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + (int) lane; r < k; r += 32)
        wl[sg][r] = int2(n_vocab, as_type<int>(0xff800000u));   // sentinels
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        device int2* dst = cand + ((size_t) t * (size_t) n_blocks + (size_t) gpos.x) * (size_t) k;
        smp_warp_merge_lists((threadgroup const int2*) &wl[0][0], smp_kSplitWarps, smp_kSelMax, k, n_vocab,
                             lane, dst, (threadgroup int*) nullptr, (threadgroup float*) nullptr);
    }
}

// **SPLIT STAGE 2: ONE WARP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL.**  The CUDA kernel copies
// the row's lists (up to 64 x 64 int2, 32 KB) to shared memory first; this GPU's threadgroup budget is the
// whole 32 KB, so the merge reads them straight from the candidate buffer (the file header's second note).
kernel void sampler_split_merge_kernel(constant const int2* cand [[buffer(0)]],
                                       constant const int& n_blocks [[buffer(1)]],
                                       constant const int& n_vocab [[buffer(2)]],
                                       constant const SamplerParams& p [[buffer(3)]],
                                       constant const int& k [[buffer(4)]],
                                       device int* out [[buffer(5)]],
                                       uint3 gpos [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_threadgroup]]) {
    const int t = (int) gpos.x;
    threadgroup int sel_ids[smp_kSelMax];
    threadgroup float sel_logit[smp_kSelMax];
    threadgroup float ex[smp_kSelMax];
    constant const int2* src = cand + (size_t) t * (size_t) n_blocks * (size_t) k;
    smp_warp_merge_lists(src, n_blocks, k, k, n_vocab, lane, nullptr, sel_ids, sel_logit);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    smp_tail_warp<false>(sel_ids, sel_logit, k, p, t, out, ex, lane, (device float*) nullptr);
}

// ---- COUPLED DRAFT SAMPLING (include/strata/core/coupled_draft.hpp): the MTP draft layer samples its draft
// with the target's chain and the target's Philox draw.  Everything that varies per request or per round is
// read from DEVICE memory - these kernels are captured into the drafter's round/step graphs. ----

// The round's inputs: the request's SamplerParams and the history base (the last h slots before `cap`),
// from mapped host memory into the device copies the chain's kernels read.
kernel void coupled_stage_kernel(constant const SamplerParams* mp [[buffer(0)]],
                                 constant const int* mh [[buffer(1)]],
                                 device SamplerParams* dp [[buffer(2)]],
                                 device int* ring [[buffer(3)]],
                                 constant const int& cap [[buffer(4)]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const constant volatile int* s = (constant const volatile int*) mp;
    device int* d = (device int*) dp;
    for (int i = (int) tid; i < (int) (sizeof(SamplerParams) / sizeof(int)); i += 256) d[i] = s[i];
    const int h = smp_hist_len(((constant const volatile SamplerParams*) mp)->penalty_last_n, cap);
    const constant volatile int* vh = (constant const volatile int*) mh;
    for (int i = cap - h + (int) tid; i < cap; i += 256) ring[i] = vh[i];
}

// The penalties, applied in place to the draft logits before the selection.  Draft j's window is the ring's
// [cap + j - h, cap + j).  Each distinct token is penalised once: the entry whose atomicOr sets its bit does
// it.  CUDA held the dedup bitmap in dynamic shared memory (nv bits); it lives in the launcher's device
// scratch here (the file header's first note).
kernel void coupled_penalize_kernel(device float* logits [[buffer(0)]],
                                    constant const int& nv [[buffer(1)]],
                                    constant const int* id_to_sub [[buffer(2)]],
                                    constant const int& id_vocab [[buffer(3)]],
                                    constant const SamplerParams* dp [[buffer(4)]],
                                    constant const int* ring [[buffer(5)]],
                                    constant const int& cap [[buffer(6)]],
                                    constant const int& j [[buffer(7)]],
                                    device uint* seen [[buffer(8)]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    SamplerParams p = *dp;
    const int h = smp_hist_len(p.penalty_last_n, cap);
    if (h <= 0) return;
    constant const int* hrow = ring + smp_hist_start(cap, j, h);
    const int words = (nv + 31) / 32;
    for (int w = (int) tid; w < words; w += 1024) seen[w] = 0u;
    threadgroup_barrier(mem_flags::mem_device);
    for (int i = (int) tid; i < h; i += 1024) {
        const int v = hrow[i];
        if (v < 0 || v >= id_vocab) continue;
        const int s = id_to_sub != nullptr ? id_to_sub[v] : v;
        if (s < 0 || s >= nv) continue;
        const uint bit = 1u << (uint) (s & 31);
        if ((atomic_fetch_or_explicit((device atomic_uint*) &seen[s >> 5], bit, memory_order_relaxed) & bit)
            != 0u)
            continue;
        logits[s] = smp_apply_penalties(logits[s], smp_history_count(hrow, h, v), p);
    }
}

// The merge of `sampler_split_merge_kernel` (lists of `kpart` entries, the request's top_k taken from them),
// then the tail with the counter of the row that will verify this draft.  Lane 0 maps the pick to its token
// id, writes it and its probability, and appends it to the ring for the next draft's penalty window.
kernel void coupled_merge_kernel(constant const int2* cand [[buffer(0)]],
                                 constant const int& n_blocks [[buffer(1)]],
                                 constant const int& nv [[buffer(2)]],
                                 constant const int& kpart [[buffer(3)]],
                                 constant const SamplerParams* dp [[buffer(4)]],
                                 constant const int* step_rec [[buffer(5)]],
                                 constant const int* sub_to_id [[buffer(6)]],
                                 device int* ring [[buffer(7)]],
                                 constant const int& cap [[buffer(8)]],
                                 constant const int& j [[buffer(9)]],
                                 device int* out_id [[buffer(10)]],
                                 device float* out_prob [[buffer(11)]],
                                 uint lane [[thread_index_in_threadgroup]]) {
    SamplerParams p = *dp;
    p.counter = smp_draft_counter((long) step_rec[0]);
    const int k = smp_sampled_k(p.top_k, nv);   // <= kpart: the first k of a union lie in the first k of each list
    threadgroup int sel_ids[smp_kSelMax];
    threadgroup float sel_logit[smp_kSelMax];
    threadgroup float ex[smp_kSelMax];
    threadgroup int pick[1];
    threadgroup float prob[1];
    smp_warp_merge_lists(cand, n_blocks, kpart, k, nv, lane, nullptr, sel_ids, sel_logit);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (p.greedy || p.temperature <= 0.0f) {   // never launched for greedy requests; the argmax, defensively
        if (lane == 0) { pick[0] = sel_ids[0]; prob[0] = 1.0f; }
    } else {
        smp_tail_warp<true>(sel_ids, sel_logit, k, p, 0, pick, ex, lane,
                            (threadgroup float*) prob);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const int s = pick[0];
        const int id = sub_to_id != nullptr ? sub_to_id[s] : s;
        *out_id = id;
        *out_prob = prob[0];
        ring[cap + j] = id;
    }
}

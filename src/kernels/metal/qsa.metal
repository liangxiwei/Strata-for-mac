// src/kernels/metal/qsa.metal - the port of src/kernels/cuda/qsa.cu's kernels (K15): the QSA indexer key
// append/pooling, the per-cell indexer scores, the threshold-search top-k, the softmax attention over the
// gathered scratch, and the sigmoid gate.  kv_append_kernel / kv_gather_kernel (qsa.cu's FP16 KV steps) stay
// in kv_q8.metal/mm exactly as they sit on the CUDA side's sibling, so they are NOT redefined here.
//
// THE DOUBLE SITES, EMULATED (Apple GPUs have no fp64 - the compiler says so outright).  qsa.cu runs the
// indexer's POOLING in double (`__dadd_rn`/`__dmul_rn`, order-fixed) and holds the spare key to a BIT-EXACT
// contract, which a compensated f32 sum alone cannot meet: the qsa_ df layer below is a double-float (two
// f32s, ~48 good bits) carrying the same operation ORDER as the double original -
//   * every product is the EXACT two-product (fma(-a, b, p) does not round), where the double original's
//     f32-input products are exact in double as well;
//   * sums accumulate through compensated df adds, so the deviation from the double sum is the double's OWN
//     per-step rounding, ~2^-47 relative over the 128 terms (a plain f32 sum sits at 2^-24 and flips the
//     spare key's last bit in ~half the components);
//   * the reciprocal square root is two df Newton steps off a precise::rsqrt seed (~2^-47 after the second);
//   * the indexer score dots (qsa_index_kernel) use the same df products and a df shuffle butterfly, matching
//     the double dot's exact products and its ascending accumulation.
// The gate's double sigmoid became f32 precise::exp (its tolerance is the fp16 store's 2.4e-4; the CUDA
// test's "bits differing" line is informational there too).
//
// STRUCTURAL NOTES.
//   * indexer_key_append's `extern __shared__ double s_mean[]` staging is GONE: every thread recomputes the
//     mean it needs from the tail (idx_dim x r f32 reads; a per-token kernel - the cost is noise) in the same
//     add order, so the bits are what a shared staging would hold, and MSL has no dynamic threadgroup size.
//   * qsa_attend's dynamic shared size (a graph-capture requirement on CUDA) becomes a FIXED threadgroup
//     stage of QSA_ATTEND_CAP + 32 floats; the launcher refuses max_ids over the capacity, loudly.
#include "strata_port.metalh"

// ---- the double-float (df) layer: two f32s, hi + lo, |lo| <= ulp(hi)/2 -------------------------
struct qsa_df {
    float hi;
    float lo;
};

static inline qsa_df qsa_df_norm(float hi, float lo) {   // renormalize (Knuth's quick two-sum)
    const float s = hi + lo;
    return qsa_df{s, hi - s + lo};
}

// Dekker's two-sum: s = fl(a+b), e = the exact residual
static inline void qsa_two_sum(float a, float b, thread float& s, thread float& e) {
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// the exact product: p = fl(a*b), e = a*b - p exactly (the fma does not round)
static inline void qsa_two_prod(float a, float b, thread float& p, thread float& e) {
    p = a * b;
    e = fma(a, b, -p);
}

static inline qsa_df qsa_df_add(qsa_df a, qsa_df b) {
    float s, e;
    qsa_two_sum(a.hi, b.hi, s, e);
    return qsa_df_norm(s, e + a.lo + b.lo);
}

static inline qsa_df qsa_df_mul(qsa_df a, qsa_df b) {
    float p, e;
    qsa_two_prod(a.hi, b.hi, p, e);
    return qsa_df_norm(p, e + (a.hi * b.lo + a.lo * b.hi));
}

static inline qsa_df qsa_df_neg(qsa_df a) { return qsa_df{-a.hi, -a.lo}; }

// 1/sqrt(x) in df: a precise::rsqrt seed, two quadratic Newton steps (y *= 1.5 - 0.5*x*y^2)
static inline qsa_df qsa_df_rsqrt(qsa_df x) {
    qsa_df y = qsa_df{metal::precise::rsqrt(x.hi), 0.0f};
    for (int it = 0; it < 2; ++it) {
        const qsa_df t = qsa_df_mul(x, qsa_df_mul(y, y));
        const qsa_df c = qsa_df_add(qsa_df{1.5f, 0.0f}, qsa_df_mul(qsa_df{-0.5f, 0.0f}, t));
        y = qsa_df_mul(y, c);
    }
    return y;
}

// a/b in df: the correctly-rounded f32 quotient, refined twice by its exact residual
static inline qsa_df qsa_df_div(qsa_df a, qsa_df b) {
    qsa_df q = qsa_df{a.hi / b.hi, 0.0f};
    for (int it = 0; it < 2; ++it) {
        const qsa_df r = qsa_df_add(a, qsa_df_neg(qsa_df_mul(b, q)));
        q = qsa_df_add(q, qsa_df{r.hi / b.hi + r.lo / b.hi, 0.0f});
    }
    return q;
}

// mrope.hpp's mrope_pos (as native_rope.metal / rope.metal spell it)
static inline int qsa_mrope_pos(constant const int* tab, int pos, int pair) {
    return tab != nullptr ? tab[(ulong) pos * 3 + (uint) pair % 3] : pos;
}

// rope.hpp's rope_neox_pair - ONE place for the pairing, the same function the CUDA kernel calls
static inline void qsa_rope_neox_pair(float a, float b, float c, float s, thread float& oa, thread float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

// ================= 2. indexer_key_append =================

// One thread per indexer dim (grid 1 x idx_dim).  Appends the raw key, and on a block completion pools AND
// ROTATES the pooled row - the rotation lives in the kernel so a captured graph replays the right position
// (the CUDA file's capturability argument, unchanged).  The double arithmetic is the qsa_ df layer above,
// in the CUDA file's order (mean in cell order, sum of squares over d ascending, (m*inv)*w).
kernel void indexer_key_append_kernel(constant const float* raw [[buffer(0)]],
                                      constant const int* pos_dev [[buffer(1)]],
                                      constant const int& pos_base [[buffer(2)]],
                                      constant const float* w_k_norm [[buffer(3)]],
                                      constant const float& eps [[buffer(4)]],
                                      device float* tail [[buffer(5)]],
                                      device float* dead [[buffer(6)]],
                                      device float* pooled [[buffer(7)]],
                                      device int* block_pos [[buffer(8)]],
                                      constant const int& idx_dim [[buffer(9)]],
                                      constant const int& r [[buffer(10)]],
                                      constant const int& n_rot [[buffer(11)]],
                                      constant const float* cos_tab [[buffer(12)]],
                                      constant const float* sin_tab [[buffer(13)]],
                                      constant const int* mtab [[buffer(14)]],
                                      uint d [[thread_position_in_grid]]) {
    const int pos = pos_dev[0];
    const int slot = pos % r;

    // the RAW tail: a cell that completes a block is not stored (the pool consumes it directly)
    if (slot < r - 1) tail[(ulong) slot * idx_dim + d] = raw[d];

    // the spare slot's key is rms_norm(raw[0]) - CONSTANT for the sequence; bit-exact against the reference
    // (the rotation is at angle 0, where the f32 rope is the exact identity, so only the norm must match)
    if (pos == 0) {
        qsa_df ss = qsa_df{0.0f, 0.0f};
        for (int i = 0; i < idx_dim; ++i) {
            float p, e;
            qsa_two_prod(raw[i], raw[i], p, e);
            ss = qsa_df_add(ss, qsa_df{p, e});
        }
        const qsa_df x = qsa_df_add(qsa_df_div(ss, qsa_df{(float) idx_dim, 0.0f}), qsa_df{eps, 0.0f});
        const qsa_df inv = qsa_df_rsqrt(x);
        const qsa_df y = qsa_df_mul(qsa_df_mul(qsa_df{raw[d], 0.0f}, inv), qsa_df{w_k_norm[d], 0.0f});
        float yf = y.hi + y.lo;
        // row 0 of the table, whatever pos_base is; at angle 0 the sine is exactly 0 in every scaling
        if ((int) d < n_rot) yf *= cos_tab[(uint) d % (uint) (n_rot / 2)];
        dead[d] = yf;
        pooled[d] = yf;
    }

    if (slot != r - 1) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the mean of this block's r raw keys in CELL order: tail rows 0..r-2, then the incoming cell
    qsa_df m = qsa_df{tail[d], 0.0f};
    for (int j = 1; j < r - 1; ++j) m = qsa_df_add(m, qsa_df{tail[(ulong) j * idx_dim + d], 0.0f});
    m = qsa_df_add(m, qsa_df{raw[d], 0.0f});
    m = qsa_df_div(m, qsa_df{(float) r, 0.0f});

    // the sum of squares over d ascending - every thread recomputes each m(i) identically (the CUDA kernel
    // stages them in shared doubles; the values and the add order are the same)
    qsa_df ss = qsa_df{0.0f, 0.0f};
    for (int i = 0; i < idx_dim; ++i) {
        qsa_df mi = qsa_df{tail[(ulong) i], 0.0f};
        for (int j = 1; j < r - 1; ++j) mi = qsa_df_add(mi, qsa_df{tail[(ulong) j * idx_dim + i], 0.0f});
        mi = qsa_df_add(mi, qsa_df{raw[i], 0.0f});
        mi = qsa_df_div(mi, qsa_df{(float) r, 0.0f});
        ss = qsa_df_add(ss, qsa_df_mul(mi, mi));
    }
    const qsa_df x = qsa_df_add(qsa_df_div(ss, qsa_df{(float) idx_dim, 0.0f}), qsa_df{eps, 0.0f});
    const qsa_df inv = qsa_df_rsqrt(x);

    const int b = pos / r;
    const qsa_df y = qsa_df_mul(qsa_df_mul(m, inv), qsa_df{w_k_norm[d], 0.0f});
    pooled[(ulong) b * idx_dim + d] = y.hi + y.lo;
    pooled[(ulong) (b + 1) * idx_dim + d] = dead[d];    // the spare slot MOVES to b+1, same constant value
    if (d == 0) block_pos[0] = pos_base + b * r;        // the block's first cell's POSITION
    threadgroup_barrier(mem_flags::mem_threadgroup);    // the row is complete only now

    // the in-place rotation of the row that just completed; i pairs with i + half (NEOX)
    const int half_n = n_rot / 2;    // `half` is a reserved TYPE NAME in MSL (the rope port's lesson)
    if ((int) d < half_n) {
        device float* row = pooled + (ulong) b * idx_dim;
        const ulong toff = (ulong) qsa_mrope_pos(mtab, pos_base + b * r, (int) d) * half_n;
        const float a_in = row[d], b_in = row[half_n + d];    // both read before either write (in place)
        float oa, ob;
        qsa_rope_neox_pair(a_in, b_in, cos_tab[toff + d], sin_tab[toff + d], oa, ob);
        row[d] = oa;
        row[half_n + d] = ob;
    }
}

// ================= 3. qsa_index =================

// One warp per indexer head, one block per pooled row; thread 0 sums the relu'd dots IN HEAD ORDER and maps
// the row's score onto its cells.  The CUDA kernel's double dots are the qsa_ df layer: exact products,
// compensated accumulation, a df shuffle butterfly - the ascending order the reference's double sum has.
kernel void qsa_index_kernel(constant const float* pooled [[buffer(0)]],
                             constant const float* q_idx [[buffer(1)]],
                             constant const float* bias [[buffer(2)]],           // nil = no bias
                             constant const int& idx_n_head [[buffer(3)]],
                             constant const int& idx_dim [[buffer(4)]],
                             constant const long& r [[buffer(5)]],
                             constant const int* step [[buffer(6)]],
                             device float* cell_scores [[buffer(7)]],
                             uint3 gpos [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    const long n_bid = (long) step[2];    // kStepNBid
    const long n_kv = (long) step[1];     // kStepNKv
    threadgroup float2 s_dot[32];
    const long b = (long) gpos.x;
    // a block past the completed-block count must do nothing: the layer launches a CONSTANT grid
    if (b > n_bid) return;
    if (sg == 0) s_dot[lane] = float2(0.0f);   // zero the unused heads thread 0 never reads
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qsa_df acc = qsa_df{0.0f, 0.0f};
    for (int d = (int) lane; d < idx_dim; d += 32) {
        float p, e;
        qsa_two_prod(pooled[(ulong) b * idx_dim + (ulong) d], q_idx[(ulong) sg * idx_dim + (ulong) d], p, e);
        acc = qsa_df_add(acc, qsa_df{p, e});
    }
    for (int o = 16; o > 0; o >>= 1) {
        const qsa_df other = qsa_df{simd_shuffle_xor(acc.hi, (uint) o), simd_shuffle_xor(acc.lo, (uint) o)};
        acc = qsa_df_add(acc, other);
    }
    if (lane == 0) s_dot[sg] = float2(acc.hi, acc.lo);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t != 0) return;
    qsa_df score = qsa_df{0.0f, 0.0f};
    for (int h = 0; h < idx_n_head; ++h) {
        const qsa_df d_h = qsa_df{s_dot[h].x, s_dot[h].y};
        if (d_h.hi > 0.0f || (d_h.hi == 0.0f && d_h.lo > 0.0f)) score = qsa_df_add(score, d_h);
    }
    if (bias != nullptr) score = qsa_df_add(score, qsa_df{bias[b], 0.0f});
    // cell_block: cells of block b for b < n_bid, the WHOLE incomplete tail for b == n_bid
    long lo = b * r, hi = lo + r;
    if (b == n_bid) hi = n_kv;
    if (hi > n_kv) hi = n_kv;
    float sc = score.hi + score.lo;
    if (b == n_bid && n_kv % r != 0) sc += 1e9f;    // llama.cpp's incomplete-tail bias, read per replay
    for (long j = lo; j < hi; ++j) cell_scores[j] = sc;
}

// ================= 4. topk_512 =================

// Total order over f32 as an unsigned key (numpy's float semantics): +0.0f maps -0.0 away, a NaN lands
// below every real key.  The selection itself: binary lifting for the width-th largest key, then ONE
// ascending walk emitting strictly-above cells and the first eq-budget equals - ggml's tie rule.
static inline uint qsa_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

constant const int QSA_TOPK_THREADS = 256;

kernel void topk_kernel(constant const float* scores [[buffer(0)]],
                        constant const int* step [[buffer(1)]],
                        device int* out_ids [[buffer(2)]],
                        uint t [[thread_index_in_threadgroup]]) {
    const long n_kv = (long) step[1];     // kStepNKv
    const long width = (long) step[3];    // kStepWidth
    threadgroup int s_a[QSA_TOPK_THREADS];
    threadgroup int s_b[QSA_TOPK_THREADS];
    threadgroup long s_cgt;
    const long chunk = (n_kv + QSA_TOPK_THREADS - 1) / QSA_TOPK_THREADS;
    const long lo = (long) t * chunk;
    long hi = lo + chunk;
    if (hi > n_kv) hi = n_kv;

    // binary lifting: the largest key v with count(key >= v) >= width
    uint v = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const uint cand = v | (1u << bit);
        int c = 0;
        for (long j = lo; j < hi; ++j) if (qsa_order_key(scores[j]) >= cand) ++c;
        s_a[t] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = QSA_TOPK_THREADS / 2; s > 0; s >>= 1) {
            if (t < (uint) s) s_a[t] += s_a[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int tot = s_a[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);    // before the next round overwrites s_a
        if (tot >= (int) width) v = cand;
    }
    const uint thr = v;

    // counts and their exclusive prefixes (thread 0 in one walk - 256 iterations once per kernel)
    int gt = 0, eq = 0;
    for (long j = lo; j < hi; ++j) {
        const uint k = qsa_order_key(scores[j]);
        if (k > thr) ++gt;
        else if (k == thr) ++eq;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        long ag = 0, ae = 0;
        for (int i = 0; i < QSA_TOPK_THREADS; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = (int) ag;
            s_b[i] = (int) ae;
            ag += g;
            ae += e;
        }
        s_cgt = ag;    // how many are strictly above the threshold, i.e. where the equals start
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THE RANK VS THE COUNT: off_eq is the equals BEFORE this chunk, compared against the selection's budget
    const long off_eq = s_b[t];
    const long eq_budget = width - s_cgt;

    // how many this chunk selects, so the ascending walk can be scattered into place
    int sel = 0;
    {
        long e = off_eq;
        for (long j = lo; j < hi; ++j) {
            const uint k = qsa_order_key(scores[j]);
            if (k > thr) ++sel;
            else if (k == thr && e < eq_budget) { ++sel; ++e; }
        }
    }
    s_a[t] = sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        long a = 0;
        for (int i = 0; i < QSA_TOPK_THREADS; ++i) {
            const int c = s_a[i];
            s_a[i] = (int) a;
            a += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // emit, walking the cells ASCENDING: the tie rule's "first by index" is the walk order itself
    long w = s_a[t];
    long e = off_eq;
    for (long j = lo; j < hi; ++j) {
        const uint k = qsa_order_key(scores[j]);
        if (k > thr) out_ids[w++] = (int) j;
        else if (k == thr && e < eq_budget) { ++e; out_ids[w++] = (int) j; }
    }
}

// ================= 6. qsa_attend =================

// One block per query head, scores in threadgroup memory then overwritten with the softmax weights.  The
// CUDA kernel sizes its stage DYNAMICALLY (a capture requirement there); MSL has no dynamic threadgroup
// size, so the stage is fixed at QSA_ATTEND_CAP + 32 floats and the launcher checks max_ids against it.
constant const int QSA_ATTEND_CAP = 4096;
constant const float QSA_FLT_MAX = 3.4028234663852886e+38f;

static inline float qsa_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = metal::precise::fmax(v, simd_shuffle_xor(v, (uint) o));
    return v;
}

static inline float qsa_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, (uint) o);
    return v;
}

kernel void qsa_attend_kernel(constant const float* q [[buffer(0)]],
                              constant const uint* k_scratch [[buffer(1)]],    // fp16 patterns
                              constant const uint* v_scratch [[buffer(2)]],
                              constant const int* step [[buffer(3)]],
                              constant const int& n_head [[buffer(4)]],
                              constant const int& n_head_kv [[buffer(5)]],
                              constant const int& head_dim [[buffer(6)]],
                              constant const int& block [[buffer(7)]],         // blockDim = head_dim
                              device float* attn [[buffer(8)]],
                              device float* weights [[buffer(9)]],             // nil = not wanted
                              uint3 gpos [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    const long n_ids = (long) step[3];    // kStepWidth
    threadgroup float s_w[32 + QSA_ATTEND_CAP];
    threadgroup float* red = s_w;         // red[0..31] live past the scores/weights: see the launch
    threadgroup float* w = s_w + 32;
    const int h = (int) gpos.x;
    // AN EMPTY SELECTION IS HANDLED HERE (the reference returns zeros; below the selection bound the
    // selection is the identity, so this is reachable only for a genuinely empty cache)
    if (n_ids == 0) {
        for (int i = (int) d; i < head_dim; i += block) attn[(ulong) h * head_dim + i] = 0.0f;
        return;
    }
    const int kv = h / (n_head / n_head_kv);       // ops.cpp L8729: iv2 = iq2 / rv2, NOT h % n_head_kv
    const float scale = 1.0f / metal::precise::sqrt((float) head_dim);
    const int nwarp = (block + 31) >> 5;

    constant const uint16_t* k16 = reinterpret_cast<constant const uint16_t*>(k_scratch);
    constant const uint16_t* v16 = reinterpret_cast<constant const uint16_t*>(v_scratch);
    for (long j = (long) d; j < n_ids; j += block) {
        constant const uint16_t* krow = k16 + (ulong) (j * n_head_kv + kv) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; ++i) acc += f32_from_f16(krow[i]) * q[(ulong) h * head_dim + i];
        w[j] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -QSA_FLT_MAX;    // not -inf, whose double -> float conversion nvcc warns about
    for (long j = (long) d; j < n_ids; j += block) mx = metal::precise::fmax(mx, w[j]);
    mx = qsa_warp_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        mx = (lane < (uint) nwarp) ? red[lane] : -QSA_FLT_MAX;
        mx = qsa_warp_max(mx);
        if (lane == 0) red[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (long j = (long) d; j < n_ids; j += block) {
        const float e = metal::precise::exp(w[j] - mx);
        w[j] = e;
        sum += e;
    }
    sum = qsa_warp_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sum = (lane < (uint) nwarp) ? red[lane] : 0.0f;
        sum = qsa_warp_sum(sum);
        if (lane == 0) red[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (long j = 0; j < n_ids; ++j)
        acc += (w[j] * inv) * f32_from_f16(v16[(ulong) (j * n_head_kv + kv) * head_dim + d]);
    attn[(ulong) h * head_dim + d] = acc;
    if (weights != nullptr) {
        for (long j = (long) d; j < n_ids; j += block) weights[(ulong) h * n_ids + j] = w[j] * inv;
    }
}

// ================= 7. qsa_gate_apply =================

// attn * sigmoid(gate), the gate from the SECOND half of each head's 2*head_dim block, ONE round to fp16.
// The double sigmoid became f32 precise::exp: the comparison's tolerance is the fp16 store's own 2.4e-4.
kernel void qsa_gate_apply_kernel(constant const float* attn [[buffer(0)]],
                                  constant const float* q_full [[buffer(1)]],
                                  constant const int& n_head [[buffer(2)]],
                                  constant const int& head_dim [[buffer(3)]],
                                  device uint* out [[buffer(4)]],               // fp16 patterns
                                  uint i_in [[thread_position_in_grid]]) {
    const ulong i = (ulong) i_in;
    if (i >= (ulong) n_head * (ulong) head_dim) return;
    const uint h = (uint) (i / (ulong) head_dim), drem = (uint) (i % (ulong) head_dim);
    const float g = q_full[(ulong) h * 2 * (ulong) head_dim + (ulong) head_dim + drem];   // the SECOND half
    const float sig = 1.0f / (1.0f + metal::precise::exp(-g));
    reinterpret_cast<device uint16_t*>(out)[i] = (uint16_t) f16_from_f32(attn[i] * sig);
}

// the same multiply WITHOUT the fp16 round, for the caller whose weight wants Q8_K
kernel void qsa_gate_apply_f32_kernel(constant const float* attn [[buffer(0)]],
                                      constant const float* q_full [[buffer(1)]],
                                      constant const int& n_head [[buffer(2)]],
                                      constant const int& head_dim [[buffer(3)]],
                                      device float* out [[buffer(4)]],
                                      uint i_in [[thread_position_in_grid]]) {
    const ulong i = (ulong) i_in;
    if (i >= (ulong) n_head * (ulong) head_dim) return;
    const uint h = (uint) (i / (ulong) head_dim), drem = (uint) (i % (ulong) head_dim);
    const float g = q_full[(ulong) h * 2 * (ulong) head_dim + (ulong) head_dim + drem];
    const float sig = 1.0f / (1.0f + metal::precise::exp(-g));
    out[i] = attn[i] * sig;
}

// =====================================================================================
// The NATIVE indexer bridge (native_qsa_indexer.cu is K18, not yet ported; qsa_parity
// calls these entry points, so the kernels ride THIS file the way the native_mmvq bridge
// rides iq_kernels.mm - DELETE them when K18 ports native_qsa_indexer.cu, or the symbols
// collide).  Numerical contract: llama.cpp's set-rows F16 storage + F32 pooling + text
// iM-RoPE, all f32 on both backends - the CUDA template <bool TAB> becomes a runtime flag
// with null table pointers, exactly as native_rope.metal spells it.
constant const int QSA_NQI_D = 128;      // idx_dim
constant const int QSA_NQI_R = 4;        // idx_block
constant const int QSA_NQI_ROT = 64;     // n_rot
constant const int QSA_NQI_THREADS = 256;

// rope_scaling.hpp's rope_yarn_ramp + rope_scaled_angle, transcribed (the analytic paths' pinned ggml
// arithmetic - the same spellings native_rope.metal carries)
static inline float qsa_rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

static inline void qsa_rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                                         float ext_factor, float mscale_in, int pair, thread float& cos_out,
                                         thread float& sin_out) {
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = qsa_rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * metal::precise::log(1.0f / freq_scale);
    }
    cos_out = metal::precise::cos(theta) * mscale;
    sin_out = metal::precise::sin(theta) * mscale;
}

// the rotation of one dim of a pooled row: the table when one applies (rope_tab_cs inlined), else the
// analytic angle - pos 0 keeps its zero angle and still carries YaRN's mscale through cos(0)
static inline float qsa_nqi_rotate(threadgroup const float* values, int d, int rope_pos, float theta_scale,
                                   float freq_scale, float corr_low, float corr_high, float ext_factor,
                                   float mscale, constant const int* mtab, bool zero_pos,
                                   constant const float* tab_cos, constant const float* tab_sin, int tab_max_pos) {
    float y = values[d];
    if (d < QSA_NQI_ROT) {
        const int pair = d % (QSA_NQI_ROT / 2);
        const int p = zero_pos ? 0 : qsa_mrope_pos(mtab, rope_pos, pair);
        float c, s;
        if (tab_cos != nullptr && p >= 0 && p < tab_max_pos) {
            c = tab_cos[(ulong) p * 32 + (uint) pair];
            s = tab_sin[(ulong) p * 32 + (uint) pair];
        } else {
            const float theta_extrap = (float) p * metal::precise::pow(theta_scale, (float) pair);
            qsa_rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        const float a = values[pair], z = values[pair + QSA_NQI_ROT / 2];
        y = d < QSA_NQI_ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}

static inline float qsa_nqi_warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += simd_shuffle_xor(x, (uint) offset);
    return x;
}

// the block-wide sum of squares of the mean: two warp-level butterflies through partials[32] (all 8 warps
// write theirs - warps past D write 0 - so the second butterfly reads only initialized slots)
static inline float qsa_nqi_norm_scale(threadgroup float* partials, float mean, uint d, uint lane, uint sg) {
    float square_sum = d < (uint) QSA_NQI_D ? mean * mean : 0.0f;
    square_sum = qsa_nqi_warp_sum(square_sum);
    if (lane == 0) partials[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square_sum = lane < QSA_NQI_THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = qsa_nqi_warp_sum(square_sum);
    return square_sum;
}

kernel void qsa_native_append_kernel(constant const float* raw [[buffer(0)]],
                                     constant const int* pos_dev [[buffer(1)]],
                                     constant const int& pos_base [[buffer(2)]],
                                     constant const float* gamma [[buffer(3)]],
                                     constant const float& epsilon [[buffer(4)]],
                                     device float* tail [[buffer(5)]],
                                     device float* dead [[buffer(6)]],
                                     device float* pooled [[buffer(7)]],
                                     device int* block_pos [[buffer(8)]],
                                     constant const int& max_cells [[buffer(9)]],
                                     constant const float& theta_scale [[buffer(10)]],
                                     constant const float& freq_scale [[buffer(11)]],
                                     constant const float& corr_low [[buffer(12)]],
                                     constant const float& corr_high [[buffer(13)]],
                                     constant const float& ext_factor [[buffer(14)]],
                                     constant const float& mscale [[buffer(15)]],
                                     constant const int* mtab [[buffer(16)]],
                                     constant const float* tab_cos [[buffer(17)]],
                                     constant const float* tab_sin [[buffer(18)]],
                                     constant const int& tab_max_pos [[buffer(19)]],
                                     uint d [[thread_position_in_grid]]) {
    const int pos = pos_dev[0];
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % QSA_NQI_R;
    float incoming = 0.0f;
    if (d < (uint) QSA_NQI_D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32
        incoming = f32_from_f16(f16_from_f32(raw[d]));
        if (slot < QSA_NQI_R - 1) tail[(ulong) slot * QSA_NQI_D + d] = incoming;
    }
    if (pos != 0 && slot != QSA_NQI_R - 1) return;
    threadgroup float values[QSA_NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) QSA_NQI_D) {
        // the spare's four gather indices all name cell zero; completed blocks use chronological slices
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < QSA_NQI_R; ++j)
            sum = sum + (pos == 0 || j == QSA_NQI_R - 1 ? incoming : tail[(ulong) j * QSA_NQI_D + d]);
        mean = fma(0.25f, sum, 0.0f);    // SCALE includes a zero bias
    }
    const float square_sum = qsa_nqi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) QSA_NQI_D + epsilon);
    if (d < (uint) QSA_NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) QSA_NQI_D) return;
    const int b = pos / QSA_NQI_R;
    const int rope_pos = pos == 0 ? 0 : pos_base + QSA_NQI_R * b;
    const float y = qsa_nqi_rotate(values, (int) d, rope_pos, theta_scale, freq_scale, corr_low, corr_high,
                                   ext_factor, mscale, mtab, pos == 0, tab_cos, tab_sin, tab_max_pos);
    pooled[(ulong) b * QSA_NQI_D + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[(ulong) (b + 1) * QSA_NQI_D + d] = dead[d];
    if (d == 0 && pos != 0) block_pos[0] = rope_pos;
}

// cell 0 of a sequence: the spare (every gather index names cell 0), written to pooled[0] and dead
kernel void qsa_native_append_first_kernel(constant const float* raw [[buffer(0)]],
                                           constant const float* gamma [[buffer(1)]],
                                           constant const float& epsilon [[buffer(2)]],
                                           device float* dead [[buffer(3)]],
                                           device float* pooled [[buffer(4)]],
                                           constant const float& theta_scale [[buffer(5)]],
                                           constant const float& freq_scale [[buffer(6)]],
                                           constant const float& corr_low [[buffer(7)]],
                                           constant const float& corr_high [[buffer(8)]],
                                           constant const float& ext_factor [[buffer(9)]],
                                           constant const float& mscale [[buffer(10)]],
                                           constant const int* mtab [[buffer(11)]],
                                           constant const float* tab_cos [[buffer(12)]],
                                           constant const float* tab_sin [[buffer(13)]],
                                           constant const int& tab_max_pos [[buffer(14)]],
                                           uint d [[thread_position_in_grid]]) {
    threadgroup float values[QSA_NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) QSA_NQI_D) {
        const float incoming = f32_from_f16(f16_from_f32(raw[d]));
        float sum = incoming;
#pragma unroll
        for (int j = 1; j < QSA_NQI_R; ++j) sum = sum + incoming;
        mean = fma(0.25f, sum, 0.0f);
    }
    const float square_sum = qsa_nqi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) QSA_NQI_D + epsilon);
    if (d < (uint) QSA_NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) QSA_NQI_D) return;
    const float y = qsa_nqi_rotate(values, (int) d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
                                   mscale, mtab, true, tab_cos, tab_sin, tab_max_pos);
    pooled[d] = y;
    dead[d] = y;
}

// C-2: the batched append's completed blocks - each with the single append's arithmetic, keys rounded
// through F16, summed tail[0]+tail[1]+tail[2]+incoming in that order, the RMS norm, gamma, the rotation
kernel void qsa_native_append_blocks_kernel(constant const float* raw [[buffer(0)]],
                                            constant const long& n [[buffer(1)]],
                                            constant const long& p0 [[buffer(2)]],
                                            constant const int& pos_base [[buffer(3)]],
                                            constant const float* gamma [[buffer(4)]],
                                            constant const float& epsilon [[buffer(5)]],
                                            constant const float* tail [[buffer(6)]],
                                            constant const float* dead [[buffer(7)]],
                                            device float* pooled [[buffer(8)]],
                                            device int* block_pos [[buffer(9)]],
                                            constant const long& first_block [[buffer(10)]],
                                            constant const long& last_block [[buffer(11)]],
                                            constant const float& theta_scale [[buffer(12)]],
                                            constant const float& freq_scale [[buffer(13)]],
                                            constant const float& corr_low [[buffer(14)]],
                                            constant const float& corr_high [[buffer(15)]],
                                            constant const float& ext_factor [[buffer(16)]],
                                            constant const float& mscale [[buffer(17)]],
                                            constant const int* mtab [[buffer(18)]],
                                            constant const float* tab_cos [[buffer(19)]],
                                            constant const float* tab_sin [[buffer(20)]],
                                            constant const int& tab_max_pos [[buffer(21)]],
                                            uint3 gpos [[threadgroup_position_in_grid]],
                                            uint d [[thread_index_in_threadgroup]]) {
    const long b = first_block + (long) gpos.x;
    threadgroup float values[QSA_NQI_D];
    threadgroup float partials[32];
    const uint lane = d % 32;
    float mean = 0.0f;
    if (d < (uint) QSA_NQI_D) {
        // k(j): a row of this batch (cell >= p0) or the tail the previous batch left (cell < p0)
        float sum = b * QSA_NQI_R + 0 >= p0
                        ? f32_from_f16(f16_from_f32(raw[(ulong) (b * QSA_NQI_R + 0 - p0) * QSA_NQI_D + d]))
                        : tail[(ulong) 0 * QSA_NQI_D + d];
#pragma unroll
        for (int j = 1; j < QSA_NQI_R; ++j) {
            const long cell = b * QSA_NQI_R + j;
            const float k = cell >= p0 ? f32_from_f16(f16_from_f32(raw[(ulong) (cell - p0) * QSA_NQI_D + d]))
                                       : tail[(ulong) j * QSA_NQI_D + d];
            sum = sum + k;
        }
        mean = fma(0.25f, sum, 0.0f);
    }
    const float square_sum = qsa_nqi_norm_scale(partials, mean, d, lane, d / 32);
    const float scale = metal::precise::rsqrt(square_sum / (float) QSA_NQI_D + epsilon);
    if (d < (uint) QSA_NQI_D) values[d] = scale * mean * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint) QSA_NQI_D) return;
    const int rope_pos = pos_base + QSA_NQI_R * (int) b;
    pooled[(ulong) b * QSA_NQI_D + d] = qsa_nqi_rotate(values, (int) d, rope_pos, theta_scale, freq_scale,
                                                       corr_low, corr_high, ext_factor, mscale, mtab, false,
                                                       tab_cos, tab_sin, tab_max_pos);
    if (b == last_block) {
        pooled[(ulong) (b + 1) * QSA_NQI_D + d] = dead[d];
        if (d == 0) block_pos[0] = rope_pos;
    }
}

// the tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any
kernel void qsa_native_append_tail_kernel(constant const float* raw [[buffer(0)]],
                                          constant const long& n [[buffer(1)]],
                                          constant const long& p0 [[buffer(2)]],
                                          device float* tail [[buffer(3)]],
                                          uint3 gpos [[threadgroup_position_in_grid]],
                                          uint d [[thread_index_in_threadgroup]]) {
    const int s = (int) gpos.x;
    if (d >= (uint) QSA_NQI_D) return;
    const long last = p0 + n - 1;
    const long cell = last - ((last % QSA_NQI_R) - s + QSA_NQI_R) % QSA_NQI_R;   // last cell <= last, cell%4==s
    if (cell < p0) return;
    tail[(ulong) s * QSA_NQI_D + d] = f32_from_f16(f16_from_f32(raw[(ulong) (cell - p0) * QSA_NQI_D + d]));
}

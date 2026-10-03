// src/kernels/metal/verify_kernels.metal - the port of src/kernels/cuda/verify_kernels.cu's kernels (K21):
// the verify window's GDN multi kernels, the MTP-draft helpers and the resident-expert plan.
//
// The GDN kernels' arithmetic is fused_gdn.metal's own spellings, copied per expression, so a window token
// is the single-token kernel bit for bit (the header's contract): the conv window sum and SiLU, the L2 tree
// over the q/k heads, the eight ordered bf16 fma of `dot8`, the softplus series seam (MSL has no log1p) and
// the step+norm recurrence's barrier ladder.
//
// TWO RESTRUCTURES, both forced by measured behaviour of this GPU (docs/PORT_METAL/PROGRESS.md):
//   * RULE 10 (gdn_step_norm_multi): the CUDA kernel keeps its 32 state rows in a per-thread REGISTER array
//     across the token loop; that array is what pinned fused_gdn's first port at 384 threads under its
//     512-thread launch - a dispatch the runtime drops silently.  The port keeps the same 512-thread shape
//     and stages the running state in a private DEVICE buffer (the launcher's, laid out exactly like
//     `state`): token 0 reads `state` itself, every later token reads the staging copy, and the commit
//     (n_keep != null) writes it back at the end.  Each thread re-reads the same bits its register copy
//     held - the fma chains per token are the CUDA ones - only the storage of the recurrence moves.
//   * RULE 9 (fetch_blobs): its CUDA kernel dereferences pointers loaded from device memory, which is not
//     dereferenceable here - that entry is restructured on the host (verify_kernels.mm, the moe_grouped_s2
//     precedent) and has NO kernel in this file.
//
// MSL notes: the mapped-host spin flags read as relaxed device atomics with a device fence after - the
// doorbell_wait stand-in for volatile + __threadfence_system (elementwise.metal); a "volatile" store to
// mapped host memory (mtp_select's out/out_p) is a plain store - the CPU reads it after a stream sync.
#include "strata_port.metalh"

constant const int VR_S = 128;                    // GDN state size (the .cu's S)
constant const int VR_RG = 4;                     // row groups
constant const int VR_RPG = VR_S / VR_RG;         // 32 rows per thread

static inline float vrfy_sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// fused_gdn.metal's fgdn_softplus_f: log1p(exp(x)) with the large-x branch; below e ~ 0.05 the Horner'd
// series carries what f32's log(1+e) would cancel away (elementwise.metal's measured seam)
static inline float vrfy_softplus_f(float x) {
    if (x > 20.0f) return x;
    const float e = metal::precise::exp(x);
    if (e < 0.05f) return e * (1.0f - e * (0.5f - e * (1.0f / 3.0f - e * 0.25f)));
    return metal::precise::log(1.0f + e);
}

// fused_gdn.metal's fgdn_dot8: the .cu's eight ordered fmaf per uint4 - bf16 pair k is (word k's low half,
// word k's high half) against x's elements 2k and 2k+1, chained straight into the accumulator
static inline float vrfy_dot8(float acc, uint4 wv, float4 xa, float4 xb) {
    acc = fma(as_type<float>(wv.x << 16), xa.x, acc);
    acc = fma(as_type<float>(wv.x & 0xffff0000u), xa.y, acc);
    acc = fma(as_type<float>(wv.y << 16), xa.z, acc);
    acc = fma(as_type<float>(wv.y & 0xffff0000u), xa.w, acc);
    acc = fma(as_type<float>(wv.z << 16), xb.x, acc);
    acc = fma(as_type<float>(wv.z & 0xffff0000u), xb.y, acc);
    acc = fma(as_type<float>(wv.w << 16), xb.z, acc);
    acc = fma(as_type<float>(wv.w & 0xffff0000u), xb.w, acc);
    return acc;
}

// ---- the GDN verify window ---------------------------------------------------------------------------------

// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
// One 128-thread group per S channels, .y of the group position the token (rule 6); history is NOT written.
kernel void gdn_conv_l2_multi_kernel(constant const float* hist [[buffer(0)]],
                                     constant const float* qkv [[buffer(1)]],
                                     constant const float* w [[buffer(2)]],
                                     device float* h [[buffer(3)]],
                                     constant const int& C [[buffer(4)]],
                                     constant const int& qk_heads [[buffer(5)]],
                                     constant const float& eps [[buffer(6)]],
                                     constant const int& t_begin [[buffer(7)]],
                                     uint3 gpos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[VR_S / 32];
    const int t = t_begin + (int) gpos.y;
    const int c = (int) gpos.x * VR_S + (int) tid;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[(ulong) c * 3 + (uint) src] : qkv[(ulong) (src - 3) * (uint) C + (uint) c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(ulong) (uint) t * (uint) C + (uint) c];
    float sum = v0 * w[(ulong) c * 4] + v1 * w[(ulong) c * 4 + 1] + v2 * w[(ulong) c * 4 + 2] +
                x * w[(ulong) c * 4 + 3];
    float y = sum / (1.0f + metal::precise::exp(-sum));
    if ((int) gpos.x < qk_heads) {               // uniform per group, like the .cu's blockIdx test
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += simd_shuffle_xor(sq, (uint) o);
        if ((tid & 31u) == 0u) part[tid >> 5] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= metal::precise::rsqrt(ss + eps);
    }
    h[(ulong) (uint) t * (uint) C + (uint) c] = y;
}

// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
kernel void gdn_conv_commit_kernel(device float* hist [[buffer(0)]],
                                   constant const float* qkv [[buffer(1)]],
                                   constant const int& C [[buffer(2)]],
                                   constant const int* n_keep [[buffer(3)]],
                                   uint x [[thread_position_in_grid]]) {
    if (x >= (uint) C) return;
    const int c = (int) x;
    const int n = n_keep[0];
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[(ulong) c * 3 + (uint) src] : qkv[(ulong) (src - 3) * (uint) C + (uint) c];
    }
    hist[(ulong) c * 3] = seq[0];
    hist[(ulong) c * 3 + 1] = seq[1];
    hist[(ulong) c * 3 + 2] = seq[2];
}

// alpha/beta for T columns of x: one warp per row of the (2*h_v, n_embd) bf16 pair, T accumulators per
// thread, lane 0 running the epilogues.  The bf16 row strides in uint16 ELEMENTS before the uint4 view.
kernel void gdn_ab_multi_kernel(constant const float* x [[buffer(0)]],
                                constant const uint16_t* wa [[buffer(1)]],
                                constant const uint16_t* wb [[buffer(2)]],
                                constant const float* dt [[buffer(3)]],
                                constant const float* ssm_a [[buffer(4)]],
                                device float* gate [[buffer(5)]],
                                device float* beta [[buffer(6)]],
                                constant const int& n [[buffer(7)]],
                                constant const int& h_v [[buffer(8)]],
                                constant const int& T [[buffer(9)]],
                                uint3 gpos [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const int row = (int) gpos.x * 8 + (int) sg;    // blockIdx.x * 8 + (threadIdx.x >> 5)
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    constant const uint4* w4 = reinterpret_cast<constant const uint4*>((is_beta ? wb : wa) +
                                                                       (ulong) r * (ulong) n);
    float acc[8];                                   // kVerifyMaxT
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int j = (int) lane; j < n / 8; j += 32) {
        const uint4 wv = w4[j];                     // __ldg: the constant address space caches it
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= T) break;
            constant const float* xt = x + (ulong) t * (ulong) n;
            const float4 xa = *(constant const float4*) (xt + (ulong) j * 8);
            const float4 xb = *(constant const float4*) (xt + (ulong) j * 8 + 4);
            acc[t] = vrfy_dot8(acc[t], wv, xa, xb);
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += simd_shuffle_xor(a, (uint) o);
        if (lane != 0u) continue;
        if (is_beta) {
            beta[(ulong) t * (ulong) h_v + (uint) r] = vrfy_sigmoid_f(a);
        } else {
            const float v = a + dt[r];
            const float sp = vrfy_softplus_f(v);
            gate[(ulong) t * (ulong) h_v + (uint) r] = sp * ssm_a[r];
        }
    }
}

// The recurrence + output norm for T tokens.  One 512-thread group per value head, the (128, 4) CUDA block
// read back as col = tid & 127, rg = tid >> 7 (the fused_gdn mapping).  With n_keep == null the state is
// read and NOT written (verify); otherwise the first *n_keep tokens run and the state is written (commit).
// The running state lives in `shadow` (see the file comment - the rule-10 restructure); buffer(14) is the
// one argument the CUDA kernel does not have.
#define VR_STEP_PARAMS \
    device float* state [[buffer(0)]], constant const float* hbuf [[buffer(1)]], constant const int& C [[buffer(2)]], \
    constant const float* gate [[buffer(3)]], constant const float* beta [[buffer(4)]], constant const float* z [[buffer(5)]], \
    constant const float* gamma [[buffer(6)]], constant const float& eps [[buffer(7)]], device float* y [[buffer(8)]], \
    constant const int& h_k [[buffer(9)]], constant const int& h_v [[buffer(10)]], constant const int& T [[buffer(11)]], \
    constant const int* n_keep [[buffer(12)]], constant const int& t_out_begin [[buffer(13)]], \
    device float* shadow [[buffer(14)]], uint3 gpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
// STAGE_LAST = false (gdn_step_norm_multi_tail_kernel): the last token's state is not staged in `shadow` - no
// later token reads it - and a commit writes it straight to `state`, where the staged kernel copied it back
// afterwards. Every value is the same float; only redundant device traffic goes.
template<bool STAGE_LAST>
static inline void vr_gdn_step_body(device float* state, constant const float* hbuf, const int C, constant const float* gate,
                                    constant const float* beta, constant const float* z, constant const float* gamma,
                                    const float eps, device float* y, const int h_k, const int h_v, const int T,
                                    constant const int* n_keep, const int t_out_begin, device float* shadow, uint3 gpos,
                                    uint tid, threadgroup float* sk, threadgroup float* sq,
                                    threadgroup float (*red)[VR_S], threadgroup float* wsum) {
    const int head = (int) gpos.x;
    const int col = (int) (tid & 127u);          // threadIdx.x, 0..127
    const int rg = (int) (tid >> 7);             // threadIdx.y, 0..3
    const int qh = head % h_k;
    const int qk = VR_S * h_k;                   // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = VR_S * h_v;
    const int n = n_keep != nullptr ? n_keep[0] : T;
    device float* base = state + ((ulong) (rg * VR_RPG) * (ulong) h_v + (ulong) head) * VR_S + (uint) col;
    device float* srun = shadow + ((ulong) (rg * VR_RPG) * (ulong) h_v + (ulong) head) * VR_S + (uint) col;
    const ulong row_stride = (ulong) h_v * VR_S;
    for (int t = 0; t < n; ++t) {
        constant const float* ht = hbuf + (ulong) (uint) t * (uint) C;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous token is done with sk/sq/red/wsum
        if (tid < (uint) VR_S) {
            sk[tid] = ht[(ulong) (qk + qh * VR_S) + tid];
            sq[tid] = ht[(ulong) (qh * VR_S) + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float g = metal::precise::exp(gate[(ulong) (uint) t * (ulong) h_v + (ulong) head]);
        device float* s_at = t == 0 ? base : srun;        // token 0 reads the caller's state itself
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < VR_RPG; ++r) kv = fma(s_at[r * row_stride], sk[rg * VR_RPG + r], kv);
        red[rg][col] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[(ulong) (2 * qk + head * VR_S) + (uint) col] - g * kv_col) *
                            beta[(ulong) (uint) t * (ulong) h_v + (ulong) head];
        float o = 0.0f;
        const bool last = !STAGE_LAST && t == n - 1;
        device float* s_to = last ? (n_keep != nullptr ? base : nullptr) : srun;
#pragma unroll
        for (int r = 0; r < VR_RPG; ++r) {
            const float s = fma(g, s_at[r * row_stride], sk[rg * VR_RPG + r] * delta);
            o = fma(s, sq[rg * VR_RPG + r], o);
            if (s_to != nullptr) s_to[r * row_stride] = s;   // the register array's job, staged (rule 10)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * metal::precise::rsqrt((float) VR_S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += simd_shuffle_xor(sq_part, (uint) o2);
        if ((tid & 31u) == 0u) wsum[tid >> 5] = sq_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = metal::precise::rsqrt(ss / (float) VR_S + eps);
            const float zz = z[(ulong) (uint) t * (ulong) value_dim + (ulong) (head * VR_S) + (uint) col];
            y[(ulong) (uint) t * (ulong) value_dim + (ulong) (head * VR_S) + (uint) col] =
                oc * scale * gamma[col] * vrfy_sigmoid_f(zz);
        }
    }
    if (STAGE_LAST && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < VR_RPG; ++r) base[r * row_stride] = srun[r * row_stride];
    }
}
#define VR_STEP_KERNEL(NAME, STAGE) \
kernel void NAME(VR_STEP_PARAMS) { \
    threadgroup float sk[VR_S], sq[VR_S]; \
    threadgroup float red[VR_RG][VR_S]; \
    threadgroup float wsum[VR_S * VR_RG / 32]; \
    vr_gdn_step_body<STAGE>(state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep, t_out_begin, shadow, \
                            gpos, tid, sk, sq, red, wsum); \
}
VR_STEP_KERNEL(gdn_step_norm_multi_kernel, true)
VR_STEP_KERNEL(gdn_step_norm_multi_tail_kernel, false)



// ---- the MTP draft layer -----------------------------------------------------------------------------------

// Rows of the S2/S4/S8 embedding for T token ids read from device memory: .y of the group position is the
// token (rule 6), the element the global x position (rule 7, block size as a scalar argument).
kernel void embedding_gather_dev_kernel(constant const uint8_t* codes [[buffer(0)]],
                                        constant const float* scales [[buffer(1)]],
                                        constant const float* offsets [[buffer(2)]],
                                        constant const int* tokens [[buffer(3)]],
                                        constant const long& n [[buffer(4)]],
                                        constant const int& code_bits [[buffer(5)]],
                                        constant const int& code_bias [[buffer(6)]],
                                        constant const int& group_elems [[buffer(7)]],
                                        constant const ulong& row_codes [[buffer(8)]],
                                        constant const ulong& row_groups [[buffer(9)]],
                                        device float* out [[buffer(10)]],
                                        constant const uint& block [[buffer(11)]],
                                        uint2 gpos [[threadgroup_position_in_grid]],
                                        uint t [[thread_index_in_threadgroup]]) {
    const int tk = (int) gpos.y;
    const long i = (long) gpos.x * (long) block + (long) t;
    if (i >= n) return;
    const ulong token = (ulong) (uint) tokens[tk];
    constant const uint8_t* c = codes + token * row_codes;
    constant const float* sc = scales + token * row_groups;
    constant const float* of = offsets != nullptr ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const uint mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const long group = i / group_elems;
    const float product = (float) (code + code_bias) * sc[group];
    out[(ulong) tk * (ulong) n + (ulong) i] = product + (of != nullptr ? of[group] : 0.0f);
}

kernel void broadcast_streams_kernel(constant const float* x [[buffer(0)]],
                                     device float* R [[buffer(1)]],
                                     constant const long& n [[buffer(2)]],
                                     constant const int& hc [[buffer(3)]],
                                     constant const uint& block [[buffer(4)]],
                                     uint2 gpos [[threadgroup_position_in_grid]],
                                     uint t [[thread_index_in_threadgroup]]) {
    const long i = (long) gpos.x * (long) block + (long) t;
    if (i >= n * hc) return;
    R[(ulong) gpos.y * (ulong) n * (ulong) hc + (ulong) i] = x[(ulong) gpos.y * (ulong) n + (ulong) (i % n)];
}

kernel void copy_indexed_kernel(device float* dst [[buffer(0)]],
                                constant const float* src [[buffer(1)]],
                                constant const long& stride [[buffer(2)]],
                                constant const int* index [[buffer(3)]],
                                constant const long& n [[buffer(4)]],
                                uint tpg [[threads_per_grid]],
                                uint gid [[thread_position_in_grid]]) {
    const int idx = index[0];
    if (idx < 0) return;
    for (long i = (long) gid; i < n; i += (long) tpg)
        dst[i] = src[(ulong) idx * (ulong) stride + (ulong) i];
}

// ptr[k] = base + k * blob_bytes for k < *n; the base rides as an INTEGER (the s2_expert_grouped rule-9
// pattern - the value is written, never dereferenced, and the table's readers on this backend resolve it
// on the host).
kernel void rebase_ptrs_kernel(device ulong* ptr [[buffer(0)]],
                               constant const int* n [[buffer(1)]],
                               constant const ulong& base [[buffer(2)]],
                               constant const long& bytes [[buffer(3)]],
                               uint k [[thread_position_in_grid]]) {
    if (k < (uint) n[0]) ptr[k] = base + (ulong) k * (ulong) bytes;
}

kernel void add_streams_broadcast_kernel(constant const float* h [[buffer(0)]],
                                         constant const float* e [[buffer(1)]],
                                         device float* R [[buffer(2)]],
                                         constant const long& n [[buffer(3)]],
                                         constant const int& hc [[buffer(4)]],
                                         constant const uint& block [[buffer(5)]],
                                         uint2 gpos [[threadgroup_position_in_grid]],
                                         uint t [[thread_index_in_threadgroup]]) {
    const long i = (long) gpos.x * (long) block + (long) t;
    if (i >= n * hc) return;
    R[(ulong) gpos.y * (ulong) n * (ulong) hc + (ulong) i] =
        h[(ulong) gpos.y * (ulong) n * (ulong) hc + (ulong) i] + e[(ulong) gpos.y * (ulong) n + (ulong) (i % n)];
}

kernel void ident_hits_kernel(constant const int* ids [[buffer(0)]],
                              constant const int& n [[buffer(1)]],
                              device int* slot [[buffer(2)]],
                              device int* dst [[buffer(3)]],
                              device int* count [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) { slot[i] = ids[i]; dst[i] = (int) i; }
    if (i == 0) count[0] = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is
// 2100 bytes.  The .cu's gather_rows_kernel<E> template becomes three kernels; the bodies are identical.
kernel void gather_rows_16_kernel(constant const uint4* src [[buffer(0)]],
                                  constant const long& row_e [[buffer(1)]],
                                  constant const int* ids [[buffer(2)]],
                                  constant const long& n [[buffer(3)]],
                                  device uint4* dst [[buffer(4)]],
                                  uint tpg [[threads_per_grid]],
                                  uint gid [[thread_position_in_grid]]) {
    const long total = n * row_e;
    for (long i = (long) gid; i < total; i += (long) tpg) {
        const long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long) ids[r] * row_e + o];
    }
}
kernel void gather_rows_4_kernel(constant const uint* src [[buffer(0)]],
                                 constant const long& row_e [[buffer(1)]],
                                 constant const int* ids [[buffer(2)]],
                                 constant const long& n [[buffer(3)]],
                                 device uint* dst [[buffer(4)]],
                                 uint tpg [[threads_per_grid]],
                                 uint gid [[thread_position_in_grid]]) {
    const long total = n * row_e;
    for (long i = (long) gid; i < total; i += (long) tpg) {
        const long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long) ids[r] * row_e + o];
    }
}
kernel void gather_rows_1_kernel(constant const uint8_t* src [[buffer(0)]],
                                 constant const long& row_e [[buffer(1)]],
                                 constant const int* ids [[buffer(2)]],
                                 constant const long& n [[buffer(3)]],
                                 device uint8_t* dst [[buffer(4)]],
                                 uint tpg [[threads_per_grid]],
                                 uint gid [[thread_position_in_grid]]) {
    const long total = n * row_e;
    for (long i = (long) gid; i < total; i += (long) tpg) {
        const long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long) ids[r] * row_e + o];
    }
}

kernel void map_ids_kernel(device int* ids [[buffer(0)]],
                           constant const int* table [[buffer(1)]],
                           constant const int& n [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < (uint) n) ids[i] = table[ids[i]];
}

// probs[t] = softmax(logits[t])[ids[t]] - the probability of each row's argmax; one 1024-thread group per
// row, part[32] the per-warp partials of the sum.
kernel void row_top_prob_kernel(constant const float* logits [[buffer(0)]],
                                constant const int& n_vocab [[buffer(1)]],
                                constant const int* ids [[buffer(2)]],
                                device float* probs [[buffer(3)]],
                                constant const uint& block [[buffer(4)]],
                                uint3 gpos [[threadgroup_position_in_grid]],   // blockIdx.x = the row
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    const int t = (int) gpos.x;
    constant const float* l = logits + (ulong) t * (ulong) n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = (int) tid; i < n_vocab; i += (int) block) s += metal::precise::exp(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += simd_shuffle_xor(s, (uint) o);
    if (lane == 0u) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (block >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
// row = *row_dev (device memory).  `out`/`out_p` may be mapped host memory - the .cu's volatile stores are
// plain stores here (the CPU reads them after a stream sync, which the caller does).
kernel void mtp_select_kernel(constant const float* R_src [[buffer(0)]],
                              constant const long& stride [[buffer(1)]],
                              constant const int* ids [[buffer(2)]],
                              constant const int* row_dev [[buffer(3)]],
                              device float* R_dst [[buffer(4)]],
                              device int* tok_dst [[buffer(5)]],
                              device int* out [[buffer(6)]],
                              constant const int& j [[buffer(7)]],
                              constant const float* probs [[buffer(8)]],
                              device float* out_p [[buffer(9)]],
                              uint tpg [[threads_per_grid]],
                              uint gid [[thread_position_in_grid]]) {
    const int row = row_dev[0];
    for (long i = (long) gid; i < stride; i += (long) tpg)
        R_dst[i] = R_src[(ulong) row * (ulong) stride + (ulong) i];
    if (gid == 0) {
        const int tok = ids[row];
        tok_dst[0] = tok;
        if (out != nullptr) out[j] = tok;
        if (probs != nullptr && out_p != nullptr) out_p[j] = probs[row];
    }
}

kernel void dense_steps_kernel(constant const int* cells [[buffer(0)]],
                               constant const int& n [[buffer(1)]],
                               device int* steps [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= (uint) n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

// A sliding attention window: q (.y of the group position) owns one step record; the selection becomes the
// last `window` cells and the record's width = the count.  The x threads stride the whole width.
kernel void window_ids_kernel(device int* steps [[buffer(0)]],
                              constant const int& window [[buffer(1)]],
                              device int* ids [[buffer(2)]],
                              constant const long& stride [[buffer(3)]],
                              constant const uint& xthreads [[buffer(4)]],
                              uint2 gpos [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
    const int q = (int) gpos.y;
    device int* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = (int) (gpos.x * 256u + t); j < width; j += (int) xthreads)
        ids[(ulong) q * (ulong) stride + (ulong) (uint) j] = start + j;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gpos.x == 0u && t == 0u) st[3] = width;
}

// ---- the doorbells and the resident plan -------------------------------------------------------------------

// Spin until *flag >= value (a mapped host flag, raised by a CPU store): relaxed device atomics with a
// device fence after - the doorbell_wait stand-in for volatile + __threadfence_system (elementwise.metal).
kernel void wait_flag_ge_kernel(const device atomic_uint* flag [[buffer(0)]],
                                constant const uint& value [[buffer(1)]]) {
    while (atomic_load_explicit(flag, memory_order_relaxed) < value) {}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
}

kernel void wait_flag_ge_or_kernel(const device atomic_uint* flag [[buffer(0)]],
                                   constant const uint& value [[buffer(1)]],
                                   const device atomic_uint* skip [[buffer(2)]]) {
    if (atomic_load_explicit(skip, memory_order_relaxed) == value) return;
    while (atomic_load_explicit(flag, memory_order_relaxed) < value) {}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
}

kernel void copy_i32_unless_kernel(device int* dst [[buffer(0)]],
                                   constant const int* src [[buffer(1)]],
                                   constant const int& n [[buffer(2)]],
                                   const device atomic_uint* skip [[buffer(3)]],
                                   constant const uint& value [[buffer(4)]],
                                   constant const uint& block [[buffer(5)]],
                                   uint t [[thread_index_in_threadgroup]]) {
    if (atomic_load_explicit(skip, memory_order_relaxed) == value) return;
    for (int i = (int) t; i < n; i += (int) block) dst[i] = src[i];
}

kernel void copy_or_zero_kernel(device float4* dst [[buffer(0)]],
                                constant const float4* src [[buffer(1)]],
                                constant const long& n4 [[buffer(2)]],
                                const device atomic_uint* skip [[buffer(3)]],
                                constant const uint& value [[buffer(4)]],
                                uint tpg [[threads_per_grid]],
                                uint gid [[thread_position_in_grid]]) {
    const bool zero = atomic_load_explicit(skip, memory_order_relaxed) == value;
    for (long i = (long) gid; i < n4; i += (long) tpg)
        dst[i] = zero ? float4(0.0f) : src[i];
}

// One thread: the host's exact resident-plan loop.  The cache base rides as an INTEGER and the kernel only
// WRITES pointer values (base + offset) into the table - on this backend the table's readers resolve them
// on the host (rule 9; s2_expert_grouped.mm's moe_grouped_s2 is the precedent).
kernel void resident_plan_kernel(constant const int* ids [[buffer(0)]],
                                 constant const int& n [[buffer(1)]],
                                 constant const int& k [[buffer(2)]],
                                 constant const int* res [[buffer(3)]],
                                 constant const int& n_expert [[buffer(4)]],
                                 constant const ulong& cache_base [[buffer(5)]],
                                 constant const ulong* slot_off [[buffer(6)]],
                                 constant const long& blob [[buffer(7)]],
                                 device int* pl [[buffer(8)]],
                                 constant const long& capx [[buffer(9)]],
                                 device atomic_uint* skip [[buffer(10)]],
                                 constant const uint& ring [[buffer(11)]]) {
    // at most kVerifyMaxT * 10 entries, the host's exact loop
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert || res[e] < 0) { atomic_store_explicit(skip, 0u, memory_order_relaxed); return; }
    }
    device int* counts = pl;
    device int* start = pl + 4;
    device int* dst = start + capx + 1;
    device int* tok = dst + capx;
    const long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    device ulong* ptr = reinterpret_cast<device ulong*>(pl + ptr_off);
    device int* start2 = pl + ptr_off + 4 * capx;
    int groups = 0, entries = 0;
    for (int i0 = 0; i0 < n; ++i0) {
        bool first = true;
        for (int j = 0; j < i0; ++j) if (ids[j] == ids[i0]) { first = false; break; }
        if (!first) continue;
        const int slot = res[ids[i0]];
        ptr[groups] = cache_base + (slot_off != nullptr ? slot_off[slot] : (ulong) slot * (ulong) blob);
        start[groups] = entries;
        for (int i = i0; i < n; ++i)
            if (ids[i] == ids[i0]) {
                // an entry belongs to i0's group when its first occurrence is i0: the same expert id
                dst[entries] = i;
                tok[entries] = i / k;
                ++entries;
            }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);   // __threadfence before publishing
    atomic_store_explicit(skip, ring, memory_order_relaxed);
}

// ---- the stage profiler's stamp ----------------------------------------------------------------------------
//
// MSL has NO readable GPU clock (%globaltimer / wall_clock64 do not exist), so the stamp is a DEVICE
// COUNTER: one strictly increasing launch index per executed stamp, from a private 8-byte buffer the
// launcher owns.  The counter is read on the device AT EXECUTION TIME, so captured replays keep increasing
// the sequence exactly as executed stamps did - a host-provided or baked value would repeat on every graph
// replay.  The values feed STRATA_VERIFY_PROFILE's stage differences only: on this backend the differences
// count STAMP LAUNCHES between stages, not nanoseconds (documented in verify_kernels.mm).
kernel void gpu_stamp_kernel(device ulong* buf [[buffer(0)]],
                             constant const int& i [[buffer(1)]],
                             device atomic_uint* counter [[buffer(2)]]) {
    // the counter is 32-bit: this toolchain's fetch_add has no 64-bit form (measured at compile time); the
    // stamp value itself is widened to the .cu's unsigned long long contract
    const ulong t = 1ul + (ulong) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    buf[i] = t;
}

// The decode kernels that compute a call's integer sum directly (IQ4_XS / IQ4_NL threadgroup codebook, IQ3_S and
// the Q2_0 resident down projection by float-exact FMA) and the one-pass hyper-connection norm, against the
// original kernels they replace - launched by name on the same inputs, every output compared bit for bit. Also
// checks that the public entry points route to kernels with the original's bits. Random, extreme (largest
// codebook values and activations, so the integer sums reach their bound) and special (inf / NaN / -0 / subnormal
// scales) inputs; output buffers start with different patterns, so an unwritten row cannot pass.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/metal_launch.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::metal::Launch;
std::mt19937 rng(20261003);
cudaStream_t stream{};
long long g_checked = 0, g_diff = 0;

uint16_t half_scale(int mode) {
    const unsigned r = rng();
    if (mode == 2) {
        static const uint16_t special[] = {0x7C00, 0xFC00, 0x7E00, 0x8000, 0x0000, 0x0001, 0x83FF, 0x3C00};
        if (r % 4 == 0) return special[(r >> 2) % 8];
    }
    return uint16_t(((r >> 31) << 15) | ((5 + (r >> 10) % 5) << 10) | (r & 1023));
}

void fill_blocks(std::vector<uint8_t>& w, size_t block, int mode, uint8_t fill) {
    for (auto& v : w) v = mode == 1 ? fill : (uint8_t) rng();
    for (size_t i = 0; i + block <= w.size(); i += block) {
        const uint16_t d = half_scale(mode);
        std::memcpy(w.data() + i, &d, 2);
    }
}

std::vector<uint8_t> q8(int n, int mode) {
    std::vector<uint8_t> x((size_t) (n / 32) * 36);
    for (size_t i = 0; i < x.size(); i += 36) {
        const uint16_t d = mode == 2 && rng() % 5 == 0 ? half_scale(2) : uint16_t((((rng() % 3) + 9) << 10) | (rng() & 1023));
        const uint16_t s = uint16_t(rng() & 0x7bff);
        std::memcpy(x.data() + i, &d, 2);
        std::memcpy(x.data() + i + 2, &s, 2);
        for (int j = 0; j < 32; ++j) x[i + 4 + j] = mode == 1 ? 0x80 : (uint8_t) rng();
    }
    return x;
}

template<class T>
void* upload(const std::vector<T>& v) {
    void* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

float* out_buffer(size_t n, uint8_t pattern) {
    float* p = nullptr;
    cudaMalloc((void**) &p, n * 4);
    std::vector<uint8_t> v(n * 4, pattern);
    cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice);
    return p;
}

std::vector<uint32_t> download(const void* p, size_t n) {
    cudaStreamSynchronize(stream);
    std::vector<uint32_t> v(n);
    cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost);
    return v;
}

void compare(const char* what, const std::string& shape, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    g_checked += (long long) a.size();
    g_diff += d;
    if (d) std::printf("FAIL %s %s: %lld of %zu values differ\n", what, shape.c_str(), d, a.size());
}

// One single-column MMVQ shape: the original kernel, the direct kernel and the public entry point.
// ty: 23 IQ4_XS, 20 IQ4_NL, 21 IQ3_S.
void mmvq_case(int ty, int K, int N, int mode) {
    const size_t block = ty == 23 ? 136 : ty == 20 ? 18 : 110, per = ty == 20 ? 32 : 256;
    const unsigned long rb = (unsigned long) (K / per) * block;
    std::vector<uint8_t> w(rb * N);
    fill_blocks(w, block, mode, ty == 21 ? 0xFF : 0x00);
    void* dw = upload(w);
    void* dx = upload(q8(K, mode));
    float* y0 = out_buffer(N, 0xAA);
    float* y1 = out_buffer(N, 0x55);
    float* y2 = out_buffer(N, 0x33);
    const std::string shape = std::to_string(K) + "x" + std::to_string(N) + " mode " + std::to_string(mode);
    if (ty == 21) {
        const int one = 1;
        Launch a("mmvq_multi_kernel_21_1", (unsigned) ((N + 3) / 4), 1, 1, 32, 4, 1, 0, stream);
        a.buf(dw).scalar(rb).buf(dx).buf(y0).scalar(K).scalar(N).scalar(one); a.done();
        Launch b("mmvq_direct_21_r2", (unsigned) ((N + 7) / 8), 1, 1, 32, 4, 1, 0, stream);
        b.buf(dw).scalar(rb).buf(dx).buf(y1).scalar(K).scalar(N).scalar(one); b.done();
        strata::kernels::iq_mmvq(21, dw, dx, y2, K, N, 1, stream);
    } else {
        const bool small = ty == 23 ? K / 256 < 16 : K / 32 < 64;
        const std::string stem = ty == 23 ? "native_iq4_xs_mmvq_kernel_" : "native_small_mmvq_kernel_iq4_nl_";
        Launch a((stem + (small ? "small" : "large")).c_str(), (unsigned) (small ? (N + 3) / 4 : N), 1, 1, 32, 4, 1, 0,
                 stream);
        a.buf(dw).buf(dx).buf(y0).scalar(K).scalar(N); a.done();
        Launch b(ty == 23 ? "native_iq4_xs_direct_r4" : "native_iq4_nl_direct_r4", (unsigned) ((N + 3) / 4), 1, 1, 32,
                 4, 1, 0, stream);
        b.buf(dw).buf(dx).buf(y1).scalar(K).scalar(N); b.done();
        strata::kernels::native_mmvq(ty, dw, dx, y2, K, N, 1, stream);
        if (ty == 23) {   // the simdgroup-per-row kernel, by name, at every shape
            float* y3 = out_buffer(N, 0x77);
            Launch c("native_iq4_xs_direct_sg4", (unsigned) ((N + 15) / 16), 1, 1, 32, 4, 1, 0, stream);
            c.buf(dw).buf(dx).buf(y3).scalar(K).scalar(N); c.done();
            compare("IQ4_XS sg4", shape, download(y0, N), download(y3, N));
            cudaFree(y3);
        }
    }
    const auto r0 = download(y0, N);
    compare(ty == 23 ? "IQ4_XS direct" : ty == 20 ? "IQ4_NL direct" : "IQ3_S direct", shape, r0, download(y1, N));
    compare(ty == 23 ? "IQ4_XS public" : ty == 20 ? "IQ4_NL public" : "IQ3_S public", shape, r0, download(y2, N));
    cudaFree(dw); cudaFree(dx); cudaFree(y0); cudaFree(y1); cudaFree(y2);
}

// The Q2_0 resident down projection: missing, out-of-range, non-resident and repeated experts, uniform slots and
// a per-slot offset table, row counts that leave a partial four-row warp.
void resident_down_case(long n_embd, long n_ff, int cap, bool offsets_table, int mode) {
    const int n_expert = 12, slots = 10;
    const unsigned long d_row = (unsigned long) (n_ff / 64) * 18, down_off = 4096;
    const unsigned long slot_bytes = down_off + (unsigned long) n_embd * d_row + 64;
    std::vector<uint8_t> arena(slot_bytes * slots);
    for (auto& v : arena) v = mode == 1 ? 0xFF : (uint8_t) rng();   // 0xFF: every code 3, the value 2
    for (int s = 0; s < slots; ++s)
        for (long r = 0; r < n_embd; ++r)
            for (long b = 0; b < n_ff / 64; ++b) {
                const uint16_t d = half_scale(mode);
                std::memcpy(arena.data() + s * slot_bytes + down_off + r * d_row + b * 18, &d, 2);
            }
    std::vector<int> residency(n_expert), ids(cap);
    for (int i = 0; i < n_expert; ++i) residency[i] = i < slots ? (i * 7) % slots : -1;
    for (int e = 0; e < cap; ++e) ids[e] = e % 9 == 4 ? -1 : e % 11 == 7 ? n_expert + 3 : (int) (rng() % n_expert);
    std::vector<uint64_t> offs(slots);
    for (int s = 0; s < slots; ++s) offs[s] = (uint64_t) (slots - 1 - s) * slot_bytes;
    void* da = upload(arena);
    void* di = upload(ids);
    void* dr = upload(residency);
    void* doff = upload(offs);
    void* dh = upload(q8((int) n_ff * cap, mode));
    float* y0 = out_buffer((size_t) cap * n_embd, 0xAA);
    float* y1 = out_buffer((size_t) cap * n_embd, 0x55);
    const bool canonical = n_embd == 2560 && n_ff == 640;
    float* y2 = canonical ? out_buffer((size_t) cap * n_embd, 0x33) : nullptr;
    const int k = 10, has = offsets_table ? 1 : 0;
    for (int v = 0; v < (canonical ? 3 : 2); ++v) {
        const bool direct = v != 0;
        const char* name = v == 0 ? "native_resident_down_42" : v == 1 ? "native_resident_down_direct_42_r4"
                                                                 : "native_resident_down_dim_42_r4";
        Launch d(name,
                 (unsigned) (direct ? (n_embd + 31) / 32 : (n_embd + 7) / 8), (unsigned) cap, 1, 256, 1, 1, 0, stream);
        d.buf(da).buf(offsets_table ? doff : nullptr).buf(di).buf(dr).buf(dh).scalar(n_embd).scalar(n_ff).scalar(d_row)
         .scalar(down_off).scalar(slot_bytes).scalar(n_expert).scalar(k).scalar(has).buf(v == 0 ? y0 : v == 1 ? y1 : y2);
        d.done();
    }
    compare("Q2_0 resident down", std::to_string(n_embd) + "x" + std::to_string(n_ff) + " cap " + std::to_string(cap) +
            (offsets_table ? " offsets" : " uniform") + " mode " + std::to_string(mode),
            download(y0, (size_t) cap * n_embd), download(y1, (size_t) cap * n_embd));
    if (canonical) {
        compare("Q2_0 resident down dimensions", std::to_string(cap) + (offsets_table ? " offsets" : " uniform") +
                " mode " + std::to_string(mode), download(y0, (size_t) cap * n_embd),
                download(y2, (size_t) cap * n_embd));
        cudaFree(y2);
    }
    cudaFree(da); cudaFree(di); cudaFree(dr); cudaFree(doff); cudaFree(dh); cudaFree(y0); cudaFree(y1);
}

// native_expert_resident (gate/up, SwiGLU, Q8_1, down) against the same chain with the original down kernel.
void resident_public_case(int n_tok, long n_embd = 2560, long n_ff = 640) {
    const int n_expert = 6, k = 4, cap = n_tok * k;
    const auto L = strata::kernels::native_expert_layout(42, 42, n_embd, n_ff);
    std::vector<uint8_t> arena(L.bytes * n_expert);
    fill_blocks(arena, 18, 0, 0);   // every blob section is a whole number of 18-byte Q2_0 blocks
    std::vector<int> residency = {3, -1, 0, 4, 1, 2}, ids(cap);
    for (int e = 0; e < cap; ++e) ids[e] = (int) (rng() % n_expert);
    void* da = upload(arena);
    void* di = upload(ids);
    void* dr = upload(residency);
    void* dx = upload(q8((int) n_embd * n_tok, 0));
    void* scratch = nullptr;
    cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(cap, n_ff));
    float* y0 = out_buffer((size_t) cap * n_embd, 0xAA);
    float* y1 = out_buffer((size_t) cap * n_embd, 0x55);
    strata::kernels::native_expert_resident(L, (const uint8_t*) da, nullptr, (int64_t) L.bytes, (const int32_t*) di,
                                            (const int32_t*) dr, n_expert, n_tok, k, dx, scratch, y1, stream);
    const size_t f = (size_t) cap * n_ff * 4, fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    void* hq = (uint8_t*) scratch + 3 * fa;
    const int has = 0;
    const unsigned long gu_row = L.gu_row, up_off = L.up_off, d_row = L.d_row, down_off = L.down_off, sb = L.bytes;
    Launch gu("native_resident_gu_42", (unsigned) ((2 * n_ff + 7) / 8), (unsigned) cap, 1, 256, 1, 1, 0, stream);
    gu.buf(da).buf(nullptr).buf(di).buf(dr).buf(dx).scalar(n_embd).scalar(n_ff).scalar(gu_row).scalar(up_off)
      .scalar(sb).scalar(n_expert).scalar(k).scalar(has).buf(gate).buf(up);
    gu.done();
    const long nh = (long) cap * n_ff;
    Launch sw("swiglu_entries_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    sw.buf(gate).buf(up).buf(h).scalar(nh); sw.done();
    Launch qz("quantize_q8_1_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    qz.buf(h).buf(hq).scalar(nh); qz.done();
    Launch dn("native_resident_down_42", (unsigned) ((n_embd + 7) / 8), (unsigned) cap, 1, 256, 1, 1, 0, stream);
    dn.buf(da).buf(nullptr).buf(di).buf(dr).buf(hq).scalar(n_embd).scalar(n_ff).scalar(d_row).scalar(down_off)
      .scalar(sb).scalar(n_expert).scalar(k).scalar(has).buf(y0);
    dn.done();
    compare("Q2_0 resident public", "tokens " + std::to_string(n_tok), download(y0, (size_t) cap * n_embd),
            download(y1, (size_t) cap * n_embd));
    cudaFree(da); cudaFree(di); cudaFree(dr); cudaFree(dx); cudaFree(scratch); cudaFree(y0); cudaFree(y1);
}

std::vector<float> normals(size_t n, float scale) {
    std::normal_distribution<float> nd;
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng) * scale;
    return v;
}
std::vector<uint16_t> bf16s(size_t n, float scale) {
    std::vector<uint16_t> v(n);
    const auto f = normals(n, scale);
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &f[i], 4);
        v[i] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return v;
}

// The hyper-connection read: the one-pass norm against the two-pass kernel (special values included), and the
// public fused_gr_read against norm -> down -> up with the original kernels.
void gr_case(int apply, bool specials) {
    constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
    auto R = normals(D, 1.0f);
    if (specials) { R[5] = INFINITY; R[N + 9] = NAN; R[2 * N] = -0.0f; R[3 * N + 1] = 1e-42f; }
    void* dR = upload(R);
    void* dbo = upload(normals(N, 0.5f));
    void* dinj = upload(normals(HC, 1.0f));
    void* dwn = upload(normals(D, 1.0f));
    const float eps = 1e-6f;
    float* rs0 = out_buffer(HC, 0xAA), * rs1 = out_buffer(HC, 0x55);
    float* xn0 = out_buffer(D, 0xAA), * xn1 = out_buffer(D, 0x55);
    for (int v = 0; v < 2; ++v) {
        Launch n(v ? "fused_gr_norm1_kernel" : "fused_gr_norm_kernel", 1, 1, 1, 256, 1, 1, 0, stream);
        n.buf(dR).buf(dbo).buf(dinj).buf(dwn).scalar(eps).scalar(apply).buf(v ? rs1 : rs0).buf(v ? xn1 : xn0);
        n.done();
    }
    const std::string shape = std::string("apply ") + std::to_string(apply) + (specials ? " specials" : "");
    compare("GR norm1 rs", shape, download(rs0, HC), download(rs1, HC));
    compare("GR norm1 xn", shape, download(xn0, D), download(xn1, D));
    if (!specials) {
        void* dwd = upload(bf16s((size_t) LR * D, 0.02f));
        void* dwu = upload(bf16s((size_t) D * LR, 0.05f));
        void* dwi = upload(bf16s((size_t) HC * D, 0.02f));
        float* lo0 = out_buffer(LR, 0xAA), * io0 = out_buffer(HC, 0xAA), * mx0 = out_buffer(N, 0xAA), * Ro0 = out_buffer(D, 0xAA);
        float* lo1 = out_buffer(LR, 0x55), * io1 = out_buffer(HC, 0x55), * mx1 = out_buffer(N, 0x55), * Ro1 = out_buffer(D, 0x55);
        Launch d("fused_gr_down_kernel", 41, 1, 1, 256, 1, 1, 0, stream);
        d.buf(dwd).buf(dwi).buf(xn0).buf(lo0).buf(io0); d.done();
        Launch u("fused_gr_up_kernel", N / 16, 1, 1, 256, 1, 1, 0, stream);
        u.buf(dwu).buf(lo0).buf(dR).buf(Ro0).buf(dbo).buf(dinj).buf(dwn).buf(rs0).buf(mx0).scalar(apply); u.done();
        strata::kernels::FusedGrArgs a;
        a.R = (const float*) dR; a.R_out = Ro1; a.apply = apply != 0; a.bo_prev = (const float*) dbo;
        a.inj_prev = (const float*) dinj; a.w_norm = (const float*) dwn; a.w_down = (const uint16_t*) dwd;
        a.w_up = (const uint16_t*) dwu; a.w_inject = (const uint16_t*) dwi; a.eps = eps; a.lo = lo1; a.rs = rs1;
        a.inject_out = io1; a.mixed = mx1;
        strata::kernels::fused_gr_read(a, stream);
        compare("GR public lo", shape, download(lo0, LR), download(lo1, LR));
        compare("GR public inject", shape, download(io0, HC), download(io1, HC));
        compare("GR public mixed", shape, download(mx0, N), download(mx1, N));
        if (apply) compare("GR public R_out", shape, download(Ro0, D), download(Ro1, D));
        for (void* p : {dwd, dwu, dwi, (void*) lo0, (void*) io0, (void*) mx0, (void*) Ro0, (void*) lo1, (void*) io1,
                        (void*) mx1, (void*) Ro1})
            cudaFree(p);
    }
    for (void* p : {dR, dbo, dinj, dwn, (void*) rs0, (void*) rs1, (void*) xn0, (void*) xn1}) cudaFree(p);
}

// The GDN recurrence: the kernel that stages every token's state against the one that writes the last token's
// state straight to `state` (commit) or not at all (verify), and the public entry point.
void gdn_case(int T, int keep, int t_out_begin) {
    constexpr int S = 128;
    const int h_k = 16, h_v = 32, C = 2 * S * h_k + S * h_v, VD = S * h_v;
    auto state = normals((size_t) S * S * h_v, 0.05f);
    auto h = normals((size_t) T * C, 0.1f);
    std::vector<float> gate((size_t) T * h_v), beta((size_t) T * h_v);
    for (auto& g : gate) g = -std::fabs(normals(1, 1.0f)[0]);
    for (auto& b : beta) b = 1.0f / (1.0f + std::exp(-normals(1, 1.0f)[0]));
    void* dh = upload(h);
    void* dg = upload(gate);
    void* db = upload(beta);
    void* dz = upload(normals((size_t) T * VD, 1.0f));
    void* dgm = upload(normals(S, 1.0f));
    std::vector<int32_t> nk = {keep};
    void* dk = upload(nk);
    void* st[3];
    float* y[3];
    for (int v = 0; v < 3; ++v) { st[v] = upload(state); y[v] = out_buffer((size_t) T * VD, 0x5A); }
    float* shadow = out_buffer((size_t) S * S * h_v, 0x00);
    const float eps = 1e-6f;
    const bool commit = keep >= 0;
    for (int v = 0; v < 2; ++v) {
        Launch k(v ? "gdn_step_norm_multi_tail_kernel" : "gdn_step_norm_multi_kernel", (unsigned) h_v, 1, 1, S, 4, 1, 0,
                 stream);
        k.buf(st[v]).buf(dh).scalar(C).buf(dg).buf(db).buf(dz).buf(dgm).scalar(eps).buf(y[v]).scalar(h_k).scalar(h_v)
         .scalar(T).buf(commit ? dk : nullptr).scalar(t_out_begin).buf(shadow);
        k.done();
    }
    strata::kernels::gdn_step_norm_multi((float*) st[2], (const float*) dh, C, (const float*) dg, (const float*) db,
                                         (const float*) dz, (const float*) dgm, eps, y[2], h_k, h_v, T,
                                         commit ? (const int32_t*) dk : nullptr, stream, t_out_begin);
    const std::string shape = "T " + std::to_string(T) + (commit ? " keep " + std::to_string(keep) : " verify") +
                              " from " + std::to_string(t_out_begin);
    const auto s0 = download(st[0], (size_t) S * S * h_v), y0 = download(y[0], (size_t) T * VD);
    compare("GDN tail state", shape, s0, download(st[1], (size_t) S * S * h_v));
    compare("GDN tail y", shape, y0, download(y[1], (size_t) T * VD));
    compare("GDN public state", shape, s0, download(st[2], (size_t) S * S * h_v));
    compare("GDN public y", shape, y0, download(y[2], (size_t) T * VD));
    for (void* p : {dh, dg, db, dz, dgm, dk, st[0], st[1], st[2], (void*) y[0], (void*) y[1], (void*) y[2], (void*) shadow})
        cudaFree(p);
}

// QSA block top-k: the parallel-scan kernel (the public entry) against block_topk_kernel (the ref entry), several
// queries per launch at different contexts, realistic / tied / all-equal / NaN and -0 scores.
void topk_case(int dist) {
    const auto s = strata::kernels::qsa_real_shapes();
    const long cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const long max_blocks = 32768 / 4 + 2;
    // the equal-key cut lands in different threads' ranges as the context grows: tie-heavy inputs sweep it
    // through every thread (all-equal 2052..2560 moves it across all 256 ranges, one block per thread at most)
    std::vector<long> ctx = {5, 2051, 2052, 2100, 3001, 4100, 5120, 5121, 6150, 7003, 8200, 10000, 10003, 16384, 32767};
    if (dist == 2 || dist == 4)
        for (long c = 2052; c <= 32767; c += c < 4200 ? 3 : 211) ctx.push_back(c);
    const long nq = (long) ctx.size();
    std::vector<float> sc((size_t) nq * max_blocks, 0.0f);
    std::vector<int32_t> steps((size_t) nq * 4);
    std::normal_distribution<float> nd;
    for (long q = 0; q < nq; ++q) {
        const long n_kv = ctx[q], n_bid = n_kv / 4;
        strata::kernels::qsa_step_fill(steps.data() + q * 4, n_kv - 1, s);
        for (long b = 0; b <= n_bid; ++b) {
            float v = 0.0f;
            if (dist == 0) for (int h = 0; h < 4; ++h) v += std::max(0.0f, nd(rng) * 3.0f);
            if (dist == 1) v = (float) (rng() % 7);
            if (dist == 3) { v = nd(rng); if (rng() % 9 == 0) v = NAN; if (rng() % 11 == 0) v = -0.0f; }
            if (dist == 4) v = b % 3 == 0 ? 1.0f : 0.0f;   // ties at the threshold in every thread
            sc[(size_t) q * max_blocks + b] = v;
        }
        if (n_kv % 4) sc[(size_t) q * max_blocks + n_bid] += 1e9f;
    }
    void* ds = upload(sc);
    void* dst = upload(steps);
    float* a = out_buffer((size_t) nq * cap, 0xAA);
    float* b = out_buffer((size_t) nq * cap, 0xAA);
    strata::kernels::qsa_block_topk_ref((const float*) ds, (const int32_t*) dst, nq, max_blocks, cap, s, (int32_t*) a, stream);
    strata::kernels::qsa_block_topk((const float*) ds, (const int32_t*) dst, nq, max_blocks, cap, s, (int32_t*) b, stream, 0);
    compare("QSA block top-k", "distribution " + std::to_string(dist), download(a, (size_t) nq * cap),
            download(b, (size_t) nq * cap));
    cudaFree(ds); cudaFree(dst); cudaFree(a); cudaFree(b);
}

// The prefill GEMMs: pfl_gemm2_f16 / pfl_gemm2_bf16 (64 x 64 tiles) against pfl_gemm_f16 / pfl_gemm_bf16 by name:
// partial M and N tiles, a strided output (ldy > N), beta 0 and 1, FP16 and BF16 operands with inf / NaN entries.
void gemm2_case(bool bf, unsigned M, unsigned N, unsigned K, unsigned ldy, float beta, bool specials) {
    std::normal_distribution<float> nd;
    auto h16 = [&](float f) {
        uint32_t u; std::memcpy(&u, &f, 4);
        if (bf) return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        __fp16 hv = (__fp16) f; uint16_t r; std::memcpy(&r, &hv, 2); return r;
    };
    std::vector<uint16_t> x((size_t) M * K), w((size_t) N * K);
    for (auto& v : x) v = h16(nd(rng) * 0.5f);
    for (auto& v : w) v = h16(nd(rng) * 0.05f);
    if (specials) { x[3] = h16(INFINITY); x[K + 5] = h16(NAN); w[7] = h16(-INFINITY); w[2 * K] = h16(-0.0f); }
    std::vector<float> y0((size_t) M * ldy);
    for (auto& v : y0) v = nd(rng);
    void* dx = upload(x);
    void* dw = upload(w);
    void* dy0 = upload(y0);
    void* dy1 = upload(y0);
    for (int v = 0; v < 2; ++v) {
        const std::string kn = std::string(v ? "pfl_gemm2_" : "pfl_gemm_") + (bf ? "bf16" : "f16");
        const unsigned gx = v ? (N + 63) / 64 : (N + 31) / 32, gy = v ? (M + 63) / 64 : (M + 15) / 16;
        Launch k(kn.c_str(), gx, gy, 1, v ? (bf ? 256 : 128) : 128, 1, 1, 0, stream);
        k.buf(dx).buf(dw).buf(v ? dy1 : dy0).scalar(M).scalar(N).scalar(K).scalar(ldy).scalar(beta);
        k.done();
    }
    compare(bf ? "prefill GEMM2 bf16" : "prefill GEMM2 f16",
            std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + " ldy " + std::to_string(ldy) +
                " beta " + std::to_string((int) beta) + (specials ? " specials" : ""),
            download(dy0, (size_t) M * ldy), download(dy1, (size_t) M * ldy));
    cudaFree(dx); cudaFree(dw); cudaFree(dy0); cudaFree(dy1);
}

// Prompt attention: prompt_attn_reg_m{0,1,3} against prompt_attn_kernel_m{0,1,3} by name - FP16, INT8 and K8V4
// pools, a masked (non-resident) page, a non-finite K entry, widths below and at the selection cap.
void prompt_attn_case(int mode, int pos0, bool specials) {
    constexpr int HD = 256, G = 12, KVH = 2, NH = KVH * G, PAGE = 64;
    const int nq = 48, cells = pos0 + nq + 64, pages = (cells + PAGE - 1) / PAGE;
    const long cap = 2051;
    std::normal_distribution<float> nd;
    const size_t rows = (size_t) pages * KVH * PAGE;
    std::vector<uint16_t> kp(rows * HD), vp(rows * HD), ks(rows * 4), vs(rows * 4);
    std::vector<int8_t> kq(rows * HD), vq(rows * HD);
    std::vector<uint8_t> vq4(rows * 8 * 18);
    auto h = [](float f) { __fp16 v = (__fp16) f; uint16_t u; std::memcpy(&u, &v, 2); return u; };
    for (size_t i = 0; i < kp.size(); ++i) { kp[i] = h(nd(rng) * 0.5f); vp[i] = h(nd(rng)); kq[i] = (int8_t) rng(); vq[i] = (int8_t) rng(); }
    for (size_t i = 0; i < ks.size(); ++i) { ks[i] = h(0.004f * (1 + rng() % 4)); vs[i] = h(0.01f * (1 + rng() % 4)); }
    for (size_t i = 0; i < vq4.size(); ++i) vq4[i] = (uint8_t) rng();
    for (size_t r = 0; r < rows; ++r) for (int b = 0; b < 8; ++b) { const uint16_t d = h(0.05f); std::memcpy(&vq4[(r * 8 + b) * 18], &d, 2); }
    if (specials) { kp[5 * HD + 3] = h(INFINITY); kq[7 * HD + 1] = -128; }
    std::vector<int> table(pages);
    for (int pg = 0; pg < pages; ++pg) table[pg] = pg;
    if (specials) table[2] = -1;
    std::vector<float> qv((size_t) nq * NH * HD);
    for (auto& v : qv) v = nd(rng);
    std::vector<int> ids((size_t) nq * cap, 0), steps((size_t) nq * 4);
    for (int i = 0; i < nq; ++i) {
        const int pos = pos0 + i, n_kv = pos + 1, width = (int) std::min<long>(n_kv, cap);
        steps[i * 4 + 0] = pos; steps[i * 4 + 1] = n_kv; steps[i * 4 + 2] = n_kv / 4; steps[i * 4 + 3] = width;
        std::vector<int> all(n_kv);
        for (int c = 0; c < n_kv; ++c) all[c] = c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + width);
        std::copy(all.begin(), all.begin() + width, ids.begin() + (size_t) i * cap);
    }
    void *dq = upload(qv), *dkp = upload(kp), *dvp = upload(vp), *dkq = upload(kq), *dvq = upload(vq), *dks = upload(ks),
         *dvs = upload(vs), *dvq4 = upload(vq4), *dt = upload(table), *di = upload(ids), *dst = upload(steps);
    float* o[2] = {out_buffer(qv.size(), 0xAA), out_buffer(qv.size(), 0x55)};
    const int nkvh = KVH, ps = PAGE;
    for (int v = 0; v < 2; ++v) {
        const std::string kn = std::string(v ? "prompt_attn_reg_m" : "prompt_attn_kernel_m") + std::to_string(mode);
        Launch k(kn.c_str(), (unsigned) nq, (unsigned) KVH, 1, 256, 1, 1, 0, stream);
        k.buf(dq).buf(mode == 0 ? dkp : nullptr).buf(mode == 0 ? dvp : nullptr).buf(mode ? dkq : nullptr)
         .buf(mode == 1 ? dvq : nullptr).buf(mode ? dks : nullptr).buf(mode == 1 ? dvs : nullptr).buf(nullptr)
         .buf(mode == 3 ? dvq4 : nullptr).buf(dt).buf(di).buf(dst).scalar(nkvh).scalar(ps).scalar(cap).buf(o[v]);
        k.done();
    }
    compare("prompt attention reg", "mode " + std::to_string(mode) + " pos " + std::to_string(pos0) + (specials ? " specials" : ""),
            download(o[0], qv.size()), download(o[1], qv.size()));
    for (void* p : {dq, dkp, dvp, dkq, dvq, dks, dvs, dvq4, dt, di, dst, (void*) o[0], (void*) o[1]}) cudaFree(p);
}

// Prefill expert GEMM: native_gemm2_* (16 / 32-row tiles of 64 columns) against native_gemm_* (16 x 32) through the
// public native_expert_gemm, every supported gate/up and down format, several experts with odd row counts.
void moe_gemm_case(int gu_t, int d_t) {
    const long H = 2560, FF = 640;
    const auto L = strata::kernels::native_expert_layout(gu_t, d_t, H, FF);
    if (!strata::kernels::native_expert_gemm_supported(L)) return;   // e.g. IQ4_XS down needs n_ff % 256 == 0
    const int n_exp = 5;
    std::vector<uint8_t> arena(L.bytes * n_exp);
    for (auto& v : arena) v = (uint8_t) rng();
    auto scales = [&](size_t at, int type, size_t bytes) {
        const int qk = type == 42 ? 64 : type == 20 ? 32 : 256;
        const size_t block = strata::kernels::iq_row_bytes(type, qk);
        for (size_t i = 0; i < bytes; i += block) {
            if (type == 29) { arena[at + i + 55] = (arena[at + i + 55] & 15) | 0x10; continue; }
            const __fp16 d = (__fp16) ((int(rng() % 63) - 31) / 8192.0f);
            std::memcpy(&arena[at + i], &d, 2);
        }
    };
    for (int e = 0; e < n_exp; ++e) {
        scales(e * L.bytes, gu_t, 2 * L.up_off);
        scales(e * L.bytes + L.down_off, d_t, L.bytes - L.down_off);
    }
    const int cnt[n_exp] = {1, 17, 33, 64, 70};
    int T = 0;
    for (int c : cnt) T += c;
    void* da = upload(arena);
    for (bool down : {false, true}) {
        const long Kd = down ? FF : H, Nd = down ? H : 2 * FF;
        std::vector<uint16_t> x((size_t) T * Kd);
        std::normal_distribution<float> nd;
        for (auto& v : x) { __fp16 hv = (__fp16) nd(rng); std::memcpy(&v, &hv, 2); }
        void* dx = upload(x);
        float* y[3] = {out_buffer((size_t) T * Nd, 0xAA), out_buffer((size_t) T * Nd, 0x55), out_buffer((size_t) T * Nd, 0x33)};
        for (int v = 0; v < 3; ++v) {
            const int step = v == 0 ? 16 : v == 1 ? 16 : 32;
            std::vector<int32_t> tiles;
            int off = 0;
            for (int e = 0; e < n_exp; ++e) {
                const uint64_t o = (uint64_t) e * L.bytes;
                for (int r = 0; r < cnt[e]; r += step) {
                    tiles.push_back((int32_t) (uint32_t) o); tiles.push_back(off + r);
                    tiles.push_back(std::min(step, cnt[e] - r)); tiles.push_back((int32_t) (uint32_t) (o >> 32));
                }
                off += cnt[e];
            }
            void* dt = upload(tiles);
            strata::kernels::native_expert_gemm(L, (const uint16_t*) dx, (const uint8_t*) da, (const int32_t*) dt, y[v],
                                                (int64_t) tiles.size() / 4, down, stream, v == 0 ? 0 : step);
            cudaStreamSynchronize(stream);
            cudaFree(dt);
        }
        const std::string shape = std::to_string(gu_t) + "/" + std::to_string(d_t) + (down ? " down" : " gate/up");
        const auto r0 = download(y[0], (size_t) T * Nd);
        compare("prefill MoE 16x64", shape, r0, download(y[1], (size_t) T * Nd));
        compare("prefill MoE 32x64", shape, r0, download(y[2], (size_t) T * Nd));
        cudaFree(dx); cudaFree(y[0]); cudaFree(y[1]); cudaFree(y[2]);
    }
    cudaFree(da);
}

}  // namespace

int main() {
    cudaStreamCreate(&stream);
    const std::vector<std::pair<int, int>> iq4 = {{2560, 2563}, {2560, 640}, {6144, 2561}, {10240, 1027}, {512, 7},
                                                  {4096, 2560}, {2560, 10243}, {2560, 4093}};
    const std::vector<std::pair<int, int>> nl = {{2560, 640}, {2560, 643}, {640, 2561}, {64, 5}, {4096, 999}};
    const std::vector<std::pair<int, int>> iq3 = {{2560, 6144}, {2560, 643}, {4096, 2561}, {256, 9}};
    for (int mode = 0; mode < 3; ++mode) {
        for (auto [K, N] : iq4) mmvq_case(23, K, N, mode);
        for (auto [K, N] : nl) mmvq_case(20, K, N, mode);
        for (auto [K, N] : iq3) mmvq_case(21, K, N, mode);
        resident_down_case(2560, 640, 10, false, mode);
        resident_down_case(2560, 640, 23, true, mode);
        resident_down_case(2563 - 3 + 36, 640, 23, true, mode);   // 2596 rows: a partial four-row warp
        resident_down_case(512, 128, 7, true, mode);
    }
    for (int t : {1, 3}) { resident_public_case(t); resident_public_case(t, 512, 128); }
    for (int apply : {0, 1}) { gr_case(apply, false); gr_case(apply, true); }
    for (int T : {1, 2, 4, 8}) {
        gdn_case(T, -1, 0);                      // verify: state read, not written
        gdn_case(T, T, 0);                       // commit every token
        if (T > 1) { gdn_case(T, T / 2, 0); gdn_case(T, -1, 1); }
    }
    gdn_case(3, 0, 0);                           // commit nothing
    for (int dist = 0; dist < 5; ++dist) topk_case(dist);
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42})
        for (int dt : {20, 23, 42}) moe_gemm_case(gu, dt);
    for (int mode : {0, 1, 3})
        for (int pos0 : {40, 2100, 9000}) { prompt_attn_case(mode, pos0, false); prompt_attn_case(mode, pos0, true); }
    for (bool bf : {false, true}) {
        gemm2_case(bf, 1024, 640, 2560, 640, 0.0f, false);
        gemm2_case(bf, 777, 330, 1024, 333, 1.0f, false);       // partial tiles, strided output, beta 1
        gemm2_case(bf, 61, 70, 320, 70, 0.0f, true);            // smaller than one tile, inf / NaN / -0
        gemm2_case(bf, 129, 2560, 6144, 2560, 1.0f, true);
        gemm2_case(bf, 1, 32, 2560, 32, 0.0f, false);
    }
    cudaStreamDestroy(stream);
    std::printf("direct decode kernels: %lld outputs compared, %lld bit differences\n", g_checked, g_diff);
    return g_diff != 0;
}

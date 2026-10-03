// src/prefill/kernels.mm - the port of src/prefill/kernels.cu's launchers (M4).  Every k<<<grid, block,
// 0, stream>>>(args...) becomes strata::metal::Launch with the SAME argument order - the kernel's buffer
// index equals its position in the chain (bug-class A) - and the geometry is the .cu's own.  The kernels
// live in src/kernels/metal/prefill.metal (pfl_ prefixed: this file's CUDA names collide with the decode
// path's ports) and join the shared metallib via the src/kernels/metal/*.metal glob.
//
// The two RULE 9 re-spellings the .cu needed (a pointer in argument bytes does not dereference on this
// GPU - round 9's measurements): rope_kernel's by-value RopeTab arrives as (use_tab, cos, sin, max_pos)
// with the table buffers bound (nil when absent), and kv_append_kernel's two by-value KvHostPools arrive
// as sixteen bound buffers in the struct's declaration order, nulls binding nil - the kv_q8 port's idiom.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int RG = 4, CB = 32, NCB = S / CB;
constexpr int CONV_TILE = 64;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

/// RULE 9: the KvHostPools' eight fields as eight bound buffers, declaration order, nulls nil.
metal::Launch& host_pools(metal::Launch& k, const strata::kernels::KvHostPools& p) {
    return k.buf(p.k_pool).buf(p.v_pool).buf(p.k_q).buf(p.v_q).buf(p.k_scale).buf(p.v_scale).buf(p.k_q4)
     .buf(p.v_q4);
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    const strata::kernels::KvHostPools h = host ? *host : strata::kernels::KvHostPools{};
    const strata::kernels::KvHostPools st = stage ? *stage : strata::kernels::KvHostPools{};
    metal::Launch k("pfl_kv_append_kernel", (unsigned) T, 2, 8, 64, 1, 1, 0, stream);
    k.buf(K).buf(V).scalar(pos0).buf(page_table).scalar(page_size);
    k.buf(k_pool).buf(v_pool).buf(k_q).buf(v_q).buf(k_scale).buf(v_scale);
    host_pools(k, h);
    host_pools(k, st);
    k.done();
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("pfl_to_f16_kernel", (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1, 256, 1,
                    1, 0, stream);
    k.buf(x).buf(y).scalar((unsigned long) n);
    k.done();
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("pfl_round_f16_kernel", (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1, 256,
                    1, 1, 0, stream);
    k.buf(x).buf(y).scalar((unsigned long) n);
    k.done();
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    metal::Launch k("pfl_to_bf16_kernel", (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1, 256, 1,
                    1, 0, stream);
    k.buf(x).buf(y).buf(ylo).scalar((unsigned long) n);
    k.done();
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    metal::Launch k("pfl_gr_norm_kernel", (unsigned) (T * HC), 1, 1, 256, 1, 1, 0, stream);
    k.buf(R).buf(w_norm).scalar(eps).buf(xn).buf(xn16).buf(xn16_lo);
    k.done();
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    metal::Launch k("pfl_gr_norm_rs_kernel", (unsigned) (T * HC), 1, 1, 256, 1, 1, 0, stream);
    k.buf(R).buf(w_norm).scalar(eps).buf(rs).buf(xn16).buf(xn16_lo);
    k.done();
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    metal::Launch k("pfl_gr_mix_r_kernel", blocks_for(T * N), 1, 1, 256, 1, 1, 0, stream);
    k.buf(R).buf(rs).buf(w_norm).buf(gated).buf(mixed).buf(mixed16).scalar(T).buf(mixed_h).buf(mixed16_lo);
    k.done();
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    metal::Launch k("pfl_gr_write_norm_rs_kernel", (unsigned) (T * HC), 1, 1, 256, 1, 1, 0, stream);
    k.buf(R).buf(bo).buf(inj).scalar(inj_ld).buf(w_norm_next).scalar(eps).buf(rs).buf(xn16).buf(xn16_lo);
    k.done();
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    metal::Launch k("pfl_gr_silu_kernel", blocks_for((int64_t) T * LR), 1, 1, 256, 1, 1, 0, stream);
    k.buf(lo).buf(lo16).buf(lo16_lo).scalar((unsigned long) (T * LR));
    k.done();
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    metal::Launch k("pfl_gr_mix_kernel", blocks_for(T * N), 1, 1, 256, 1, 1, 0, stream);
    k.buf(xn).buf(gated).buf(mixed).buf(mixed16).scalar(T).buf(mixed_h).buf(mixed16_lo);
    k.done();
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    metal::Launch k("pfl_gr_write_kernel", blocks_for((int64_t) T * D), 1, 1, 256, 1, 1, 0, stream);
    k.buf(R).buf(bo).buf(inj).scalar(inj_ld).scalar(T);
    k.done();
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    metal::Launch k("pfl_gr_broadcast_kernel", blocks_for((int64_t) T * D), 1, 1, 256, 1, 1, 0, stream);
    k.buf(e).buf(R).scalar(T);
    k.done();
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    metal::Launch k("pfl_gdn_gates_kernel", blocks_for((int64_t) T * HV), 1, 1, 256, 1, 1, 0, stream);
    k.buf(ab).buf(dt).buf(ssm_a).buf(gate).buf(beta).scalar(T);
    k.done();
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        metal::Launch k("pfl_gdn_conv_kernel", C / 128, 1, 1, 128, 1, 1, 0, stream);
        k.buf(history).buf(qkv).buf(conv_w).buf(h).scalar(T);
        k.done();
    } else {
        metal::Launch k("pfl_gdn_conv_tiled_kernel", C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE), 1, 128, 1,
                        1, 0, stream);
        k.buf(history).buf(qkv).buf(conv_w).buf(h).scalar(T);
        k.done();
        metal::Launch g("pfl_gdn_conv_hist_kernel", C / 128, 1, 1, 128, 1, 1, 0, stream);
        g.buf(history).buf(qkv).scalar(T);
        g.done();
    }
    metal::Launch l("pfl_gdn_l2_kernel", 2 * HK, (unsigned) T, 1, S, 1, 1, 0, stream);
    l.buf(h).scalar(eps);
    l.done();
    check("gdn_conv");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
    if (serial || T <= 0) {
        metal::Launch k("pfl_gdn_rec_kernel", HV, 1, 1, S, RG, 1, 0, stream);
        k.buf(state).buf(h).buf(gate).buf(beta).buf(z).buf(gamma).scalar(eps).buf(y).buf(y16).scalar(T);
        k.done();
    } else {
        static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
        metal::Launch k(pipe ? "pfl_gdn_rec_cols_pipe_kernel" : "pfl_gdn_rec_cols_kernel", HV * NCB, 1, 1, CB, RG, 1,
                        0, stream);
        k.buf(state).buf(h).buf(gate).buf(beta).buf(y).scalar(T);   // the same bits, the .cu's own A/B
        k.done();
        metal::Launch n("pfl_gdn_out_norm_kernel", (unsigned) T, HV, 1, S, 1, 1, 0, stream);
        n.buf(z).buf(gamma).scalar(eps).buf(y).buf(y16);
        n.done();
    }
    check("gdn_recurrence");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512 || n_expert == 256) {
        metal::Launch k("pfl_route_kernel", (unsigned) ((T + 7) / 8), 1, 1, 256, 1, 1, 0, stream);
        k.buf(logits).buf(ids).buf(weights).scalar(T).scalar((int) (n_expert / 32));
        k.done();
    } else {
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    }
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    metal::Launch k("pfl_blob_dequant_kernel", blocks_for((int64_t) 1280 * 640 + 2560 * 160), 1, 1, 256, 1, 1, 0,
                    stream);
    k.buf(blob).buf(gu16).buf(down16).scalar(0);
    k.done();
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    metal::Launch k("pfl_blob_dequant_kernel", blocks_for((int64_t) 1280 * 640 + 2560 * 160), 1, 1, 256, 1, 1, 0,
                    stream);
    k.buf(blob).buf(gu16).buf(down16).scalar(1);
    k.done();
    check("blob_dequant_f16");
}
void moe_q2_gemm(const uint16_t* x, const uint8_t* arena, const int32_t* tiles, float* y,
                 int64_t n_tiles, int64_t blob_bytes, bool down, void* stream) {
    if (n_tiles <= 0) return;
    metal::Launch k(down ? "pfl_q2_gemm_down" : "pfl_q2_gemm_gu", down ? 80 : 40,
                    (unsigned)n_tiles, 1, 128, 1, 1, 0, stream);
    k.buf(x).buf(arena).buf(tiles).buf(y).scalar((unsigned long)blob_bytes); k.done();
    check("moe_q2_gemm");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("pfl_swiglu_il_kernel", blocks_for(n * 640), 1, 1, 256, 1, 1, 0, stream);
    k.buf(gu).buf(h16).scalar(n);
    k.done();
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    metal::Launch k("pfl_swiglu_pair_kernel", blocks_for(n * 640), 1, 1, 256, 1, 1, 0, stream);
    k.buf(g).buf(u).buf(h16).scalar(n);
    k.done();
    check("swiglu_pair");
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    metal::Launch k("pfl_copy_i32_kernel", (unsigned) (b < 256 ? b : 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar((unsigned long) n);
    k.done();
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    metal::Launch k("pfl_gather_rows16_kernel", blocks_for(n * (width / 8)), 1, 1, 256, 1, 1, 0, stream);
    k.buf(x16).buf(src).buf(dst16).scalar(n).scalar(width);
    k.done();
    check("gather_rows16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    metal::Launch k("pfl_moe_combine_kernel", blocks_for((int64_t) T * N), 1, 1, 256, 1, 1, 0, stream);
    k.buf(Dm).buf(slot).buf(w).buf(shared).buf(sg).buf(bo).scalar(T);
    k.done();
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    metal::Launch k("pfl_rms_rows_kernel", (unsigned) rows, 1, 1, 256, 1, 1, 0, stream);
    k.buf(x).buf(w).scalar(cols).scalar(ld).scalar(eps);
    k.done();
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    // The engine validates the resolved config at startup with the same rule (generate.cpp), so this only
    // fires for a caller that bypassed it; the prompt path has no error return here, so it stops the process.
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = std::pow((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs kargs = scaling.kernel_args(64);   // none: the identity constants
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    const int use_tab = rt.cos != nullptr ? 1 : 0;
    metal::Launch k("pfl_rope_kernel", (unsigned) (T * heads), 1, 1, 32, 1, 1, 0, stream);
    k.buf(x).scalar(heads).scalar(dim).scalar(ld).scalar(pos0).scalar(theta_scale)
     .scalar(kargs.freq_scale).scalar(kargs.corr_low).scalar(kargs.corr_high).scalar(kargs.ext_factor)
     .scalar(kargs.attn_factor)
     .buf(strata::kernels::mrope_table())
     .scalar(use_tab).buf(rt.cos).buf(rt.sin).scalar(rt.max_pos);
    k.done();
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    metal::Launch k("pfl_split_q_kernel", blocks_for((int64_t) T * 24 * 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(q_full).buf(q).scalar(T);
    k.done();
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    metal::Launch k("pfl_gate_attn_kernel", blocks_for((int64_t) T * 24 * 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(attn).buf(q_full).buf(out16).scalar(T);
    k.done();
    check("gate_attn");
}

}  // namespace strata::prefill

// src/kernels/metal/gr.mm - the port of src/kernels/cuda/gr.cu's host half (K9).  gr_workspace_init is
// host arithmetic and moves verbatim; gr_read/gr_write launch the MSL kernels.  All three activation modes
// (bf16, fp32, native MMVF) are live - the native path is the bf16_gemv/native_gr ports.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

}  // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),      // 4: lo (FP32 activation experiment)
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);

    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
        bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
        native_gr_down_silu(ws.lo, hc_lr, hc, stream);
        bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);
        // The existing null-injection contract identifies the final mixer
        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    } else {
        const int act32 = use_fp32 ? 1 : 0;
        // xn/xq as the norm's outputs, lo/lq as the down projection's - the inactive one may bind null
        metal::Launch n("gr_norm_kernel", (unsigned) hc, 1, 1, THREADS, 1, 1, 0, stream);
        n.buf(R).buf(w_norm).scalar(eps).scalar(n_embd).buf(ws.xn).buf(ws.xq).scalar(act32)
         .scalar((unsigned) THREADS);
        n.done();

        metal::Launch d("gr_down_kernel", (unsigned) hc_lr, 1, 1, THREADS, 1, 1, 0, stream);
        d.buf(use_fp32 ? (const float*) ws.xn : nullptr)
         .buf(use_fp32 ? (const uint16_t*) ws.xq : ws.xq)
         .buf(w_down).scalar(hc_dim).scalar(hc_lr).scalar(hc)
         .buf(ws.lo).buf(ws.lq).scalar(act32).scalar((unsigned) THREADS);
        d.done();

        metal::Launch g("gr_gate_kernel", (unsigned) ((hc_dim + WARPS - 1) / WARPS), 1, 1, THREADS, 1, 1, 0,
                        stream);
        g.buf(use_fp32 ? (const float*) ws.lo : nullptr)
         .buf(use_fp32 ? (const uint16_t*) ws.lq : ws.lq)
         .buf(w_up).buf(ws.xn).scalar(hc_dim).scalar(hc_lr).buf(ws.gated).scalar(act32);
        g.done();

        metal::Launch m("gr_mean_kernel", (unsigned) ((n_embd + THREADS - 1) / THREADS), 1, 1, THREADS, 1, 1,
                        0, stream);
        m.buf(ws.gated).scalar(n_embd).scalar(hc).buf(mixed);
        m.done();
    }
    if (w_inject != nullptr) {
        if (use_native) {
            bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);
        } else {
            const int nthreads = 32 * hc;
            metal::Launch ij("gr_inject_kernel", 1, 1, 1, (unsigned) nthreads, 1, 1, 0, stream);
            ij.buf(use_fp32 ? (const float*) ws.xn : nullptr)
              .buf(use_fp32 ? (const uint16_t*) ws.xq : ws.xq)
              .buf(w_inject).scalar(hc_dim).scalar(hc).buf(inject)
              .scalar(fp32_activations || use_native ? 1 : 0);
            ij.done();
        }
    }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gr_read launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            std::fprintf(stderr, "gr_read: %s\n", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    const long long n = (long long) s.hc * s.n_embd;
    const int blocks = (int) ((n + THREADS - 1) / THREADS);
    if (native_mmvf) {
        native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);
    } else {
        metal::Launch k("gr_write_kernel", (unsigned) blocks, 1, 1, THREADS, 1, 1,
                        (size_t) s.hc * sizeof(float), stream);
        k.buf(R).buf(block_out).buf(inject).scalar((int) s.n_embd).scalar((int) s.hc).buf(R_out)
         .scalar((long) ((long long) blocks * THREADS));
        k.done();
    }
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gr_write: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels

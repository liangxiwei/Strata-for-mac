// src/kernels/metal/fused_gr.mm - the port of src/kernels/cuda/fused_gr.cu's host half (K16).  The read is
// three launches per token (norm, down, up), every pointer a BOUND buffer argument - the CUDA struct's
// by-value pointer table does not survive on Apple Silicon (fused_gr.metal's file comment has the
// measurements), so the T-token GrMulti becomes T one-token launches whose bound arguments the runtime
// resolves, interior pointers included.  The single-token read is the same path with T = 1 and a private xn
// buffer: the CUDA single-token kernel stages xn (40 KB) in shared memory and this GPU's threadgroup limit
// is 32 KB (measured), and the multi path already stages xn in device memory, so T = 1 through it is the
// whole single-token read, bit for bit.
//
// The v3 read (STRATA_GR_V3, another summation order) and the split/staged variants (#315) are CUDA
// shared-memory and cp.async stagings of the SAME arithmetic; this port's down kernel reads the activations
// straight from device memory, where there is nothing to choose between them, so the default read is the
// only one and it is what runs (fused_gr_variant says so).
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int N = 2560;                    // n_embd
constexpr int HC = 4;                      // streams
constexpr int D = N * HC;                  // 10240
constexpr int LR = 320;                    // hc_lr
constexpr int DOWN_BLOCKS = LR / WARPS;    // 40; one more block carries the inject rows
constexpr int UPM_BLOCKS = N / 16;         // 160 blocks of 16 columns

/// The single-token read's xn staging: a small ring of permanent device buffers, handed out in turn.  CUDA's
/// xn lives in shared memory, which is per LAUNCH - two reads captured into one graph each keep their own -
/// so a ring keeps two captured reads from sharing a buffer the CUDA original would not have shared.  The
/// multi read passes its own scratch and never comes here.
float* take_xn_slot() {
    static float* ring[4] = {nullptr, nullptr, nullptr, nullptr};
    static unsigned next = 0;
    const unsigned i = next++ & 3u;
    if (ring[i] == nullptr) {
        const cudaError_t e = cudaMalloc((void**) &ring[i], (size_t) D * sizeof(float));
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read: no xn staging buffer: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
    return ring[i];
}

/// STRATA_METAL_GR_NORM1=0 keeps the two-pass norm kernel; fused_gr_norm1_kernel writes the same bits in one pass.
const char* norm_kernel() {
    static const char* name = [] {
        const char* v = std::getenv("STRATA_METAL_GR_NORM1");
        return v != nullptr && std::atoi(v) == 0 ? "fused_gr_norm_kernel" : "fused_gr_norm1_kernel";
    }();
    return name;
}

/// One token's read: the norm, the down projection, the up projection, all pointers bound as arguments.
/// The CUDA original tiles the multi down kernel by what fits the card's shared-memory opt-in; the Metal
/// down kernel reads xn from device memory, so every token count fits one launch and there is no tiling.
void launch_token(const FusedGrArgs& a, float* xn, void* stream) {
    const int apply = a.apply ? 1 : 0;
    metal::Launch n(norm_kernel(), 1, 1, 1, THREADS, 1, 1, 0, stream);
    n.buf(a.R).buf(a.bo_prev).buf(a.inj_prev).buf(a.w_norm).scalar(a.eps).scalar(apply).buf(a.rs).buf(xn);
    n.done();

    metal::Launch d("fused_gr_down_kernel", (unsigned) (DOWN_BLOCKS + 1), 1, 1, THREADS, 1, 1, 0, stream);
    d.buf(a.w_down).buf(a.w_inject).buf(xn).buf(a.lo).buf(a.inject_out);
    d.done();

    metal::Launch u("fused_gr_up_kernel", (unsigned) UPM_BLOCKS, 1, 1, THREADS, 1, 1, 0, stream);
    u.buf(a.w_up).buf(a.lo).buf(a.R).buf(a.R_out).buf(a.bo_prev).buf(a.inj_prev).buf(a.w_norm).buf(a.rs)
     .buf(a.mixed).scalar(apply);
    u.done();
}

}  // namespace

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    launch_token(a, take_xn_slot(), stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf, int stamp_i0) {
    // the profile's stamps after the norm and after the down projection, the CUDA launch_multi's own two
    // (gpu_stamp is ported now - on this backend a stamp is a monotonic launch counter, see verify_kernels.mm)
    static const bool v3_note = [] {
        const char* v = std::getenv("STRATA_GR_V3");
        if (v != nullptr && std::atoi(v) != 0)
            std::fprintf(stderr, "strata: STRATA_GR_V3=1 is not ported to Metal - the default read runs\n");
        return true;
    }();
    (void) v3_note;
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // the norm for every token, then the down projection, then the up - the CUDA dataflow; each down/up
    // launch touches only its own token's xn/lo, so per-token launches keep exactly that order
    for (int t = 0; t < n_tok; ++t) {
        metal::Launch n(norm_kernel(), 1, 1, 1, THREADS, 1, 1, 0, stream);
        const FusedGrArgs& x = a[t];
        const int apply = x.apply ? 1 : 0;
        n.buf(x.R).buf(x.bo_prev).buf(x.inj_prev).buf(x.w_norm).scalar(x.eps).scalar(apply).buf(x.rs)
         .buf(xn_scratch + (size_t) t * D);
        n.done();
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        metal::Launch d("fused_gr_down_kernel", (unsigned) (DOWN_BLOCKS + 1), 1, 1, THREADS, 1, 1, 0, stream);
        d.buf(x.w_down).buf(x.w_inject).buf(xn_scratch + (size_t) t * D).buf(x.lo).buf(x.inject_out);
        d.done();
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        const int apply = x.apply ? 1 : 0;
        metal::Launch u("fused_gr_up_kernel", (unsigned) UPM_BLOCKS, 1, 1, THREADS, 1, 1, 0, stream);
        u.buf(x.w_up).buf(x.lo).buf(x.R).buf(x.R_out).buf(x.bo_prev).buf(x.inj_prev).buf(x.w_norm).buf(x.rs)
         .buf(x.mixed).scalar(apply);
        u.done();
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

int fused_gr_variant() {
    // 1 = kHcPlain.  The split and staged variants are CUDA shared-memory / cp.async stagings of the plain
    // read's arithmetic (checked bit for bit there by fused_gr_check); the Metal down kernel reads the
    // activations from device memory, so here they would all be the same code - the plain read is the answer.
    return 1;
}

void fused_gr_check() {
    static const bool said = [] {
        std::fprintf(stderr, "strata hc: METAL: the hyper-connection read runs as the plain one (the split and "
                             "staged variants are CUDA stagings of the same arithmetic; this port has one read)\n");
        return true;
    }();
    (void) said;
}

}  // namespace strata::kernels

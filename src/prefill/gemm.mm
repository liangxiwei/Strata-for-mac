// Metal prefill GEMM: Y[T, ldy] = X[T,K] * W[N,K]^T, FP32 accumulation.
// The default SIMD-group matrix kernels read FP16 directly and expand BF16 in
// threadgroup memory, avoiding a full widened copy of the weights. Strided
// output, partial tiles and beta=0/1 are checked against a double reference.
// STRATA_METAL_MPS_GEMM=1 keeps the MPS comparison path. MPS rejects BF16 input
// on the tested M2 Max, so that path first widens BF16 into FP32 scratch.
// Both paths follow the caller's serial stream. MPS ends the current compute
// encoder and uses the same command buffer; it cannot be recorded in a tape.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/platform/metal_launch.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace strata::prefill {
namespace {

// the per-Gemm state that does not fit the fixed header: the f32 widen scratch (the .cu kept its
// hipBLASLt tuning state here on HIP; the field is a plain void* and this port owns it the same way)
struct WidenScratch {
    float* x = nullptr;     // X's f32 image (T*K floats, grown on demand)
    float* w = nullptr;     // W's f32 image (N*K floats)
    size_t x_elems = 0, w_elems = 0;
    ~WidenScratch() { if (x) cudaFree(x); if (w) cudaFree(w); }
};

// MPSMatrixMultiplication objects bake M, N, K, alpha, beta and the input dtype at init; they also
// compile/cache their pipelines per object, so one is kept per key (retained; manual refcounting - an
// autorelease anywhere in the call path is what round M1's lesson forbids).
struct MpsKey {
    uint32_t m, n, k, beta_bits, in_f32;
    bool operator<(const MpsKey& o) const {
        return std::tie(m, n, k, beta_bits, in_f32) < std::tie(o.m, o.n, o.k, o.beta_bits, o.in_f32);
    }
};
std::mutex g_mps_mu;
std::map<MpsKey, MPSMatrixMultiplication*> g_mps;

void die(const char* what) {
    std::fprintf(stderr, "prefill gemm: %s\n", what);
    std::exit(1);
}

MPSMatrixMultiplication* mps_gemm(id<MTLDevice> dev, uint64_t M, uint64_t N, uint64_t K, float beta,
                                  bool in_f32) {
    uint32_t beta_bits = 0;
    static_assert(sizeof beta_bits == sizeof beta, "");
    std::memcpy(&beta_bits, &beta, sizeof beta_bits);
    const MpsKey key{(uint32_t) M, (uint32_t) N, (uint32_t) K, beta_bits, in_f32 ? 1u : 0u};
    std::lock_guard<std::mutex> lock(g_mps_mu);
    const auto it = g_mps.find(key);
    if (it != g_mps.end()) return it->second;
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev
          transposeLeft:NO                   // A = X, [T, K]
         transposeRight:YES                  // B = W, [N, K] used transposed: the .cu's OP_T
             resultRows:(NSUInteger) M
          resultColumns:(NSUInteger) N
        interiorColumns:(NSUInteger) K
                  alpha:1.0
                   beta:(double) beta];
    if (mm == nil) die("MPSMatrixMultiplication init failed");
    g_mps.emplace(key, mm);                  // the map's copy of the pointer rides our retain
    return mm;
}

/// Wrap a runtime pointer as an MPSMatrix.  `row_bytes` must be >= cols * element size (MPS asserts);
/// the byte offset is the registry's resolution of the interior pointer, exactly what Launch binds.
MPSMatrix* wrap(const void* ptr, uint64_t rows, uint64_t cols, uint64_t row_bytes, MPSDataType dt,
                const char* what) {
    void* b = nullptr;
    uint64_t off = 0;
    if (!metal::resolve_buffer(ptr, &b, &off)) {
        std::fprintf(stderr, "prefill gemm: %s: %p is not a runtime buffer\n", what, ptr);
        std::exit(1);
    }
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger) rows
                                                                   columns:(NSUInteger) cols
                                                                  rowBytes:(NSUInteger) row_bytes
                                                                  dataType:dt];
    MPSMatrix* m = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>) b
                                              offset:(NSUInteger) off
                                           descriptor:d];
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>) b;
    [buf release];                           // resolve_buffer's retain: the matrix keeps its own
    if (m == nil) die("MPSMatrix wrap failed");
    return m;
}

/// Encode one GEMM on the stream's open command buffer.  `in_f32`: inputs are the f32 widen images
/// (the pointers are the raw 16-bit or 32-bit buffers; the dtype flag tells MPS which they are).
void encode(void* stream, const void* X, const void* W, float* Y, uint64_t T, uint64_t N,
            uint64_t K, uint64_t ldy, float beta, bool in_f32) {
    @autoreleasepool {
    id<MTLDevice> dev = (__bridge id<MTLDevice>) metal::mtl_device();
    if (dev == nil) die("no Metal device");
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
        die("the MPS GEMM cannot be captured into a stream tape (the prompt path never captures)");
    const MPSDataType dt = in_f32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
    const uint64_t in_bytes = in_f32 ? 4 : 2;
    MPSMatrix* xm = wrap(X, T, K, K * in_bytes, dt, "X");
    MPSMatrix* wm = wrap(W, N, K, K * in_bytes, dt, "W");
    // (the casts below are for the reader: X/W are 16-bit or the f32 widen images per `in_f32`)
    MPSMatrix* ym = wrap(Y, T, N, ldy * 4, MPSDataTypeFloat32, "Y");
    MPSMatrixMultiplication* mm = mps_gemm(dev, T, N, K, beta, in_f32);
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) metal::mtl_command_buffer(stream);
    [mm encodeToCommandBuffer:cb leftMatrix:xm rightMatrix:wm resultMatrix:ym];
    [cb release];                            // mtl_command_buffer's retain; the stream keeps its own
    [xm release]; [wm release]; [ym release];
    }
}

/// Grow a widen-scratch half to `want` f32 elements (never shrunk; freed with the Gemm).
float* grow(float* cur, size_t& have, uint64_t want) {
    if (want <= have) return cur;
    size_t elems = have > (size_t) want ? have : (size_t) want;
    if (cur != nullptr) cudaFree(cur);
    void* p = nullptr;
    if (const cudaError_t e = cudaMalloc(&p, elems * sizeof(float)); e != cudaSuccess)
        die("the bf16->f32 widen scratch does not fit");
    have = elems;
    return (float*) p;
}

void widen(const uint16_t* src, uint64_t n, float* dst, void* stream) {
    metal::Launch k("pfl_widen_f32_kernel", (unsigned) ((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(src).buf(dst).scalar((unsigned long) n);
    k.done();
}

}  // namespace

Gemm::~Gemm() {
    delete static_cast<WidenScratch*> (hipblaslt_state_);
    hipblaslt_state_ = nullptr;
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace,
                         size_t ws_bytes, std::string& err) {
    (void) ws_bytes;
    if (metal::mtl_device() == nullptr) {
        err = "prefill gemm: no Metal device";
        return false;
    }
    handle_ = nullptr;                       // no cuBLAS handle on this backend
    stream_ = stream;
    external_ = true;
    workspace_ = workspace;                  // bookkeeping only: MPS manages its own intermediates
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    if (hipblaslt_state_ == nullptr) hipblaslt_state_ = new WidenScratch;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    (void) ws_bytes;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    if (metal::mtl_device() == nullptr) {
        err = "prefill gemm: no Metal device";
        return false;
    }
    stream_ = stream;
    workspace_ = nullptr;                    // MPS needs no caller workspace
    if (scratch_elems > 0) {
        void* p = nullptr;
        if (const cudaError_t e = cudaMalloc(&p, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = std::string("prefill gemm: dequant scratch of ") + cudaGetErrorString(e);
            return false;
        }
        scratch_ = (uint16_t*) p;
    }
    scratch_elems_ = scratch_elems;
    if (hipblaslt_state_ == nullptr) hipblaslt_state_ = new WidenScratch;
    return true;
}

// STRATA_METAL_PREFILL_GEMM2=0 keeps the 16 x 32 tile kernels. The 64 x 64 ones (prefill.metal) give the same
// bits; they need K % 32 == 0 and 8-byte aligned operand rows for their vector loads.
static bool gemm2_ok(const void* X, const void* W, int64_t K) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_METAL_PREFILL_GEMM2");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on && K % 32 == 0 && ((uintptr_t) X % 8) == 0 && ((uintptr_t) W % 8) == 0;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    static const bool mps = std::getenv("STRATA_METAL_MPS_GEMM") != nullptr;
    if (!mps && gemm2_ok(X, W, K)) {
        metal::Launch k("pfl_gemm2_bf16", (unsigned)((N + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, stream_);
        k.buf(X).buf(W).buf(Y).scalar((unsigned)T).scalar((unsigned)N).scalar((unsigned)K)
         .scalar((unsigned)ldy).scalar(beta);
        k.done();
        return;
    }
    if (!mps) {
        metal::Launch k("pfl_gemm_bf16", (unsigned)((N + 31) / 32), (unsigned)((T + 15) / 16), 1,
                        128, 1, 1, 0, stream_);
        k.buf(X).buf(W).buf(Y).scalar((unsigned)T).scalar((unsigned)N).scalar((unsigned)K)
         .scalar((unsigned)ldy).scalar(beta);
        k.done();
        return;
    }
    auto* s = static_cast<WidenScratch*> (hipblaslt_state_);
    if (s == nullptr) {
        s = new WidenScratch;
        hipblaslt_state_ = s;
    }
    const uint64_t xn = (uint64_t) T * (uint64_t) K, wn = (uint64_t) N * (uint64_t) K;
    s->x = grow(s->x, s->x_elems, xn);
    s->w = grow(s->w, s->w_elems, wn);
    widen(X, xn, s->x, stream_);
    widen(W, wn, s->w, stream_);
    encode(stream_, s->x, s->w, Y, (uint64_t) T, (uint64_t) N, (uint64_t) K, (uint64_t) ldy, beta, true);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    static const bool mps = std::getenv("STRATA_METAL_MPS_GEMM") != nullptr;
    if (!mps && gemm2_ok(X, W, K)) {
        metal::Launch k("pfl_gemm2_f16", (unsigned)((N + 63) / 64), (unsigned)((T + 63) / 64), 1, 128, 1, 1, 0, stream_);
        k.buf(X).buf(W).buf(Y).scalar((unsigned)T).scalar((unsigned)N).scalar((unsigned)K)
         .scalar((unsigned)ldy).scalar(beta);
        k.done();
        return;
    }
    if (!mps) {
        metal::Launch k("pfl_gemm_f16", (unsigned)((N + 31) / 32), (unsigned)((T + 15) / 16), 1,
                        128, 1, 1, 0, stream_);
        k.buf(X).buf(W).buf(Y).scalar((unsigned)T).scalar((unsigned)N).scalar((unsigned)K)
         .scalar((unsigned)ldy).scalar(beta);
        k.done();
        return;
    }
    encode(stream_, X, W, Y, (uint64_t) T, (uint64_t) N, (uint64_t) K, (uint64_t) ldy, beta, false);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                  int64_t K, int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill

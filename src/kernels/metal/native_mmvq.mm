// src/kernels/metal/native_mmvq.mm - the port of src/kernels/cuda/native_mmvq.cu's host half.  Same entry
// points (include/strata/kernels/native_mmvq.hpp), same validation, same launch decisions (the small-K
// rule, the multi_exact layouts, the ncols switch), same forwards to iq_mmvq for the i-quants and Q5_1 -
// the kernels are metal::Launch chains whose argument order IS each kernel's [[buffer(N)]] order (rule A)
// and every pointer rides as a bound buffer (rule 9: nothing but scalars travels as bytes).
//
// The kernel names carry the CUDA templates' arguments: native_q5_k_mmvq_kernel_small/_large (<SmallK>),
// native_small_mmvq_kernel_<fmt>_small/_large (<Weight, Qi, SmallK>) and
// native_mmvq_multi_kernel_<fmt>_r1/_r2/_r4 (<F, NCOLS, NW, ROWS> with ROWS spelled and NCOLS/NW runtime -
// see native_mmvq.metal's file comment for why runtime NCOLS/NW keep the multi == single bitwise contract).
//
// THE ONE HOST-SIDE ADDITION (the iq_kernels.mm precedent, "make_host_visible"): the shim's cudaMemcpy
// orders only against the NULL stream, and callers read the quantizer's output with a bare cudaMemcpy right
// after launching on their own stream - native_quantize_q8_1 therefore commits+waits the caller's stream
// before returning.  Inside a graph capture there is no pending command buffer, so it is a no-op there.
// The bridge this file replaces behaved the same way (it forwarded to iq_kernels' quantize_q8_1_rows, which
// carries the same sync).
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/metal_mmvq.hpp"
#include "strata/platform/metal_launch.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QUANT_THREADS = 256;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int MAX_NCOLS = 8;

// llama.cpp's Q81Block: { fp16 d ; fp16 sum ; int8 qs[32] } = 36 bytes (ggml-common.h's block_q8_1)
constexpr int Q81_BYTES = 36;

bool g_multi_exact = true;   // CUDA's default: the ncols == 1 layout (every column bitwise equal to it)

struct ExpandedIq4 { void* data; size_t bytes; int n_in, n_out; };
std::mutex g_expanded_mu;
std::map<const void*, ExpandedIq4> g_expanded;
size_t g_expanded_bytes = 0;

void launch_check() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "native MMVQ launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// the shim-gap workaround of the file comment (iq_kernels.mm's make_host_visible, same reason)
void make_host_visible(void* stream) { cudaStreamSynchronize((cudaStream_t) stream); }

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}

// The format table native_mmvq.metal instantiates: the two kernel stems (without the _small/_large and
// _r1/_r2/_r4 suffixes) and the two launch-decision constants - DIV, the block element count, and BPI, the
// blocks-per-iteration at NW = 4 warps (the CUDA file's per-format BLOCKS_PER_ITER / SmallTraits::BPI).
struct FmtHost {
    const char* single;
    const char* multi;
    int div;
    int bpi;
};
constexpr FmtHost k_fmts[10] = {
    {"native_q5_k_mmvq_kernel",         "native_mmvq_multi_kernel_q5_k",    256, 8},    // VDR*WARPS*WARP/QI
    {"native_q4_k_mmvq_kernel",         "native_mmvq_multi_kernel_q4_k",    256, 8},
    {"native_q2_0_mmvq_kernel",         "native_mmvq_multi_kernel_q2_0",     64, 64},   // WARPS*WARP/2
    {"native_q3_k_mmvq_kernel",         "native_mmvq_multi_kernel_q3_k",    256, 8},    // WARPS*WARP/16
    {"native_q6_k_mmvq_kernel",         "native_mmvq_multi_kernel_q6_k",    256, 4},    // WARPS*WARP/32
    {"native_iq4_xs_mmvq_kernel",       "native_mmvq_multi_kernel_iq4_xs",  256, 16},   // 4*WARPS*WARP/32
    {"native_small_mmvq_kernel_q4_0",   "native_mmvq_multi_kernel_q4_0",     32, 64},   // 2*WARPS*WARP/Qi, Qi=4
    {"native_small_mmvq_kernel_q5_0",   "native_mmvq_multi_kernel_q5_0",     32, 64},
    {"native_small_mmvq_kernel_q8_0",   "native_mmvq_multi_kernel_q8_0",     32, 32},   // Qi=8
    {"native_small_mmvq_kernel_iq4_nl", "native_mmvq_multi_kernel_iq4_nl",   32, 64},
};

// STRATA_METAL_IQ4_DIRECT=0 keeps the byte_perm codebook kernels for IQ4_XS / IQ4_NL single columns; the
// direct kernels (native_mmvq.metal) compute the same bits from the original weights, faster and without a view
bool iq4_direct() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_METAL_IQ4_DIRECT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// the ncols == 1 kernels: the small-K rule picks 4 rows per block (grid rounded up) or 1 row (grid n_out)
void launch_single(int f, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, void* stream) {
    const FmtHost& fmt = k_fmts[f];
    char name[80];
    if ((f == 5 || f == 9) && iq4_direct()) {
        bool expanded = false;
        if (f == 5) {
            std::lock_guard<std::mutex> lock(g_expanded_mu);
            expanded = g_expanded.count(weights) != 0;
        }
        if (!expanded) {
            // simdgroup-per-row for the wide IQ4_XS matrices, 4 rows per threadgroup below (measured crossover)
            static const bool sg_on = [] {
                const char* v = std::getenv("STRATA_METAL_IQ4_SG");   // =0: 4 rows per threadgroup everywhere
                return v == nullptr || std::atoi(v) != 0;
            }();
            const bool sg = sg_on && f == 5 && n_out >= 2560;
            metal::Launch k(sg ? "native_iq4_xs_direct_sg4" : f == 5 ? "native_iq4_xs_direct_r4" : "native_iq4_nl_direct_r4",
                            (unsigned) (((size_t) n_out + (sg ? 15 : 3)) / (sg ? 16 : 4)), 1, 1, WARP, WARPS, 1, 0, stream);
            k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out);
            k.done();
            return;
        }
    }
    if (f == 5) {
        std::lock_guard<std::mutex> lock(g_expanded_mu);
        const auto found = g_expanded.find(weights);
        if (found != g_expanded.end() && found->second.n_in == n_in && found->second.n_out == n_out) {
            const bool small = n_in / fmt.div < fmt.bpi;
            metal::Launch k(small ? "native_iq4_xs_expanded_small" : "native_iq4_xs_expanded_large",
                            (unsigned) ((n_out + (small ? 3 : 0)) / (small ? 4 : 1)),
                            1, 1, WARP, WARPS, 1, 0, stream);
            k.buf(found->second.data).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out); k.done();
            return;
        }
    }
    if (n_in / fmt.div < fmt.bpi) {
        const unsigned blocks = (unsigned) (((size_t) n_out + WARPS - 1) / WARPS);
        std::snprintf(name, sizeof name, "%s_small", fmt.single);
        metal::Launch k(name, blocks, 1, 1, WARP, WARPS, 1, 0, stream);
        k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out);
        k.done();
    } else {
        std::snprintf(name, sizeof name, "%s_large", fmt.single);
        metal::Launch k(name, (unsigned) n_out, 1, 1, WARP, WARPS, 1, 0, stream);
        k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out);
        k.done();
    }
}

// launch_multi_n: the exact layout (NW = WARPS, ROWS by the same small-K rule) or llama.cpp's upstream
// table (NW = NCOLS <= 4 ? 4 : 2, always 2 rows per block) when multi_exact is off
void launch_multi_n(int f, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                    void* stream) {
    const FmtHost& fmt = k_fmts[f];
    char name[80];
    if (!g_multi_exact) {
        const int nw = ncols <= 4 ? 4 : 2;
        const unsigned blocks = (unsigned) (((size_t) n_out + 1) / 2);
        std::snprintf(name, sizeof name, "%s_r2", fmt.multi);
        metal::Launch k(name, blocks, 1, 1, WARP, (unsigned) nw, 1, 0, stream);
        k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out).scalar(ncols).scalar(nw);
        k.done();
        return;
    }
    if (n_in / fmt.div < fmt.bpi) {
        const unsigned blocks = (unsigned) (((size_t) n_out + WARPS - 1) / WARPS);
        std::snprintf(name, sizeof name, "%s_r4", fmt.multi);
        metal::Launch k(name, blocks, 1, 1, WARP, WARPS, 1, 0, stream);
        k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out).scalar(ncols).scalar(WARPS);
        k.done();
    } else {
        std::snprintf(name, sizeof name, "%s_r1", fmt.multi);
        metal::Launch k(name, (unsigned) n_out, 1, 1, WARP, WARPS, 1, 0, stream);
        k.buf(weights).buf(x_q8_1).buf(y).scalar(n_in).scalar(n_out).scalar(ncols).scalar(WARPS);
        k.done();
    }
}

void launch_multi(int f, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    switch (ncols) {
        case 2: case 3: case 4: case 5: case 6: case 7: case 8:
            launch_multi_n(f, weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

// the shared per-format shape of the ten own-kernel entry points (validate + the two paths)
void mmvq_common(int f, int block_elems, const void* weights, const void* x_q8_1, float* y, int n_in,
                 int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi(f, weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    launch_single(f, weights, x_q8_1, y, n_in, n_out, stream);
    launch_check();
}

// the shared shape of the ten f32 compositions (caller owns the Q8_1 scratch, native_q8_1_bytes sizes it)
void f32_common(int f, int block_elems, const void* weights, const float* x, void* scratch_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    mmvq_common(f, block_elems, weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

}  // namespace

std::size_t metal_iq4_prepare(int type, const void* weights, int n_in, int n_out) {
    if (type != 23 || !weights || n_in <= 0 || n_in % 256 || n_out <= 0) return 0;
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) return 0;
    const char* enabled = std::getenv("STRATA_METAL_IQ4_EXPAND");
    // Opt-in only. Since the direct-codebook kernels (STRATA_METAL_IQ4_DIRECT, default on) the view is slower
    // than no view in full decode (19.76 vs 21.78-21.99 tok/s, decode-opt2) and costs 2.24 GiB; a view, when
    // made, still takes precedence for its matrix. Retain the original path on allocation failure or when the
    // bounded cache is full.
    if (!enabled || std::atoi(enabled) == 0) return 0;
    const size_t bytes = (size_t) (n_in / 256) * n_out * 264;
    const size_t budget = std::min<size_t>(4ull << 30, total / 16);
    std::lock_guard<std::mutex> lock(g_expanded_mu);
    if (g_expanded.count(weights) || bytes > budget || g_expanded_bytes > budget - bytes || bytes > free / 4)
        return 0;
    void* data = nullptr;
    if (cudaMalloc(&data, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
    const unsigned long words = (size_t) (n_in / 256) * n_out * 32;
    metal::Launch expand("native_iq4_xs_expand", (unsigned) ((words + 255) / 256), 1, 1, 256, 1, 1, 0, nullptr);
    expand.buf(weights).buf(data).scalar(words); expand.done();
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) { cudaFree(data); cudaGetLastError(); return 0; }
    g_expanded.emplace(weights, ExpandedIq4{data, bytes, n_in, n_out});
    g_expanded_bytes += bytes;
    return bytes;
}

void metal_iq4_release(const void* weights) {
    std::lock_guard<std::mutex> lock(g_expanded_mu);
    const auto found = g_expanded.find(weights);
    if (found == g_expanded.end()) return;
    cudaFree(found->second.data);
    g_expanded_bytes -= found->second.bytes;
    g_expanded.erase(found);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return (std::size_t) ncols * (std::size_t) (n_in / Q8K) * Q81_BYTES;
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const long long n_total = (long long) n_in * ncols;
    const unsigned blocks = (unsigned) ((n_total + QUANT_THREADS - 1) / QUANT_THREADS);
    metal::Launch k("native_quantize_q8_1_kernel", blocks, 1, 1, QUANT_THREADS, 1, 1, 0, stream);
    k.buf(x).buf(x_q8_1).scalar(n_total);
    k.done();
    make_host_visible(stream);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(0, QK, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(0, QK, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(2, 64, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(2, 64, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(3, 256, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(3, 256, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    mmvq_common(5, 256, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    f32_common(5, 256, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(1, 256, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(1, 256, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(4, 256, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(4, 256, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(6, 32, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(6, 32, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(7, 32, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(7, 32, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    mmvq_common(8, 32, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    f32_common(8, 32, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    mmvq_common(9, 32, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    f32_common(9, 32, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = (std::size_t) (n_in / block_elems) * (std::size_t) block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / (std::size_t) n_out) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * (std::size_t) n_out;
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                 int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

}  // namespace strata::kernels

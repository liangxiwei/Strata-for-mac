// src/kernels/metal/native_qsa.mm - the port of src/kernels/cuda/native_qsa.cu's host half (K18).  The
// validation that throws (dimensions, span alignment, overlap or EXACT input/output alias, stream) is the
// CUDA file's, verbatim; each k<<<...>>> becomes a metal::Launch with every pointer a bound buffer
// argument (R9) and only the small scalars riding as bytes.  The template <int BlockSize>'s two widths
// are two launch sites of one kernel whose block argument carries the width.  No sync, exactly as the
// CUDA original promises ("no allocation or synchronization") - the sticky-launch-error check is the
// cudaGetLastError() the CUDA file ends with.
#include "strata/kernels/native_qsa.hpp"
#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}

}  // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    // the CUDA file's own dispatch: 256 threads below n_cols 1024, else 1024
    if (n_cols < 1024) {
        metal::Launch k("nqs_norm_kernel", unsigned(n_rows), 1, 1, 256, 1, 1, 0, stream);
        k.buf(input).buf(gamma).buf(output).scalar(n_cols).scalar(epsilon).scalar(256);
        k.done();
    } else {
        metal::Launch k("nqs_norm_kernel", unsigned(n_rows), 1, 1, 1024, 1, 1, 0, stream);
        k.buf(input).buf(gamma).buf(output).scalar(n_cols).scalar(epsilon).scalar(1024);
        k.done();
    }
    check_launch();
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    metal::Launch k("nqs_gate_kernel", unsigned((count + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(attn).buf(q_full).buf(output).scalar(n_head).scalar(head_dim);
    k.done();
    check_launch();
}

}  // namespace strata::kernels

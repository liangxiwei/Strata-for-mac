// src/kernels/metal/native_gdn_preprocess.mm - the port of src/kernels/cuda/native_gdn_preprocess.cu's host
// half (K10's native sibling, wave 3).  Same header contract (include/strata/kernels/native_gdn_preprocess.hpp);
// every k<<<grid, block, 0, stream>>> becomes a metal::Launch whose chained .buf()/.scalar() order is
// EXACTLY the kernel signature's [[buffer(N)]] order, every pointer a bound buffer argument and only the
// ints/floats scalars (rule 9).  Host-side validation ports verbatim from the .cu, comments included - they
// record the API contract (disjoint spans, width 128, count <= 65535) the callers rely on.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    metal::Launch kern("conv_silu", unsigned((channels + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(history).buf(input).buf(weights).buf(raw_output).buf(silu_output).scalar(int(channels));
    kern.done();
    check_launch();
}
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    metal::Launch kern("l2_norm", unsigned(rows), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(input).scalar(epsilon / S).scalar(1.0f / sqrtf(float(S)));
    kern.done();
    check_launch();
}
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    metal::Launch kern("beta_sigmoid", unsigned((heads + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(beta).scalar(int(heads));
    kern.done();
    check_launch();
}
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    metal::Launch kern("gate_softplus", unsigned((heads + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(alpha).buf(dt).buf(ssm_a).buf(gate).scalar(int(heads));
    kern.done();
    check_launch();
}
void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    metal::Launch kern("out_norm", unsigned(heads), 1, 1, 256, 1, 1, 0, stream);
    kern.buf(output).buf(z).buf(gamma).buf(destination).scalar(epsilon);
    kern.done();
    check_launch();
}
} // namespace strata::kernels

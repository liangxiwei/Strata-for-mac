// src/kernels/metal/native_moe.mm - the port of src/kernels/cuda/native_moe.cu's host half.  Same contract,
// same span/overlap validation and the same exception strings; the launch is a metal::Launch whose argument
// order IS the kernel's [[buffer(N)]] order (the block size rides as a scalar - rule 7 - and a null `shared`
// binds nil - rule 4).
#include "strata/kernels/native_moe.hpp"
#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

void launch_combine(const float* parts, const float* weights, const float* shared, float* output,
                    int64_t n_embd, int64_t k, int n_tok, void* stream) {
    metal::Launch c("nmoe_combine_kernel", (unsigned) ((n_embd + 255) / 256), (unsigned) n_tok, 1, 256, 1, 1,
                    0, stream);
    c.buf(parts).buf(weights).buf(shared).buf(output)
     .scalar((int) n_embd)
     .scalar((int) k)
     .scalar((unsigned) 256);
    c.done();
}

}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    launch_combine(parts, weights, shared, output, n_embd, k, 1, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    launch_combine(parts, weights, shared, output, n_embd, k, n_tok, stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels

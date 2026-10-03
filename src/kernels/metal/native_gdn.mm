// src/kernels/metal/native_gdn.mm - the port of src/kernels/cuda/native_gdn.cu's host half (K10's native
// sibling, wave 3): the enabled flag and the pinned recurrence's launcher.  Same header contract
// (include/strata/kernels/native_gdn.hpp); every pointer is a bound buffer argument and only the ints and
// the scale ride as scalars (rule 9), chained in EXACTLY the kernel signature's [[buffer(N)]] order.
// The CUDA launch step<<<dim3(h_v, 1, S/4), dim3(32, 4)>>> becomes Launch("step", h_v, S/4, 1, 128, ...):
// the .cu's grid z axis (the column tile) is the MTL grid's y, which is the gpos.y the kernel reads (rule
// 6), and its 2D 128-thread block is the flat 128-thread group whose simdgroups are the .cu's threadIdx.y.
// Host-side validation ports verbatim from the .cu, comments included.
#include "strata/kernels/native_gdn.hpp"
#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int S = 128;

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 ||
        shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / sqrtf(float(S));
    metal::Launch kern("step", unsigned(shape.h_v), unsigned(S / 4), 1, 32 * 4, 1, 1, 0, stream);
    kern.buf(state).buf(q).buf(k).buf(v).buf(gate).buf(beta).buf(output)
        .scalar(int(shape.h_k)).scalar(int(shape.h_v)).scalar(scale);
    kern.done();
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels

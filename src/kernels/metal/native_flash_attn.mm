// src/kernels/metal/native_flash_attn.mm - the port of src/kernels/cuda/native_flash_attn.cu's host half.
// Same contract and same span validation; the attend<<<24, dim3(32,4)>>> launch is a metal::Launch chain
// whose argument order IS the kernel's [[buffer(N)]] order (a null mask binds nil - rule 4).
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}

}  // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* output, int32_t* status,
                                  const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 ||
        shapes.idx_block != 4 || shapes.idx_top_k < 256 || capacity < 256 ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 ||
        max_context < 1 || max_context > 256)
        throw std::invalid_argument("native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an explicit stream");
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, 4 * 4, 4}, {output, 24 * 256 * 4, 4},
                          {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    metal::Launch a("attend", 24, 1, 1, 32, 4, 1, 0, stream);
    a.buf(q).buf(k).buf(v).buf(step).scalar(max_context).scalar(256).scalar(0.0625f).buf(output)
     .buf(status).buf(mask);
    a.done();
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native FlashAttention launch: ") + cudaGetErrorString(result));
}

}  // namespace strata::kernels

// src/kernels/metal/native_qsa_score.mm - the port of src/kernels/cuda/native_qsa_score.cu's host half
// (K18).  The validation that throws (fixed geometry, exact capacities, aligned DISJOINT spans, explicit
// stream) is the CUDA file's, verbatim; the launch is the CUDA file's gfx1100 configuration (one block
// per pooled row, 128 threads, scalar F32 dots) - Apple GPUs have no ldmatrix/mma TF32, and the CUDA file
// itself ships that scalar kernel as the no-tensor-core contract (native_qsa_score.metal's file comment).
// No sync, as the header promises; the sticky-launch-error check is the cudaGetLastError the CUDA file
// ends with.
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int D = 128, HEADS = 4, R = 4;

struct Span { const void* p; size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}

}  // namespace

void native_qsa_score_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_score(const float* pooled, const float* query, const float* bias,
                      const QsaShapes& s, const int32_t* step, int64_t max_blocks, int64_t max_cells,
                      float* cells, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 ||
        max_cells < 1 || max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[] = {{pooled, size_t(max_blocks) * D * 4}, {query, HEADS * D * 4},
        {step, kStepCount * 4}, {cells, size_t(max_cells) * 4}, {bias, bias ? size_t(max_blocks) * 4 : 0}};
    const int count = bias ? 5 : 4;
    for (int i = 0; i < count; ++i) validate(spans[i]);
    for (int i = 0; i < count; ++i) for (int j = i + 1; j < count; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");
    metal::Launch k("nqss_score_kernel", unsigned(max_blocks), 1, 1, 128, 1, 1, 0, stream);
    k.buf(pooled).buf(query).buf(bias).buf(step).scalar((int) max_cells).buf(cells);
    k.done();
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels

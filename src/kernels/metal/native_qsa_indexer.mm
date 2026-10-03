// src/kernels/metal/native_qsa_indexer.mm - the port of src/kernels/cuda/native_qsa_indexer.cu's host half
// (K18).  The validation that throws (fixed geometry, aligned position base, positive capacity/epsilon,
// valid scaling, aligned DISJOINT spans, explicit stream) is the CUDA file's, verbatim; each k<<<...>>>
// becomes a metal::Launch whose pointers are all bound buffer arguments (R9 - the CUDA RopeTab's pointers
// ride by value there, which does not work on this GPU) and whose scalars ride as bytes.  The template
// <bool TAB>'s two launches are one kernel whose table pointers are null on the analytic path (the
// native_rope.metal spelling; the two CUDA variants' arithmetic is line-identical).
//
// THE SYNC THE CUDA FILE DOES NOT HAVE: CUDA's legacy default stream implicitly orders a blocking stream
// against null-stream memcpys, which is what a caller updating `pos_dev` (and swapping the raw row)
// between calls relies on; this backend's null stream is just another serial stream, so the kernel is
// waited before the call returns (a no-op while capturing).  Same workaround the graph-capturing
// moe_grouped_s2 carries; qsa_parity's batch-vs-sequential section is the test that pins it.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;

struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}

/// The rotation constants both entry points resolve identically (theta_scale is the CUDA file's powf).
struct Rotate {
    float theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale;
    const int32_t* mtab;
    RopeTab rt;
};
Rotate rotate_args(const RopeScaling& scaling) {
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const RopeTab rt = rope_table_for(scaling);
    return Rotate{theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
                  mrope_table(), rt};
}

}  // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency/scaling and explicit stream");
    const Span spans[] = {{raw, D * 4}, {relative_pos_device, 4}, {gamma, D * 4}, {b.tail, std::size_t(R - 1) * D * 4},
        {b.dead, D * 4}, {b.pooled, std::size_t(max_cells / R + 1) * D * 4}, {b.block_pos, 4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const Rotate rc = rotate_args(scaling);
    metal::Launch k("nqsi_append_kernel", 1, 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(raw).buf(relative_pos_device).scalar((int) pos_base).buf(gamma).scalar(epsilon)
     .buf(b.tail).buf(b.dead).buf(b.pooled).buf(b.block_pos).scalar((int) max_cells)
     .scalar(rc.theta_scale).scalar(rc.freq_scale).scalar(rc.corr_low).scalar(rc.corr_high).scalar(rc.ext_factor)
     .scalar(rc.mscale)
     .buf(rc.mtab).buf(rc.rt.cos).buf(rc.rt.sin).scalar(rc.rt.max_pos);
    k.done();
    // the shim's stream model, worked around here (the file comment above): order against the caller's
    // next null-stream copy
    cudaStreamSynchronize((cudaStream_t) stream);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions, parameters or scaling");
    const Rotate rc = rotate_args(scaling);
    if (p0 == 0) {
        metal::Launch f("nqsi_append_first_kernel", 1, 1, 1, THREADS, 1, 1, 0, stream);
        f.buf(raw).buf(gamma).scalar(epsilon).buf(b.dead).buf(b.pooled)
         .scalar(rc.theta_scale).scalar(rc.freq_scale).scalar(rc.corr_low).scalar(rc.corr_high)
         .scalar(rc.ext_factor).scalar(rc.mscale)
         .buf(rc.mtab).buf(rc.rt.cos).buf(rc.rt.sin).scalar(rc.rt.max_pos);
        f.done();
    }
    // completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n)
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;       // the smallest b with 4b+3 >= p0
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;   // the largest b with 4b+3 <= p0+n-1
    if (hi >= first) {
        metal::Launch bl("nqsi_append_blocks_kernel", (unsigned) (hi - first + 1), 1, 1, THREADS, 1, 1, 0, stream);
        bl.buf(raw).scalar((long) n).scalar((long) p0).scalar((int) pos_base).buf(gamma).scalar(epsilon)
          .buf(b.tail).buf(b.dead).buf(b.pooled).buf(b.block_pos)
          .scalar((long) first).scalar((long) hi)
          .scalar(rc.theta_scale).scalar(rc.freq_scale).scalar(rc.corr_low).scalar(rc.corr_high)
          .scalar(rc.ext_factor).scalar(rc.mscale)
          .buf(rc.mtab).buf(rc.rt.cos).buf(rc.rt.sin).scalar(rc.rt.max_pos);
        bl.done();
    }
    metal::Launch t("nqsi_append_tail_kernel", (unsigned) (R - 1), 1, 1, D, 1, 1, 0, stream);
    t.buf(raw).scalar((long) n).scalar((long) p0).buf(b.tail);
    t.done();
    cudaStreamSynchronize((cudaStream_t) stream);   // as above: order against the caller's next null-stream copy
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels

// src/kernels/metal/native_ple_postops.mm - the port of src/kernels/cuda/native_ple_postops.cu's host half.
// Same contract, same span validation and same launch order; the two rms norms go through the already-ported
// native_gr_rms_norm_weighted (its `weighted_rms_norm` kernel, the same arithmetic the CUDA file calls), and
// the rest are metal::Launch chains whose argument order IS each kernel's [[buffer(N)]] order.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}

}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,(size_t)D*4,4}, {hidden,(size_t)D*4,4}, {value,(size_t)N*4,4},
        {history,(size_t)HISTORY*D*4,4}, {w.norm_key,(size_t)D*4,4}, {w.norm_query,(size_t)D*4,4},
        {w.norm_conv,(size_t)D*4,4}, {w.conv1d_f16,(size_t)4*D*2,2}};
    const Span outputs[] = {{b.key,(size_t)D*4,4}, {b.query,(size_t)D*4,4}, {b.gate,(size_t)H*4,4},
        {b.gated,(size_t)D*4,4}, {b.normalized,(size_t)D*4,4}, {b.conv,(size_t)D*4,4}, {b.result,(size_t)D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = stream;
    {
        metal::Launch g("nple_gate_kernel", H, 1, 1, 512, 1, 1, 0, st);
        g.buf(b.key).buf(b.query).buf(b.gate).scalar(1.0f / std::sqrt(float(N)));
        g.done();
        metal::Launch bc("nple_broadcast_kernel", D / 256, 1, 1, 256, 1, 1, 0, st);
        bc.buf(value).buf(b.gate).buf(b.gated);
        bc.done();
        launch_check();
    }
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    {
        metal::Launch cr("nple_conv_residual_kernel", D / 256, 1, 1, 256, 1, 1, 0, st);
        cr.buf(history).buf(b.normalized).buf(w.conv1d_f16).buf(hidden).buf(b.gated).buf(b.conv).buf(b.result);
        cr.done();
        launch_check();
    }
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto st = stream;
    const unsigned rows = unsigned(T) * H;
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    {
        metal::Launch r1("nple_rms_rep_kernel", rows, 1, 1, 1024, 1, 1, 0, st);
        r1.buf(key).buf(w.norm_key).buf(key).scalar(NG_RMS_EPS);
        r1.done();
        metal::Launch r2("nple_rms_rep_kernel", rows, 1, 1, 1024, 1, 1, 0, st);
        r2.buf(hidden).buf(w.norm_query).buf(query_norm).scalar(NG_RMS_EPS);
        r2.done();
        metal::Launch g("nple_gate_kernel", rows, 1, 1, 512, 1, 1, 0, st);
        g.buf(key).buf(query_norm).buf(gate).scalar(1.0f / std::sqrt(float(N)));
        g.done();
        metal::Launch bb("nple_broadcast_batch_kernel", blocks, 1, 1, 256, 1, 1, 0, st);
        bb.buf(value).buf(gate).buf(gated).scalar(T);
        bb.done();
        metal::Launch r3("nple_rms_rep_kernel", rows, 1, 1, 1024, 1, 1, 0, st);
        r3.buf(gated).buf(w.norm_conv).buf(query_norm).scalar(NG_RMS_EPS);
        r3.done();
        metal::Launch cb("nple_conv_residual_batch_kernel", blocks, 1, 1, 256, 1, 1, 0, st);
        cb.buf(history).buf(query_norm).buf(w.conv1d_f16).buf(hidden).buf(gated).scalar(T);
        cb.done();
        metal::Launch hb("nple_history_batch_kernel", D / 256, 1, 1, 256, 1, 1, 0, st);
        hb.buf(history).buf(query_norm).scalar(T);
        hb.done();
    }
    launch_check();
}

}  // namespace strata::kernels

// src/kernels/metal/ple.mm - the port of src/kernels/cuda/ple.cu's host half (K20).  Same entry points, same
// scratch carving, same launch order and same export table; the launches are metal::Launch chains whose
// argument order IS each kernel's [[buffer(N)]] order, and the exports are the shim's stream-ordered D2D
// memcpyAsync (a blit on the stream, capture-aware - what cudaMemcpyAsync always was here).
//
// The NATIVE POSTOPS path is wired since native_ple_postops.mm's port landed (K18): ple_set_native_postops
// selects the pinned postprojection arithmetic exactly as on CUDA (the two leading gnorms are then the
// postops' own, and `key` exports the postops' normalized key).  The NATIVE BF16 value projection and the
// NATIVE KEY path were already ported (bf16_gemv's MMVF pair; native_mmvq.mm).  (The canonical Q2_0 key
// projection calls the real `s2_gemv_q8()` since that file's port landed - the private copy is gone.)
#include "strata/kernels/ple.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
bool native_bf16 = false;
bool native_postops = false;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_block: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

// ------------------------------------------------- the entry points (ple.cu's own order)

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    metal::Launch k("history_advance_kernel", (unsigned) ((NG_HC_DIM + THREADS - 1) / THREADS), 1, 1,
                    THREADS, 1, 1, 0, stream);
    k.buf(hist).buf(normalized);
    k.done();
    ck(cudaGetLastError(), "history advance launch");
}

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, then the BF16 embedding copy, each
    // 16-byte aligned because `d_scratch` is cast to `float*` and `d_act` to `uint8_t*` at those offsets.
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    cudaStream_t st = (cudaStream_t) stream;

    // One allocation for every intermediate, carved from the CALLER's workspace exactly as the CUDA file
    // does (its comment records the three-cudaMallocs-per-call capture error this shape replaces).
    //
    // FIVE SEPARATE hc_dim BUFFERS, deliberately - `key` and `normalized` are both needed at the end for the
    // oracle comparison, so aliasing one onto the other returns the WRONG `key` (the CUDA file's own note).
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) {
        std::fprintf(stderr, "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)\n");
        std::exit(1);
    }
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}
    };
    for (const auto& item : exports) {
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    }
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) ||
            (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}
        };
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb). The optional native projection follows the Q8_1+MMVQ bridge;
    // the default retains its canonical Q8_0 path.
    if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1,
                    d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, stream);
    }
    if (!native_postops) {
        // the native postops path runs its own key/query norms (native_gr_norm_weighted inside
        // native_ple_postops), so these two launches are the legacy path's alone
        metal::Launch g1("gnorm_kernel", (unsigned) hc, 1, 1, THREADS, 1, 1, 0, st);
        g1.buf(d_key).buf(w.norm_key).buf(d_key).scalar(n_embd).scalar(NG_RMS_EPS);
        g1.done();
        metal::Launch g2("gnorm_kernel", (unsigned) hc, 1, 1, THREADS, 1, 1, 0, st);
        g2.buf(hidden).buf(w.norm_query).buf(d_query).scalar(n_embd).scalar(NG_RMS_EPS);
        g2.done();
    }

    // The value projection's independent option leaves the nonlinear PLE operations unchanged.
    if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);
    } else {
        metal::Launch t("ple_to_bf16_kernel", (unsigned) ((n_embd + THREADS - 1) / THREADS), 1, 1,
                        THREADS, 1, 1, 0, st);
        t.buf(emb).buf(d_emb16).scalar(n_embd);
        t.done();
        metal::Launch v("bf16_gemv_kernel", (unsigned) ((n_embd + THREADS - 1) / THREADS), 1, 1,
                        THREADS, 1, 1, 0, st);
        v.buf(d_emb16).buf(w.value_bf16).buf(d_value).scalar(n_embd).scalar(n_embd);
        v.done();
    }

    const float* normalized_key = d_key;
    if (native_postops) {
        // The key norm has a distinct destination; temporary query storage can
        // be reused for normalized gated values after the gate consumes it.
        NativePlePostopsBuffers buffers{d_query, d_norm, d_gate, d_gated, d_norm, d_conv, out.result};
        native_ple_postops(d_key, hidden, d_value, hist_rows, w, buffers, stream);
        normalized_key = d_query;
    } else {
        metal::Launch gt("gate_kernel", (unsigned) hc, 1, 1, THREADS, 1, 1, 0, st);
        gt.buf(d_key).buf(d_query).buf(d_gate).scalar(n_embd)
          .scalar(1.0f / std::sqrt((float) n_embd));
        gt.done();
        metal::Launch bc("bcast_kernel", (unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1,
                         THREADS, 1, 1, 0, st);
        bc.buf(d_value).buf(d_gate).buf(d_gated).scalar(n_embd).scalar(hc);
        bc.done();
        metal::Launch g3("gnorm_kernel", (unsigned) hc, 1, 1, THREADS, 1, 1, 0, st);
        g3.buf(d_gated).buf(w.norm_conv).buf(d_norm).scalar(n_embd).scalar(NG_RMS_EPS);
        g3.done();
        metal::Launch cv("conv_kernel", (unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1,
                         THREADS, 1, 1, 0, st);
        cv.buf(hist_rows).buf(d_norm).buf(w.conv1d_f16).buf(d_conv).scalar(hc_dim)
          .scalar((int) PLE_CONV_KERNEL).scalar((int) NGRAM_SIZE).scalar((int) NG_HIST);
        cv.done();
        metal::Launch a3("add3_kernel", (unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1,
                         THREADS, 1, 1, 0, st);
        a3.buf(hidden).buf(d_gated).buf(d_conv).buf(out.result).scalar(hc_dim);
        a3.done();
    }

    // ---- the intermediates the oracle comparison needs, exported with the shim's stream-ordered D2D copy
    //      (a blit encoder on the stream; inside a capture the tape records it and it runs at replay).
    //      `key` is the NORMALISED key because the source's `cb(key, ...)` capture is after `gnorm` - in the
    //      native-postops path that is the postops' normalized key (d_query).
    if (out.key) ck(cudaMemcpyAsync(out.key, normalized_key, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "key");
    if (out.value)
        ck(cudaMemcpyAsync(out.value, d_value, n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st), "value");
    if (out.gate) ck(cudaMemcpyAsync(out.gate, d_gate, hc * sizeof(float), cudaMemcpyDeviceToDevice, st), "gate");
    if (out.gated)
        ck(cudaMemcpyAsync(out.gated, d_gated, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "gated");
    if (out.normalized)
        ck(cudaMemcpyAsync(out.normalized, d_norm, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "norm");
    if (out.conv)
        ck(cudaMemcpyAsync(out.conv, d_conv, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "conv");

    ck(cudaGetLastError(), "launch");
    // **NO `cudaStreamSynchronize` HERE** - the CUDA file's own rule: inside a capture it is an error, and a
    // caller that wants the result immediately synchronises itself.
}

}  // namespace strata::kernels

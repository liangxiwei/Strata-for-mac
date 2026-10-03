// src/kernels/metal/qsa_select.mm - the port of src/kernels/cuda/qsa_select.cu's host half (K15).
//
// DISPATCH DIFFERENCES, both the CUDA file's own fallbacks:
//   * qsa_block_scores_tc always returns false: the 3xTF32 mma.sync kernel is sm_80-only and the gfx12 WMMA
//     one is HIP-only; Metal has neither instruction, so the warp kernel scores - the path a pre-sm_80
//     CUDA card or a non-gfx12 AMD one takes (and the tc kernel's own header says it is not bitwise).
//   * qsa_block_topk always launches the ref kernel: the 1024-thread register variant's per-warp histograms
//     alone are the whole 32,768 B threadgroup budget (qsa_select.metal's header has the numbers), and the
//     CUDA file documents the two as producing the same ids (STRATA_TOPK_OLD is therefore a no-op here).
#include "strata/kernels/qsa_select.hpp"

#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int MQ = 8;

void launch_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                   int64_t nq, int64_t max_blocks, float* scores, void* stream, int64_t reach) {
    metal::Launch k("block_scores_kernel", (unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                    (unsigned) nq, 1, SCORE_WARPS * 32, 1, 1, 0, stream);
    k.buf(pooled).buf(dead).buf(q_idx).buf(steps).scalar((long) max_blocks).buf(scores);
    k.done();
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        metal::Launch k("block_scores_multi_kernel", 256, 1, 1, SCORE_WARPS * 32, 1, 1, 0, stream);
        k.buf(pooled).buf(dead).buf(q_idx).buf(steps).scalar((int) nq).scalar((long) max_blocks).buf(scores);
        k.done();
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    launch_scores(pooled, dead, q_idx, steps, nq, max_blocks, scores, stream, reach);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    // Metal has no mma.sync (sm_80) and no gfx12 WMMA: the tensor-core scorer does not exist here, so this
    // refuses every pool and the caller keeps the warp kernel - the same contract a pre-sm_80 CUDA card and
    // a non-gfx12 AMD one get (the kernel is explicitly not bitwise with the warp one either way).
    (void) pooled; (void) dead; (void) q_idx; (void) steps; (void) nq; (void) max_blocks; (void) s;
    (void) scores; (void) stream; (void) active_blocks;
    return false;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    metal::Launch k("block_topk_kernel", (unsigned) nq, 1, 1, 256, 1, 1, 0, stream);
    k.buf(scores).buf(steps).scalar((long) max_blocks).scalar((long) cap).buf(ids);
    k.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // The register kernel (keys of up to 33 blocks per thread, 1024 threads) does not fit this GPU's fixed
    // 32,768 B threadgroup budget - its histograms alone are exactly that - so the ref kernel's selection (keys
    // from memory on every radix pass) always runs: block_topk_scan_kernel computes the same ids with parallel
    // scans; STRATA_METAL_TOPK_SCAN=0 keeps block_topk_kernel itself.
    (void) active_blocks;
    static const bool scan = [] {
        const char* v = std::getenv("STRATA_METAL_TOPK_SCAN");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!scan) { qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream); return; }
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    metal::Launch k("block_topk_scan_kernel", (unsigned) nq, 1, 1, 256, 1, 1, 0, stream);
    k.buf(scores).buf(steps).scalar((long) max_blocks).scalar((long) cap).buf(ids);
    k.done();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels

// src/kernels/metal/elementwise.mm - the port of src/kernels/cuda/elementwise.cu's launchers (K1,
// docs/PORT_METAL/STATUS.md).  Same header contract (include/strata/kernels/elementwise.hpp), same grids and
// blocks as the CUDA launchers; each `<<<grid, block, shared, stream>>>(args...)` is a strata::metal::Launch.
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr unsigned THREADS = 256;

inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    metal::Launch k("embedding_gather_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(codes).buf(scales).buf(offsets);          // null offsets bind null: the kernel tests it
    k.scalar(n).scalar(code_bits).scalar(code_bias).scalar(group_elems).buf(out);
    k.done();
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    metal::Launch k("gdn_gate_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(alpha).buf(dt).buf(ssm_a).buf(gate).scalar(h_v);
    k.done();
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    metal::Launch k("scale_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(x).scalar(n).scalar(s);
    k.done();
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("add_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar(n);
    k.done();
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("to_f16_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(x).buf(y).scalar(n);
    k.done();
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("to_bf16_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(x).buf(y).scalar(n);
    k.done();
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    metal::Launch k("silu_kernel", grid_for(n), 1, 1, THREADS, 1, 1, 0, stream);
    k.buf(x).scalar(n);
    k.done();
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    metal::Launch k("doorbell_ring_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    k.buf(d_seq);
    k.done();
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    metal::Launch k("doorbell_wait_kernel", 1, 1, 1, 1, 1, 1, 0, stream);
    k.buf(d_flag).buf(d_seq);
    k.done();
    check_launch("doorbell_wait");
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    metal::Launch k("copy_from_mapped_kernel", (unsigned) blocks, 1, 1, 256, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar(n4);
    k.done();
    check_launch("copy_from_mapped");
}

void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t row4 = width / 4;
    metal::Launch k("copy_rows_from_mapped_kernel", (unsigned) rows, 1, 1, 128, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar(row4).buf(hit_rows).buf(count);
    k.done();
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    const int ni = (int) n, ki = (int) k;
    metal::Launch kp("doorbell_publish_kernel", 1, 1, 1, 1024, 1, 1, 0, stream);
    kp.buf(x).buf(ids).buf(weights).scalar(ni).scalar(ki)
       .buf(x_out).buf(ids_out).buf(weights_out).buf(d_seq);
    kp.done();
    check_launch("doorbell_publish");
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int ni = (int) n;
    metal::Launch k("copy_i32_from_mapped_kernel", 1, 1, 1, 128, 1, 1, 0, stream);
    k.buf(dst).buf(src).scalar(ni);
    k.done();
    check_launch("copy_i32_from_mapped");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    metal::Launch k("rms_norm_weighted_kernel", grid, 1, 1, warps_per_block * 32, 1, 1, 0, stream);
    k.buf(x).buf(w).scalar(rows).scalar(cols).scalar(eps);   // null w binds null
    k.done();
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels

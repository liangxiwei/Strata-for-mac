// src/kernels/metal/s_gemv.mm - the port of src/kernels/cuda/s_gemv.cu's launchers (K12).  Same entry
// points and checks as the CUDA file; every `<<<grid, block, shared, stream>>>` is a metal::Launch chain
// whose argument order IS the kernel's [[buffer(N)]] order (the port's rule: buffers first, then scalars,
// index = position - see PROGRESS.md round 9).
#include "strata/kernels/s_gemv.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void finish(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream != nullptr) return;
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", who, cudaGetErrorString(s));
        std::exit(1);
    }
}

bool q8k_form_ok(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of group_elems %d\n", who, (long long) n_in,
                     form.group_elems);
        return false;
    }
    if (n_in % 256 != 0) {
        // `quantize_q8_K` requires this too, and a partial block would read past the end of the activation.
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of the Q8_K block %d\n", who, (long long) n_in,
                     256);
        return false;
    }
    return true;
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n", (long long) n_in,
                     form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    const char* kernel = nullptr;
    switch (form.code_bits) {
        case 2: kernel = "s_gemv_kernel_s2"; break;
        case 4: kernel = "s_gemv_kernel_s4"; break;
        case 8: kernel = "s_gemv_kernel_s8"; break;
        default:
            std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_out + 127) / 128);
    metal::Launch k(kernel, grid, 1, 1, 128, 1, 1, 0, nullptr);
    k.buf(x).buf(codes).buf(scales).buf(offset).buf(y)
     .scalar(n_in)
     .scalar(n_out)
     .scalar(form.code_bias)
     .scalar((int) form.codebook)
     .scalar(form.group_elems)
     .scalar(form.has_offset ? 1 : 0);
    k.done();
    finish("s_gemv", nullptr);
}

namespace {

// One BLOCK per output row, `threads_per_row` threads splitting it - parallelism n_out * threads_per_row,
// deterministic shared-memory tree reduction.  See the CUDA file for the group-as-shift invariant: every
// group size the format defines is a power of two, so the divisor is passed as its logarithm and the host
// refuses anything else rather than compute a wrong index quietly.
void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales,
                       const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form,
                       int threads_per_row, void* stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024) {
        std::fprintf(stderr, "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP: four consecutive elements share a scale and the kernel reads
    // exactly one.  Every group size this format defines is 16, 32 or 64, so this cannot fire on a real pack.
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const char* kernel = nullptr;
    switch (form.code_bits) {
        case 2: kernel = "s_gemv_split_kernel_s2"; break;
        case 4: kernel = "s_gemv_split_kernel_s4"; break;
        case 8: kernel = "s_gemv_split_kernel_s8"; break;
        default:
            std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    metal::Launch k(kernel, (unsigned) n_out, 1, 1, (unsigned) threads_per_row, 1, 1,
                    (size_t) threads_per_row * sizeof(float), stream);
    k.buf(x).buf(codes).buf(scales).buf(offset).buf(y)
     .scalar(n_in)
     .scalar(n_out)
     .scalar(form.code_bias)
     .scalar((int) form.codebook)
     .scalar(form.group_elems)
     .scalar(group_shift)
     .scalar(form.has_offset ? 1 : 0)
     .scalar(threads_per_row);
    k.done();
    if (sync) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "s_gemv_split: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, nullptr, true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, stream, false);
}

// ---- the Q8_K entry points -----------------------------------------------------------------------------

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k")) std::exit(1);
    const char* kernel = nullptr;
    switch (form.code_bits) {
        case 4: kernel = "s_gemv_q8k_kernel_s4"; break;
        case 8: kernel = "s_gemv_q8k_kernel_s8"; break;
        default:
            // 2-bit S2 never has a Q8_K activation: its `vec_dot_type` is Q8_0.  Refusing is better than
            // running a kernel that would be numerically wrong in a way the caller cannot see.
            std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract "
                                 "(S2 uses Q8_0; see docs/activation-contract.md)\n",
                         form.code_bits);
            std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_out + 127) / 128);
    metal::Launch k(kernel, grid, 1, 1, 128, 1, 1, 0, stream);
    k.buf(x_q8k).buf(codes).buf(scales).buf(offset).buf(y)
     .scalar(n_in)
     .scalar(n_out)
     .scalar(form.code_bias)
     .scalar((int) form.codebook)
     .scalar(form.group_elems)
     .scalar(form.has_offset ? 1 : 0);
    k.done();
    finish("s_gemv_q8k", stream);
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k_split")) std::exit(1);
    // THE GROUP SIZE IS PASSED AS ITS LOGARITHM: a non-power-of-two group would make the shift a WRONG
    // INDEX rather than a slow one, so it is refused here.
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    if ((1 << group_shift) != form.group_elems) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a power of two\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP.  Every group size this format defines is 16, 32 or 64, so
    // this cannot fire on a real pack - it is here because a future one could.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    const char* kernel = nullptr;
    switch (form.code_bits) {
        case 4: kernel = "s_gemv_q8_split_kernel_s4_q8k"; break;
        case 8: kernel = "s_gemv_q8_split_kernel_s8_q8k"; break;
        default:
            std::fprintf(stderr, "s_gemv_q8k_split: code_bits %d has no Q8_K contract\n", form.code_bits);
            std::exit(1);
    }
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
    metal::Launch k(kernel, grid, 1, 1, threads, 1, 1, 0, stream);
    k.buf(x_q8k).buf(codes).buf(scales).buf(offset).buf(y)
     .scalar(n_in)
     .scalar(n_out)
     .scalar(form.code_bias)
     .scalar((int) form.codebook)
     .scalar(group_shift)
     .scalar(form.has_offset ? 1 : 0)
     .scalar((unsigned) threads);   // block: the MSL kernel derives warps_per_block from it
    k.done();
    finish("s_gemv_q8k_split", stream);
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION IS `block_q8_0`, 32 elements per block, so `n_in` must be a multiple of 32 - which is
    // a WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist: `ffn_down_shexp`
    // has n_in = 640, a multiple of 32 that is NOT a multiple of 256.
    if (n_in % 32 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: n_in %lld is not a multiple of %d\n", (long long) n_in, 32);
        std::exit(1);
    }
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    // SIXTEEN, NOT FOUR: a lane-iteration takes QE = 16 consecutive elements under ONE scale (see the
    // kernel), so a group of 4 or 8 would read the wrong scale for most of them.  The Q8_K launcher
    // already said 16.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;

    const char* kernel = nullptr;
    switch (form.code_bits) {
        case 4: kernel = "s_gemv_q8_split_kernel_s4_q80"; break;
        case 8: kernel = "s_gemv_q8_split_kernel_s8_q80"; break;
        default:
            std::fprintf(stderr, "s_gemv_q8_0_split: code_bits %d has no Q8_0 contract\n", form.code_bits);
            std::exit(1);
    }
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
    metal::Launch k(kernel, grid, 1, 1, threads, 1, 1, 0, stream);
    k.buf(x_q8_0).buf(codes).buf(scales).buf(offset).buf(y)
     .scalar(n_in)
     .scalar(n_out)
     .scalar(form.code_bias)
     .scalar((int) form.codebook)
     .scalar(group_shift)
     .scalar(form.has_offset ? 1 : 0)
     .scalar((unsigned) threads);
    k.done();
    finish("s_gemv_q8_0_split", stream);
}

}  // namespace strata::kernels

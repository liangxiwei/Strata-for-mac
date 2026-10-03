// include/strata/platform/metal_launch.hpp - how a ported launcher dispatches a kernel on the Metal backend.
//
// A kernel's .cu launcher becomes a .mm launcher with the SAME header contract (include/strata/kernels/*.hpp);
// inside, every `k<<<grid, block, shared, stream>>>(args...)` becomes
//
//     strata::metal::Launch k("k_kernel", grid, block, shared, stream);
//     k.buf(a, bytes_a); ... k.scalar(x); ...
//     k.done();
//
// `buf` takes any pointer the runtime knows - a cudaMalloc'd MTLBuffer or an interior pointer into one (the
// registry resolves buffer + offset), or mapped host memory (a no-copy MTLBuffer over the host pages).  `scalar`
// passes a small value by bytes.  On a capturing stream the builder records the entry on the tape instead of
// encoding; cudaGraphLaunch replays it (docs/PORT_METAL/PLAN.md - "the tape").
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::metal {

class Launch {
public:
    Launch(const char* kernel, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
           size_t shared_bytes, void* stream);
    ~Launch();
    /// A pointer argument - a cudaMalloc'd buffer or an interior pointer into one (the registry resolves
    /// buffer + offset; the kernel bounds itself, exactly as a CUDA launch passes bare pointers).
    Launch& buf(const void* ptr);
    /// A small by-value argument (ints, floats; copied into the entry, so a capture bakes it as CUDA does).
    Launch& scalar(const void* p, size_t bytes);
    /// The same, spelled the way a launch site wants it: `k.scalar(n)`.
    template<class T> Launch& scalar(const T& v) { return scalar((const void*) &v, sizeof(T)); }
    void done();

private:
    struct Impl;
    Impl* impl_;
    Launch(const Launch&) = delete;
    Launch& operator=(const Launch&) = delete;
};

/// The kernel library's state: "ok", or why it could not load (device_code_error()'s Metal branch).
const char* code_state();

/// The DeviceArena's NaN fill (src/core/device.cu's poison_kernel), chunked the same way.
void poison(float* p, uint64_t n_floats);

// ---- the MPS seam (M4, the prefill GEMM) ------------------------------------------------
// MetalPerformanceShaders encodes its own work; these three hand a ported .mm enough of the runtime to do
// that ON THE STREAM: the stream's open command buffer (so MPS work serializes with Launch's compute work
// on the same serial queue), the pointer resolver Launch::buf() itself uses, and the device MPS objects
// are created against.  All opaque void*: this header is included by plain C++ translations units too
// (the engine's .cpp files), which must not see Metal headers.

/// The stream's CURRENT command buffer, RETAINED (as void*; cast to id<MTLCommandBuffer> in a .mm) - MPS
/// `encodeToCommandBuffer:` onto it is stream-ordered behind everything Launch encoded before.  The caller
/// releases it (the runtime keeps its own retain in the Stream; a fresh call returns the same buffer).
void* mtl_command_buffer(void* stream);

/// Resolve a pointer the runtime knows (a cudaMalloc'd buffer or an interior pointer into one) to its
/// MTLBuffer (RETAINED, returned as void*; cast to id<MTLBuffer> in a .mm) and byte offset - the exact
/// pair `Launch::buf` binds, handed to MPSMatrix's `initWithBuffer:offset:descriptor:`.  False when the
/// pointer is not a runtime buffer (the same refusal buf() would make).  The caller releases the buffer.
bool resolve_buffer(const void* ptr, void** buffer, uint64_t* offset);

/// The process's one Metal device (borrowed, NOT retained - it lives for the process), as void*.
void* mtl_device();

}  // namespace strata::metal

// src/kernels/metal/poison.metal - DeviceArena's NaN fill (src/core/device.cu's poison_kernel).
// The NaN pattern, not zero: zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight, and NaNs are not.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void poison_kernel(device float* p [[buffer(0)]],
                          constant ulong& n_floats [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid < n_floats) p[tid] = as_type<float>(0x7fc00000u);
}

// cudaMemsetAsync on the stream (a compute fill; blit has none)
kernel void memset8_kernel(device uchar* p [[buffer(0)]],
                           constant const uchar& v [[buffer(1)]],
                           constant const ulong& n [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < n) p[i] = v;
}

// TEMP M1 probe: scale with gdn_gate's argument SHAPE (4 buffers + 1 scalar)
kernel void scale4_kernel(device float* x [[buffer(0)]],
                          constant const float* d1 [[buffer(1)]],
                          constant const float* d2 [[buffer(2)]],
                          constant const float* d3 [[buffer(3)]],
                          constant const ulong& n [[buffer(4)]],
                          constant const float& s [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = (x[i] + d1[0] + d2[0] + d3[0]) * s;
}

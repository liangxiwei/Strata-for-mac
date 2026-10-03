#include <metal_stdlib>
using namespace metal;

// Short stream copies stay in the serial compute pass. These kernels move bits only.
kernel void runtime_copy16(device uint4* dst [[buffer(0)]],
                           device const uint4* src [[buffer(1)]],
                           constant ulong& n [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

kernel void runtime_copy1(device uchar* dst [[buffer(0)]],
                          device const uchar* src [[buffer(1)]],
                          constant ulong& n [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

kernel void runtime_copy2d16(device uint4* dst [[buffer(0)]],
                             device const uint4* src [[buffer(1)]],
                             constant ulong& dpitch [[buffer(2)]],
                             constant ulong& spitch [[buffer(3)]],
                             constant ulong& width [[buffer(4)]],
                             uint2 i [[thread_position_in_grid]]) {
    if (i.x < width) dst[ulong(i.y) * dpitch + i.x] = src[ulong(i.y) * spitch + i.x];
}

kernel void runtime_copy2d1(device uchar* dst [[buffer(0)]],
                            device const uchar* src [[buffer(1)]],
                            constant ulong& dpitch [[buffer(2)]],
                            constant ulong& spitch [[buffer(3)]],
                            constant ulong& width [[buffer(4)]],
                            uint2 i [[thread_position_in_grid]]) {
    if (i.x < width) dst[ulong(i.y) * dpitch + i.x] = src[ulong(i.y) * spitch + i.x];
}

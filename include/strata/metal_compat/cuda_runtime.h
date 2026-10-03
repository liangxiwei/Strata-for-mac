// include/strata/metal_compat/cuda_runtime.h - the Metal backend's CUDA-runtime shim (docs/PORT_METAL/).
//
// The engine's host code is written against the CUDA runtime's API surface; the HIP backend plays the same
// trick with a force-included cuda_runtime.h of its own. Here the names map onto the Metal runtime in
// src/platform/metal_runtime.mm: allocations are shared-storage MTLBuffers (unified memory - every "device"
// pointer is also a host pointer, and the copies are memcpys), a stream is a serial chain of command buffers
// on one queue, capture/replay is a tape of launches that cudaGraphLaunch re-encodes.
//
// Deliberate simplifications, all measured-or-marked in docs/PORT_METAL/PROGRESS.md when they bite:
//   * cudaEventElapsedTime reads host timestamps at signal time (a log-scale approximation, not a profiler);
//   * cudaStreamWaitEvent waits the event on the host thread (rare path - cross-stream ordering);
//   * one device: the Mac's GPU. Ordinals beyond 0 are refused.
#pragma once
#ifndef STRATA_METAL_COMPAT_CUDA_RUNTIME_H
#define STRATA_METAL_COMPAT_CUDA_RUNTIME_H

// A CUDA spelling that keeps the HIP/CUDA graph-diagnostic paths compiling; the Metal tape has no node
// parameters to hand back, and verify.cpp only prints what it can get.
#define CUDART_VERSION 0

#include <cstddef>
#include <cstdint>

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNotReady = 600,
    cudaErrorUnknown = 999,
    cudaErrorStreamCaptureUnsupported = 9001,     // cudaStreamCaptureUnsupported on CUDA's enum
    cudaErrorStreamCaptureInvalidated = 9002
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4
} cudaMemcpyKind;

enum { cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2 };
enum { cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2 };
enum { cudaEventDisableTiming = 2, cudaEventBlockingSync = 4 };
enum { cudaStreamNonBlocking = 1 };
enum { cudaDeviceScheduleSpin = 1, cudaDeviceScheduleAuto = 0, cudaDeviceMapHost = 2 };
enum { cudaGraphNodeTypeKernel = 0, cudaGraphNodeTypeMemcpy = 1, cudaGraphNodeTypeMemset = 2, cudaGraphNodeTypeEmpty = 9 };
typedef int cudaGraphNodeType;   // the enum-typed spelling verify.cpp's node listing declares a variable of
enum { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
enum { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
enum { cudaDevAttrMaxSharedMemoryPerBlock = 8, cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
       cudaDevAttrMultiProcessorCount = 16, cudaDevAttrClockRate = 13,
       cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76 };
enum { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
enum { cudaEventRecordDefault = 0 };

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int multiProcessorCount;
    int warpSize;
    int clockRate;
    int major, minor;
    size_t totalConstMem;
    int maxThreadsPerBlock;
    char gcnArchName[64];        // read only on the HIP backend; stays empty here
};

struct cudaFuncAttributes {
    int maxThreadsPerBlock;
    size_t sharedSizeBytes;
    int numRegs;
};

typedef struct CUFunc_st* cudaFunction_t;          // graph diagnostics hand these around opaquely

struct cudaKernelNodeParams {
    const void* func;
    int gridDim[3], blockDim[3];
    size_t sharedMemBytes;
    void** kernelParams;
};

typedef struct { unsigned x, y, z; } uint3;
typedef struct dim3_s {
    unsigned x, y, z;
    dim3_s(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
} dim3;

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaGraphNode_t;
typedef int cudaStreamCaptureStatus;

#ifdef __cplusplus
extern "C" {
#endif

// ---- device ----
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int ordinal);
cudaError_t cudaGetDevice(int* ordinal);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int ordinal);
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int ordinal);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaInitDevice(int ordinal, unsigned flags, int flags2);
cudaError_t cudaDriverGetVersion(int* version);
cudaError_t cudaRuntimeGetVersion(int* version);
const char* cudaGetErrorString(cudaError_t e);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);

// ---- memory ----
cudaError_t cudaMalloc(void** ptr, size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaMallocHost(void** ptr, size_t bytes);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostAlloc(void** ptr, size_t bytes, unsigned flags);
cudaError_t cudaHostRegister(void* ptr, size_t bytes, unsigned flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** device_ptr, const void* host_ptr, unsigned flags);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* ptr, int value, size_t bytes);
cudaError_t cudaMemsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t bytes, size_t offset,
                               cudaMemcpyKind kind);     // s2_gemv_fast's tables - refused until ported

// ---- functions ----
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetAttribute(const void* func, int attr, int value);
cudaError_t cudaFuncGetName(const char** name, const void* func);

// ---- streams & events ----
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData);

// ---- graphs ----
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, void*, void*, unsigned long long);
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* count);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, int* type);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* params);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);

#ifdef __cplusplus
}
// CUDA's allocators are templates over the pointer type (`float* d; cudaMalloc(&d, n)`); these thin wrappers
// keep that call shape working against the void** bases above.  A plain void** call still picks the base.
template<class T> static inline cudaError_t cudaMalloc(T** ptr, size_t bytes) {
    void* p = nullptr;
    const cudaError_t e = ::cudaMalloc(&p, bytes);
    *ptr = (T*) p;
    return e;
}
template<class T> static inline cudaError_t cudaMallocHost(T** ptr, size_t bytes) {
    void* p = nullptr;
    const cudaError_t e = ::cudaMallocHost(&p, bytes);
    *ptr = (T*) p;
    return e;
}
template<class T> static inline cudaError_t cudaHostAlloc(T** ptr, size_t bytes, unsigned flags) {
    void* p = nullptr;
    const cudaError_t e = ::cudaHostAlloc(&p, bytes, flags);
    *ptr = (T*) p;
    return e;
}
template<class T> static inline cudaError_t cudaHostGetDevicePointer(T** device_ptr, const void* host_ptr,
                                                                    unsigned flags) {
    void* p = nullptr;
    const cudaError_t e = ::cudaHostGetDevicePointer(&p, host_ptr, flags);
    *device_ptr = (T*) p;
    return e;
}
// CUDA 12's shorter spelling of graph instantiation (the two middle pointers were removed upstream)
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                               unsigned long long flags) {
    return ::cudaGraphInstantiate(exec, graph, nullptr, nullptr, flags);
}
// CUDA's one-argument spelling (the bench blocks in the parity tests use it): record on the default stream
static inline cudaError_t cudaEventRecord(cudaEvent_t event) { return ::cudaEventRecord(event, nullptr); }
// CUDA's four-argument spelling of the async copy (the default stream): generate.cpp's PCIe H2D probe
static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    return ::cudaMemcpyAsync(dst, src, bytes, kind, nullptr);
}
#endif    // __cplusplus

#endif    // STRATA_METAL_COMPAT_CUDA_RUNTIME_H

# Opt-in Metal configuration (Apple-Silicon Macs). The engine's CUDA-shaped host code compiles against
# include/strata/metal_compat/cuda_runtime.h - the same trick hip_backend.cmake plays with HIP, at the API
# level - and the kernels are ported file by file into src/kernels/metal/ (MSL bodies + launcher .mms behind the
# same header contracts). docs/PORT_METAL/ is the plan and the live status of that port.
if(NOT APPLE)
  message(FATAL_ERROR "STRATA_ENABLE_METAL is for Apple-Silicon Macs; on a PC use STRATA_ENABLE_CUDA or STRATA_ENABLE_HIP")
endif()
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_ENABLE_METAL is a separate backend; enable only one")
endif()

enable_language(OBJCXX)
find_library(MTL_METAL Metal REQUIRED)
find_library(MTL_FOUNDATION Foundation REQUIRED)

set(STRATA_METAL_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/metal_compat")
add_library(strata_metal_runtime INTERFACE)
target_include_directories(strata_metal_runtime BEFORE INTERFACE
  "${STRATA_METAL_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_metal_runtime INTERFACE STRATA_USE_METAL=1 "STRATA_METAL_BACKEND=1")
target_link_libraries(strata_metal_runtime INTERFACE ${MTL_METAL} ${MTL_FOUNDATION})
# The shim renames the CUDA runtime to Metal, force-included into every host source - the file's own
# `#include <cuda_runtime.h>` also resolves to ours because the compat dir comes first on the include path.
file(TO_CMAKE_PATH "${STRATA_METAL_COMPAT_INCLUDE_DIR}/cuda_runtime.h" _strata_metal_force)
target_compile_options(strata_metal_runtime INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-include>" "$<$<COMPILE_LANGUAGE:CXX>:${_strata_metal_force}>"
  "$<$<COMPILE_LANGUAGE:OBJCXX>:-include>" "$<$<COMPILE_LANGUAGE:OBJCXX>:${_strata_metal_force}>")

# ---- the MSL kernels: every src/kernels/metal/*.metal into one metallib, embedded as bytes ----
file(GLOB _strata_metal_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal/*.metal")
file(GLOB _strata_metal_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal/*.metalh")
set(STRATA_METALLIB "${CMAKE_BINARY_DIR}/strata.metallib")
set(_strata_metal_gen "${CMAKE_BINARY_DIR}/generated/strata_metallib.mm")
add_custom_command(
  OUTPUT "${STRATA_METALLIB}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/generated"
  # no fast math, no cross-statement FP contraction: xcrun metal's defaults are lossy (fast reciprocal
  # division and contraction both flipped parity bits - measured, docs/PORT_METAL/PROGRESS.md).  The price
  # is that the plain exp/log/rsqrt spellings disappear; the port's .metal files call metal::precise::
  # explicitly for exactly this reason.
  COMMAND xcrun -sdk macosx metal -fno-fast-math -ffp-contract=off ${_strata_metal_sources} -o "${STRATA_METALLIB}"
  DEPENDS ${_strata_metal_sources} ${_strata_metal_headers}
  COMMENT "Strata Metal: compiling ${_strata_metal_sources}")
# the runtime loads the library from these bytes; no loose file to ship or find
add_custom_command(
  OUTPUT "${_strata_metal_gen}"
  COMMAND ${CMAKE_COMMAND} -DIN="${STRATA_METALLIB}" -DOUT="${_strata_metal_gen}" -DNAME=strata_metallib_bytes
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
  DEPENDS "${STRATA_METALLIB}"
  COMMENT "Strata Metal: embedding strata.metallib")
add_library(strata_metal_kernels STATIC "${_strata_metal_gen}" src/platform/metal_runtime.mm)
target_include_directories(strata_metal_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(strata_metal_kernels PUBLIC strata_metal_runtime Threads::Threads)

# The kernel files whose port is complete (a .metal + a .mm behind the same header contract). A parity test
# is registered for the Metal build only when its file is listed here AND its .mm exists (CMakeLists' EXISTS
# gate) - docs/PORT_METAL/STATUS.md is the same list in human form. Keep the two in step. The stems after
# native_gr are pre-registered for the parallel porting waves and flip on as their files land.
set(STRATA_METAL_PORTED "elementwise;dequant_s2;s2_gemv;quantize_act;router_top10;rope;kv_q8;gr;bf16_gemv;native_gr;cvec;fused_gr;gdn;fused_gdn;sampler;s_gemv;s2_gemv_quads;s2_gemv_fast;shared_expert;iq_kernels;qsa;qsa_select;qsa_decode_attn;qsa_prompt_attn;kv_q4;kv_stream;s2_expert_grouped;ple;s2_gemv_q8;native_mmvq;native_qsa;native_qsa_indexer;native_qsa_score;native_gdn;native_gdn_preprocess;native_router;native_flash_attn;native_ple_postops;native_moe;verify_kernels" CACHE INTERNAL "kernel file stems ported to Metal")

message(STATUS "Strata: Metal enabled (${_strata_metal_sources} MSL files so far)")

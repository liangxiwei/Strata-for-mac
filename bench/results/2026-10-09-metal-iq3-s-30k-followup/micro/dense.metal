#include "prefill.metal"
PFL_MATRIX2_KERNEL(pq_f16_64x128_24, half, false, 64, 128, 2, 4)
PFL_MATRIX2_KERNEL(pq_f16_128x64_42, half, false, 128, 64, 4, 2)
PFL_MATRIX2_KERNEL(pq_f16_128x128_44, half, false, 128, 128, 4, 4)
PFL_MATRIX2_KERNEL(pq_bf16_64x128_28, float, true, 64, 128, 2, 8)
PFL_MATRIX2_KERNEL(pq_bf16_128x64_44, float, true, 128, 64, 4, 4)

#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."
mkdir -p /tmp/strata-mtp-kernel-opt
for mode in 0 1 2; do
  perm=0
  if [ "$mode" = 2 ]; then perm=1; fi
  xcrun -sdk macosx metal -fno-fast-math -ffp-contract=off -DCANDIDATE_FLOAT_DP4A=$mode -DCANDIDATE_VECTOR_PERM=$perm -I src/kernels/metal -c bench/results/2026-10-11-metal-cuda-emulation/iq-intrinsics-candidates.metal -o /tmp/strata-mtp-kernel-opt/iq-intrinsic-$mode.air
  xcrun -sdk macosx metallib /tmp/strata-mtp-kernel-opt/iq-intrinsic-$mode.air -o /tmp/strata-mtp-kernel-opt/iq-intrinsic-$mode.metallib
 done
clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal bench/results/2026-10-11-metal-cuda-emulation/iq-intrinsics-micro.mm -o /tmp/strata-mtp-kernel-opt/iq-intrinsics-micro
xcrun -sdk macosx metal -fno-fast-math -ffp-contract=off -I src/kernels/metal -c bench/results/2026-10-11-metal-mtp-kernel-optimization/q6-candidates.metal -o /tmp/strata-mtp-kernel-opt/q6-single.air
xcrun -sdk macosx metallib /tmp/strata-mtp-kernel-opt/q6-single.air -o /tmp/strata-mtp-kernel-opt/q6-single.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal bench/results/2026-10-11-metal-cuda-emulation/q6-single-micro.mm -o /tmp/strata-mtp-kernel-opt/q6-single-micro
/tmp/strata-mtp-kernel-opt/iq-intrinsics-micro /tmp/strata-mtp-kernel-opt/iq-intrinsic-0.metallib /tmp/strata-mtp-kernel-opt/iq-intrinsic-1.metallib /tmp/strata-mtp-kernel-opt/iq-intrinsic-2.metallib > bench/results/2026-10-11-metal-cuda-emulation/iq-parity.csv 2> bench/results/2026-10-11-metal-cuda-emulation/iq-parity.log
/tmp/strata-mtp-kernel-opt/iq-intrinsics-micro /tmp/strata-mtp-kernel-opt/iq-intrinsic-0.metallib /tmp/strata-mtp-kernel-opt/iq-intrinsic-1.metallib /tmp/strata-mtp-kernel-opt/iq-intrinsic-2.metallib bench > bench/results/2026-10-11-metal-cuda-emulation/iq-bench.csv 2> bench/results/2026-10-11-metal-cuda-emulation/iq-bench.log
/tmp/strata-mtp-kernel-opt/q6-single-micro /tmp/strata-mtp-kernel-opt/q6-single.metallib > bench/results/2026-10-11-metal-cuda-emulation/q6-single-parity.csv 2> bench/results/2026-10-11-metal-cuda-emulation/q6-single-parity.log
/tmp/strata-mtp-kernel-opt/q6-single-micro /tmp/strata-mtp-kernel-opt/q6-single.metallib bench > bench/results/2026-10-11-metal-cuda-emulation/q6-single-bench.csv 2> bench/results/2026-10-11-metal-cuda-emulation/q6-single-bench.log
xcrun -sdk macosx metal -fno-fast-math -ffp-contract=off -I src/kernels/metal -c bench/results/2026-10-11-metal-cuda-emulation/gu-row-reuse.metal -o /tmp/strata-mtp-kernel-opt/gu-rows.air
xcrun -sdk macosx metallib /tmp/strata-mtp-kernel-opt/gu-rows.air -o /tmp/strata-mtp-kernel-opt/gu-rows.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal bench/results/2026-10-11-metal-cuda-emulation/gu-row-reuse-micro.mm -o /tmp/strata-mtp-kernel-opt/gu-row-reuse-micro
/tmp/strata-mtp-kernel-opt/gu-row-reuse-micro /tmp/strata-mtp-kernel-opt/gu-rows.metallib /tmp/strata-mtp-kernel-opt/gu-rows.metallib /tmp/strata-mtp-kernel-opt/gu-rows.metallib > bench/results/2026-10-11-metal-cuda-emulation/gu-parity.csv 2> bench/results/2026-10-11-metal-cuda-emulation/gu-parity.log
/tmp/strata-mtp-kernel-opt/gu-row-reuse-micro /tmp/strata-mtp-kernel-opt/gu-rows.metallib /tmp/strata-mtp-kernel-opt/gu-rows.metallib /tmp/strata-mtp-kernel-opt/gu-rows.metallib bench > bench/results/2026-10-11-metal-cuda-emulation/gu-bench.csv 2> bench/results/2026-10-11-metal-cuda-emulation/gu-bench.log

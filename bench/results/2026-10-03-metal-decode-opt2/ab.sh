#!/bin/bash
# Same-binary decode A/B on the frozen 10K prompt (round-21 driver). control = round 21's kernels + its
# 2.24 GiB IQ4 view; direct = the new kernels with no view; direct-view = the new kernels with the view kept.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-decode-opt2
RUN="python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json --tokens 256 --repeats 4"
OFF="--env STRATA_METAL_IQ4_DIRECT=0 --env STRATA_METAL_EXPERT_DIRECT=0 --env STRATA_METAL_GR_NORM1=0"
# round 23's kernels did not exist in round 22's binary: keep them off on later binaries to reproduce this round
R23="--env STRATA_METAL_IQ4_SG=0 --env STRATA_METAL_GDN_TAIL=0 --env STRATA_METAL_TOPK_SCAN=0"
RUN="$RUN $R23"
for pass in a b; do
  $RUN --output $OUT/control-$pass $OFF > $OUT/control-$pass.log 2>&1
  $RUN --output $OUT/direct-$pass --env STRATA_METAL_IQ4_EXPAND=0 > $OUT/direct-$pass.log 2>&1
done
$RUN --output $OUT/direct-view --env STRATA_METAL_IQ4_EXPAND=1 > $OUT/direct-view.log 2>&1
echo done

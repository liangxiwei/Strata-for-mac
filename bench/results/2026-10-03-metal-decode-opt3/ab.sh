#!/bin/bash
# Same-binary decode A/B, one engine at a time. round1 = the first round's kernels (this round's three switches
# off); all = everything on. Daily config (no IQ4 view), frozen 10K prompt, 256 recomputed tokens x 4.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-decode-opt3
RUN="python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json --tokens 256 --repeats 4"
OFF="--env STRATA_METAL_IQ4_SG=0 --env STRATA_METAL_GDN_TAIL=0 --env STRATA_METAL_TOPK_SCAN=0"
for pass in a b; do
  if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "another engine is running"; exit 1; fi
  $RUN --output $OUT/round1-$pass $OFF > $OUT/round1-$pass.log 2>&1
  $RUN --output $OUT/all-$pass > $OUT/all-$pass.log 2>&1
done
echo done

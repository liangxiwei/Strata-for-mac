#!/bin/bash
# Same-binary decode A/B: the one-token window writes its GDN state back and the commit skips the recurrence (on)
# against the recurrence in the commit (STRATA_METAL_GDN_INPLACE=0). One engine at a time, three alternating passes.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-gdn-inplace
RUN="python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json --tokens 256 --repeats 4"
for pass in a b c; do
  if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "another engine is running"; exit 1; fi
  $RUN --output $OUT/off-$pass --env STRATA_METAL_GDN_INPLACE=0 > $OUT/off-$pass.log 2>&1
  $RUN --output $OUT/on-$pass > $OUT/on-$pass.log 2>&1
done
echo done

#!/bin/bash
# Real-model numerical gate (round 20's scripts): short fixtures and 10K + cached follow-up, the new kernels on,
# no IQ4 view, against round 20's saved resident traces. Diagnostics read logits back; no timing is taken here.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-gdn-inplace
OLD=bench/results/2026-10-03-metal-iq2-xs-opt
export STRATA_METAL_IQ4_EXPAND=0
for c in short 10k; do
  python3 $OLD/audit_model.py --mode resident --case $c --output $OUT/audit-$c > $OUT/audit-$c.log 2>&1
  python3 $OLD/compare_audit.py $OLD/audit-$c-resident $OUT/audit-$c --output $OUT/compare-$c.json > $OUT/compare-$c.log 2>&1
done
echo done

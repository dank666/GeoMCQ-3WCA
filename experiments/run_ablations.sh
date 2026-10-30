#!/bin/sh
# Ablations and DCR robustness with GLM-4-9B as the judging model (Section 6.4 and Appendix C).
# Run after the main test runs; model calls already made are read from the cache.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
M=THUDM/GLM-4-9B-0414
for S in test gaokao; do
  $PY experiments/run_twc.py --split $S --model $M --tag nodcr --workers 3
  $PY experiments/run_twc.py --split $S --model $M --set knowledge=0 --tag noknow --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --set three_valued=0 --tag twovalued --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --set defer=0 --tag nodefer --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --set min_multi=1 --tag min1 --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr Qwen/Qwen3.5-122B-A10B --tag dcr122b --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --tau 0 --tag tau0 --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --tau 0.8 --tag tau08 --workers 3
  $PY experiments/run_twc.py --split $S --model $M --dcr $M --tau 0.9 --tag tau09 --workers 3
done

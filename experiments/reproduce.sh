#!/bin/sh
# Recompute every result of Section 6 and Appendix C, then the tables.
# With the released cache (results/cache, kb/cache) no model is called.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODELS="Qwen/Qwen2.5-7B-Instruct THUDM/GLM-4-9B-0414 Qwen/Qwen3.5-122B-A10B"
for S in test gaokao; do
  $PY experiments/run_dictionary_matching.py --split $S
  for M in $MODELS; do
    $PY experiments/run_twc.py --split $S --model $M --dcr $M --tag full
    $PY experiments/run_direct.py --split $S --model $M
  done
done
for M in $MODELS; do
  $PY experiments/run_direct.py --split test --model $M --at-least-two
done
sh experiments/run_ablations.sh
$PY experiments/make_results.py
$PY experiments/plot_figures.py

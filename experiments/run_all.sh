#!/usr/bin/env bash
# Full reproduction: human analyses -> UNet feature cache -> RIDL ablation ->
# learning curves -> figures -> paper macros -> visual abstract.
# Usage:  THREADS=16 ./run_all.sh      (from experiments/; uses `uv run` unless PYTHON is set)
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-uv run python}
TH=${THREADS:-16}
mkdir -p ../results
log() { echo "[$(date '+%F %T')] $*" | tee -a ../results/run_all.log; }

log "human analyses"
$PY human_analysis.py > ../results/human_analysis.log 2>&1 || log "human_analysis FAILED"
$PY figures_human.py >> ../results/human_analysis.log 2>&1 || log "figures_human FAILED"

log "frozen SD-UNet feature cache (skips cached segments; Task1/Task2 first)"
$PY extract_unet_features.py --threads "$TH" > ../results/extract.log 2>&1 &
EXT=$!
until [ "$(ls ../cache/unet64/*Task*.npz 2>/dev/null | wc -l)" -ge 40 ] || ! kill -0 $EXT 2>/dev/null; do sleep 60; done
kill $EXT 2>/dev/null && log "extraction stopped (Reference segments are not used by the ablation)"

log "RIDL ablation"
$PY ridl_ablation.py --seeds 2 --threads "$TH" --tag ablation >> ../results/ablation.log 2>&1 || log "ablation FAILED"

log "learning curves"
$PY ridl_ablation.py --curves --seeds 2 --curve-iters 15 --threads "$TH" >> ../results/curves.log 2>&1 || log "curves FAILED"

log "figures, macros, visual abstract"
$PY figures_rl.py >> ../results/figures_rl.log 2>&1 || log "figures_rl FAILED"
$PY make_macros.py >> ../results/run_all.log 2>&1 || log "make_macros FAILED"
$PY visual_abstract.py >> ../results/run_all.log 2>&1 || log "visual_abstract FAILED"
log "DONE"

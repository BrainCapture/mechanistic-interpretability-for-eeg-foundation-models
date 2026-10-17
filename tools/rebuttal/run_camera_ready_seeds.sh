#!/usr/bin/env bash
# Camera-ready reruns of the single-seed rebuttal experiments with fixed SAE seeds 0-4.
# Outputs: results/camera_ready/seed{S}/ (defaults write to results/rebuttal/).
# Run from repo root:  bash tools/rebuttal/run_camera_ready_seeds.sh
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=logs/camera_ready; mkdir -p "$LOG"
SEEDS="${SEEDS:-0 1 2 3 4}"

run() {  # gpu seed outdir script [args...]
  local gpu=$1 seed=$2 out=$3; shift 3
  local name; name=$(basename "$1" .py)
  if [ -f "$out/.done_${name}${TAG:-}" ]; then return 0; fi
  echo "[$(date +%H:%M:%S)] gpu$gpu seed$seed $name $*"
  if CUDA_VISIBLE_DEVICES=$gpu CR_SEED=$seed CR_OUT_DIR=$out $PY "$@" \
       > "$LOG/${name}${TAG:-}_seed${seed}.log" 2>&1; then
    touch "$out/.done_${name}${TAG:-}"
  else
    echo "  FAILED: $name seed$seed (see $LOG/${name}${TAG:-}_seed${seed}.log)"
  fi
}

lane() {  # gpu, then the seeds handled by this lane
  local gpu=$1; shift
  for s in "$@"; do
    local o=results/camera_ready/seed$s; mkdir -p "$o"
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_collateral_matrix.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_leace.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_amnesic.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_selectivity_controlled.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_native_vs_probe.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_tuev_separability.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_slowing_separability.py
    run "$gpu" "$s" "$o" tools/rebuttal/rebuttal_tuev_steering.py --clamp-frac 0.5
    local o25=results/camera_ready/seed${s}_clamp25; mkdir -p "$o25"
    run "$gpu" "$s" "$o25" tools/rebuttal/rebuttal_tuev_steering.py --clamp-frac 0.25
  done
}

read -r -a S <<< "$SEEDS"
lane 0 "${S[0]}" ${S[4]:+"${S[4]}"} &
lane 1 "${S[1]}" &
lane 2 "${S[2]}" &
lane 3 "${S[3]}" &
wait
echo "[$(date +%H:%M:%S)] all lanes done"

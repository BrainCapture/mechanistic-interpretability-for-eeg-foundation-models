#!/usr/bin/env bash
# Camera-ready Fig 4/5: 5 fresh SAE seeds per (encoder, layer) at E=1, k=8, all 5 concepts,
# at the layers the figure displays (1-indexed SleepFM L1-3, LaBraM L1/4/8/12, REVE L1/6/11/16/22).
# Outputs results/camera_ready/fig4/{experiment}.json. Run from repo root.
cd "$(dirname "$0")/../.."
SEEDS="0 1 2 3 4"
CONCEPTS=(Pathology Age Gender "Medication (Psychiatric)" "Medication (ASM)")
r() {  # gpu experiment layer [max_train_tokens]
  local cap=(); [ -n "${4:-}" ] && cap=(--max-train-tokens "$4")
  echo "[$(date +%H:%M:%S)] gpu$1 $2"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python tools/rebuttal/steering_seed_robustness.py \
    --experiment "$2" --layer-idx "$3" --seeds $SEEDS --concepts "${CONCEPTS[@]}" "${cap[@]}" \
    --out-json "results/camera_ready/fig4/$2.json" > "logs/camera_ready/fig4_$2.log" 2>&1 \
    && echo "  done $2" || echo "  FAILED $2"
}
( r 0 sleepfm_finetuned_layer0 0; r 0 sleepfm_finetuned_layer1 1; r 0 sleepfm_finetuned_layer2 2 ) &
( r 1 labram_layer0 0 1000000;  r 1 labram_layer3 3 1000000 ) &
( r 2 labram_layer7 7 1000000;  r 2 labram_layer11 11 1000000 ) &
( for L in 0 5 10 15 21; do r 3 reve_qjbe08_layer$L $L 1000000; done ) &
wait; echo "[$(date +%H:%M:%S)] fig4 multiseed done"

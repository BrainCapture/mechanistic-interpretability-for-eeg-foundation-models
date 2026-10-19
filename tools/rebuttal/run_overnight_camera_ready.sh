#!/usr/bin/env bash
# Overnight camera-ready queue (starts after the Fig 4/5 multiseed run). Run from repo root.
#  A  our-cohort pretrained vs finetuned encoding (8 seeds)        → results/camera_ready/pretrain/
#  C  population spectral steering over 5 SAE seeds               → results/camera_ready/population_spectral_seeds.json
#  B  seeds 5-9 of the single-SAE reruns (→ 10 seeds in total)     → results/camera_ready/seed{5..9}/
cd "$(dirname "$0")/../.."
L=logs/camera_ready
echo "[$(date +%H:%M)] starting camera-ready queue"
mkdir -p results/camera_ready/pretrain
( for e in sleepfm_bc_pretrained_layer2 sleepfm_finetuned_layer2; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/rebuttal/steering_seed_robustness.py --experiment $e \
      --layer-idx 2 --seeds 0 1 2 3 4 5 6 7 --concepts Pathology Age \
      --out-json results/camera_ready/pretrain/$e.json > $L/pretrain_$e.log 2>&1 && echo "A done $e" || echo "A FAILED $e"
  done ) &
( CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/rebuttal/rebuttal_population_spectral_seeds.py --models SleepFM LaBraM \
    > $L/spectral_seeds_a.log 2>&1 && echo "C done SleepFM+LaBraM" || echo "C FAILED SleepFM+LaBraM" ) &
wait
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/rebuttal/rebuttal_population_spectral_seeds.py --models REVE \
    > $L/spectral_seeds_b.log 2>&1 && echo "C done REVE" || echo "C FAILED REVE"
SEEDS="5 6 7 8 9" bash tools/rebuttal/run_camera_ready_seeds.sh > $L/driver_seeds5to9.log 2>&1 && echo "B done"
.venv/bin/python tools/rebuttal/aggregate_camera_ready.py > $L/aggregate.log 2>&1 && echo "aggregated"
echo "[$(date +%H:%M)] overnight queue done"

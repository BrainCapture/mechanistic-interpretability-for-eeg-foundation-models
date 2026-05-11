#!/usr/bin/env bash
# Re-run the SleepFM expansion sweep at SCALED k (k/n constant).
# Original sweep used fixed k=8 across all expansions, which conflated
# capacity with sparsity. Here we set k = 8·E so every (E, k) pair has the
# same active fraction k/n = 8/128 = 6.3%.
#
# Outputs land in NEW experiment directories so the original fixed-k results
# stay intact for direct comparison:
#   sae_sleepfm_exp{E}_k{K}_layer2.pt           (NEW SAE checkpoints)
#   results/experiments/sleepfm_finetuned_exp{E}_k{K}_layer2/   (NEW metadata)
#
# Usage:
#   bash tools/run_expansion_sweep_kscaled.sh 2>&1 | tee logs/kscaled_sweep.log
#
# Runtime: ~4–6 hours total (7 SAEs × ~30 min + 7 app caches × ~15 min)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENCODER_NS="sleepfm_finetuned"          # output namespace
ENCODER_BASE="sleepfm"                  # backend
WEIGHTS="checkpoints/finetuned/sleepfm1.ckpt"
TAG="finetuned"
LAYER=2
EMBED_DIM=128
ACTIVE_FRACTION_K0=8                    # k at E=1; scaled as k = K0·E

# Expansion ratios to sweep. E=1 is already correct (k=8) — skip retraining
# the baseline by default; pass --include-baseline to also rerun it.
EXPANSIONS=(1 2 4 8 16 32 64)

mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Training ──────────────────────────────────────────────────────────────────
for EXP in "${EXPANSIONS[@]}"; do
    K=$(( ACTIVE_FRACTION_K0 * EXP ))
    log "=== E=${EXP}  k=${K}  (k/n=${ACTIVE_FRACTION_K0}/${EMBED_DIM}=$(( 100*ACTIVE_FRACTION_K0/EMBED_DIM ))%) ==="
    uv run tools/train_sae_layers.py \
        --encoder "$ENCODER_BASE" \
        --weights-path "$WEIGHTS" \
        --tag "$TAG" \
        --layers "$LAYER" \
        --expansion "$EXP" \
        --k "$K"
done

# ── Metadata ──────────────────────────────────────────────────────────────────
for EXP in "${EXPANSIONS[@]}"; do
    K=$(( ACTIVE_FRACTION_K0 * EXP ))
    N_FEATURES=$(( EMBED_DIM * EXP ))
    EXP_NAME="${ENCODER_NS}_exp${EXP}_k${K}_layer${LAYER}"
    EXP_DIR="results/experiments/${EXP_NAME}"
    mkdir -p "$EXP_DIR"
    cat > "$EXP_DIR/metadata.json" <<JSON
{
  "name": "${EXP_NAME}",
  "display_name": "SleepFM finetuned · exp${EXP} · k${K} (kscaled) · layer ${LAYER}",
  "encoder": "sleepfm",
  "embed_dim": ${EMBED_DIM},
  "fs": 128,
  "patch_size": 128,
  "target_layer": ${LAYER},
  "expansion": ${EXP}.0,
  "k": ${K},
  "n_features": ${N_FEATURES},
  "n_clusters": 200,
  "sae_checkpoint": "results/features/sleepfm_finetuned/sae_sleepfm_exp${EXP}.0_k${K}_layer${LAYER}.pt",
  "xae_checkpoint": "results/xae/sleepfm_finetuned/xae_checkpoint.pt",
  "codebook_path": "results/xae/sleepfm_finetuned/codebook/codebook.pt",
  "feature_explanations": "results/xae/sleepfm_finetuned/explanations/feature_explanations.json",
  "weights_path": "checkpoints/finetuned/sleepfm1.ckpt"
}
JSON
    log "metadata: $EXP_DIR/metadata.json   (n=${N_FEATURES}, k=${K})"
done

# ── App caches ────────────────────────────────────────────────────────────────
for EXP in "${EXPANSIONS[@]}"; do
    K=$(( ACTIVE_FRACTION_K0 * EXP ))
    EXP_NAME="${ENCODER_NS}_exp${EXP}_k${K}_layer${LAYER}"
    log "=== App cache for ${EXP_NAME} ==="
    uv run tools/build_app_cache.py \
        --experiment "$EXP_NAME" \
        --skip-umap \
        --skip-morphology
done

log "=== Done. Next: run plot_taxonomy_expansion.py with the new experiment list ==="

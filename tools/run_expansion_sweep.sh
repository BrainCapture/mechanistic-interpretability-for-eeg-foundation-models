#!/usr/bin/env bash
# Overnight expansion sweep: train SAEs at exp 2×, 4×, 8× on sleepfm_finetuned
# layer 2, build enrichment caches, run monosemanticity analysis and concept steering.
#
# Usage:
#   bash tools/run_expansion_sweep.sh 2>&1 | tee logs/expansion_sweep.log
#
# Runtime estimate: ~3–5 hours total

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENCODER="sleepfm_finetuned"   # experiment namespace (results/features/sleepfm_finetuned/)
ENCODER_BASE="sleepfm"        # actual encoder name for train_sae_layers.py
ENCODER_WEIGHTS="checkpoints/finetuned/sleepfm1.ckpt"
ENCODER_TAG="finetuned"
LAYER=2
K=8
EXPANSIONS=(2 4 8)

mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Train SAEs ────────────────────────────────────────────────────────
for EXP in "${EXPANSIONS[@]}"; do
    log "=== Training SAE expansion=${EXP} ==="
    uv run tools/train_sae_layers.py \
        --encoder "$ENCODER_BASE" \
        --weights-path "$ENCODER_WEIGHTS" \
        --tag "$ENCODER_TAG" \
        --layers "$LAYER" \
        --expansion "$EXP" \
        --k "$K"
    log "Done training exp=${EXP}"
done

# ── Step 2: Create experiment metadata directories ────────────────────────────
for EXP in "${EXPANSIONS[@]}"; do
    N_FEATURES=$(( 128 * EXP ))
    EXP_DIR="results/experiments/${ENCODER}_exp${EXP}_layer${LAYER}"
    mkdir -p "$EXP_DIR"
    cat > "$EXP_DIR/metadata.json" <<JSON
{
  "name": "${ENCODER}_exp${EXP}_layer${LAYER}",
  "display_name": "SLEEPFM finetuned · exp${EXP} · layer ${LAYER}",
  "encoder": "sleepfm",
  "embed_dim": 128,
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
    log "Created metadata: $EXP_DIR/metadata.json  (n_features=${N_FEATURES})"
done

# ── Step 3: Build enrichment caches ──────────────────────────────────────────
# --skip-umap and --skip-morphology shave off the slow steps;
# enrichment still runs on the full dataset (enr_max_tokens = full).
for EXP in "${EXPANSIONS[@]}"; do
    EXPERIMENT="${ENCODER}_exp${EXP}_layer${LAYER}"
    log "=== Building app cache for ${EXPERIMENT} ==="
    uv run tools/build_app_cache.py \
        --experiment "$EXPERIMENT" \
        --skip-umap \
        --skip-morphology
    log "Done cache for ${EXPERIMENT}"
done

# ── Step 4: Monosemanticity analysis ─────────────────────────────────────────
log "=== Running monosemanticity analysis ==="
uv run tools/analyze_monosemanticity.py --filter sleepfm_finetuned
log "Done monosemanticity analysis"

# ── Step 5: Concept steering at each expansion ────────────────────────────────
# Update plot_concept_steering_xae_all.py to read from a different experiment dir
# is future work; for now run the standard script (uses exp1 baseline) to confirm
# the existing figures are stable, then produce age+classification figures
# for each new expansion using a dedicated runner.
log "=== Running concept steering for baseline (exp1) age+classification ==="
uv run tools/plot_concept_steering_xae_all.py \
    --concepts age classification classification_child classification_adult
log "Done steering baseline"

log "=== All done. Check results/monosemanticity/ for updated figures. ==="

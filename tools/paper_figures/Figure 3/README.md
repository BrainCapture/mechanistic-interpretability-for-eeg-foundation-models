# Figure 3 — Monosemanticity taxonomy

> Per-encoder grid of the fraction of concept-enriched SAE features in each of three taxonomy classes (Separable / Entangled / Dead), across layers × expansion factors. Highlights the optimal (ℓ*, E*) operating point per encoder.

![figure](figure.png)

## Regenerate

**Data bundling: TODO.** The script currently expects `results/experiments/{exp}/taxonomy_cache.pt` (or `app_cache.pt`) for each of ~140 (encoder × layer × expansion) configurations. Combined size is GB-scale, so we need to pre-compute the (separable, entangled, dead) percentages and ship only a small summary `data.npz`.

Until then, the script only runs in the development repo where the full result caches are available.

## Data provenance (when bundled)

Each cell is a single percentage derived from `feature_meta_enrichment` in the per-experiment `taxonomy_cache.pt`, classified via `_classify(...)` from the development repo's `tools/_archive/plot_taxonomy_expansion.py`. Output shape: 3 encoders × ~7 layers × ~7 expansion factors × 3 classes ≈ 450 numbers — easily a few KB once bundled.

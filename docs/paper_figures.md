# Paper Figure Registry

All figures live in `paper/figures/`. Referenced in `paper/neurips2026.typ` as `image("figures/<name>")`.

## Figure → script mapping

| Figure file | Paper label | Generation script | Notes |
|---|---|---|---|
| `tcav_layer_emergence.png` | `fig-tcav-layers` | `tools/plot_tcav_layer_emergence.py` | Requires `results/tcav/sleepfm_finetuned_layer{0,1,2}/tcav_cache.pt` |
| `taxonomy_expansion.png` | `fig-taxonomy` | `tools/plot_taxonomy_expansion.py` | Uses `results/features/sleepfm_finetuned/` sweep results |
| `concept_steering_xae_age_combined.png` | `fig-steering` | `tools/combine_age_steering_figures.py` | Composites child + adult PNGs; regenerate if either source changes |
| `concept_steering_xae_classification_child_all.png` | _(source)_ | `tools/plot_concept_steering_xae_all.py --group child` | Use `--n-subsample 500` to keep runtime ~2 min |
| `concept_steering_xae_classification_adult_all.png` | _(source)_ | `tools/plot_concept_steering_xae_all.py --group adult` | Use `--n-subsample 500` to keep runtime ~2 min |
| `concept_steering_curve.png` | `fig-probe` | `tools/plot_concept_steering_curve.py` | AUROC Pareto front; age vs clinical as features clamped |
| `expansion_steering_comparison_age.png` | `fig-expansion-steering` | `tools/plot_expansion_steering_comparison.py` | Requires exp1/2/4/8 app caches for sleepfm_finetuned |

## Regeneration order

When re-running after a new model checkpoint:
1. `plot_tcav_layer_emergence.py` (reads TCAV caches directly)
2. `plot_taxonomy_expansion.py` (reads SAE checkpoints + feature explanations)
3. `plot_concept_steering_xae_all.py` × 2 (child, adult — slow; use `--n-subsample 500`)
4. `combine_age_steering_figures.py` (composites the two above)
5. `plot_concept_steering_curve.py`
6. `plot_expansion_steering_comparison.py`

## Background theme

All figures use light background (white). Confirmed via corner pixel check:
`python3 -c "from PIL import Image; img=Image.open('paper/figures/X.png'); print(img.getpixel((0,0)))"` — value >200 = white.

## Figures NOT in the paper (deferred)

- `spike_before_after.png`, `spike_zoom.png`, `spike_steering_n20.png`, `spike_diag.png` — spike steering, deferred until granular label training (see `docs/granular_training_plan.md`)
- `concept_steering_probe.png` — probe AUROC before/after SAE reconstruction (not yet computed)

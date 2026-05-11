# Figure 6 — Spectrum-level concept steering (abnormal → normal)

> SleepFM layer 2, E=8. Baseline abnormal source vs. normal target centroid (left), and decoded spectrum after clamping the top n=104 and n=164 TCAV-aligned features to the target centroid (centre, right). Shading = 95% bootstrap CIs on the target mean.

![figure](figure.png)

## Regenerate

**Data bundling: TODO.** The script currently expects:

- `results/experiments/sleepfm_finetuned_layer2/metadata.json`
- `results/experiments/sleepfm_finetuned_layer2/app_cache.pt` (~100s of MB)
- the SAE checkpoint referenced in metadata (`results/features/sleepfm_finetuned/sae_*.pt`)
- `results/steering_cache/sleepfm_finetuned_layer2/steering_cache.pt` (~GB-scale)

Self-contained version is pending: we need to pre-compute the post-clamp decoded spectra (and source/target centroids) once and ship only those as a small `data.npz`. Until then the script runs only in the development repo.

## Current invocation (in dev repo)

```bash
uv run python tools/paper_figures/Figure\ 6/plot.py \
  --experiment sleepfm_finetuned_layer2 \
  --n-clamp 104 164
```

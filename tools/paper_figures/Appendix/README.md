# Camera-ready appendix figures

| File | Paper | Content |
|---|---|---|
| `app_collateral.png` | App. D | Steer one concept, measure all five: SAE (10 seeds), LEACE, INLP |
| `app_matched.png` | App. D | Off-target age preserved at matched abnormality suppression (static image) |
| `app_native_head.png` | App. E | Deployed classifier head vs probe vs random clamping |
| `app_spectral_seeds.png` | App. G | Population band-amplitude shift under steering, 5 SAE seeds (static image) |
| `app_pretrain_finetune.png` | App. H | Pretrained vs finetuned encoding, abn–age CAV cosine (static image) |

```bash
uv run python tools/paper_figures/Appendix/plot.py
```

`plot.py` regenerates `app_collateral` and `app_native_head`. The three static images are the exact files included in the camera-ready paper.

`data.json` bundles `results/camera_ready/summary.json` (from
`tools/rebuttal/aggregate_camera_ready.py`), the per-seed deployed-head and matched-suppression
outputs, and `population_spectral_seeds.json`.

# Robustness, baselines, and public-corpus replication

Scripts behind §4.6 and Appendices B–H of the camera-ready paper. They were written for the
NeurIPS 2026 review discussion, then rerun with fixed SAE seeds for the camera-ready.
Run every script from the repository root.

| Paper item | Script(s) | Output |
|---|---|---|
| Fig. 4–5: selectivity per (encoder, layer, concept), 5 SAE seeds | `run_fig4_multiseed.sh` → `steering_seed_robustness.py` (uses `steering_significance.analyse_cell`) | `results/camera_ready/fig4/{experiment}.json` |
| Table 3: direction sources (PCA, ICA, dense CAV, random dictionary, centroid) | `rebuttal_steering_baselines.py`, `rebuttal_pca_ica_figure.py` | `results/rebuttal/steering_baselines_*.json` |
| Matched-strength erasers (LEACE, INLP) | `rebuttal_selectivity_controlled.py`, `rebuttal_leace.py`, `rebuttal_amnesic.py` | `selectivity_controlled.json`, `leace_matrix.json`, `amnesic_matrix.json` |
| Collateral matrix | `rebuttal_collateral_matrix.py` | `collateral_matrix.json` |
| Deployed classifier head | `rebuttal_native_vs_probe.py`, `steering_native_head.py` | `native_vs_probe.json` |
| TUAB replication | `preprocess_tuab_rebuttal.py` → `finetune_sleepfm_tuab.py` → `build_tuab_steering_cache.py` → `rebuttal_tuab_entanglement.py` | `sleepfm_tuab_layer2_entanglement.json` |
| TUEV / TUSL | `preprocess_tuev_rebuttal.py` (set `TUEV_ROOT`), `rebuttal_tuev_separability.py`, `rebuttal_tuev_steering.py`, `rebuttal_slowing_separability.py` | `tuev_*.json`, `slowing_separability.json` |
| Pretraining vs finetuning | `build_bc_pretrained_cache.py`, `rebuttal_cav_cosine.py` | `cav_cosine.json` |
| Population spectral steering | `rebuttal_population_spectral.py`, `rebuttal_population_spectral_multi.py` | `population_spectral*.json` |
| Taxonomy-rule sensitivity | `rebuttal_taxonomy_sensitivity.py` | `taxonomy_sensitivity.md` |

**Seeded reruns.** `run_camera_ready_seeds.sh` reruns the single-SAE experiments with seeds
0–4. It writes to `results/camera_ready/seed{S}/` through two environment variables read by
`_cr.py`: `CR_OUT_DIR` sets the output directory and `CR_SEED` sets the SAE seed.
`aggregate_camera_ready.py` combines the runs into `results/camera_ready/summary.{json,md}`.
The camera-ready figures are drawn by `tools/paper_figures/plot_fig4_camera_ready.py`,
`plot_fig5_camera_ready.py`, and `plot_camera_ready_appendix.py`.

**Data.** The clinical cohort is private, as described in the top-level README. TUAB, TUEV,
and TUSL are public corpora from the Temple University Hospital EEG resource and require its
data use agreement. Every result in the table above can be reproduced from them.

# Analysis pipeline

The minimum-viable set of scripts that produces every result in the paper. Pruned from the development repo to keep this companion repo focused on the paper's methodology.

The pipeline is sequential — later steps depend on outputs from earlier steps. None of these scripts will run end-to-end here because the clinical EEG dataset and fine-tuned encoder weights used in the paper are private; the scripts are included so the methodology is auditable, and so researchers porting the pipeline to their own corpus have a clear starting point.

## Pipeline overview

```
                            ┌────────────────────┐
                            │  pretrained encoder │
                            │ (SleepFM/REVE/LaBraM)│
                            └─────────┬──────────┘
                                      │  binary normal/abnormal
                                      ▼
                        finetune_{sleepfm_v2,reve,labram}_binary.py
                                      │
                                      ▼
                              train_sae_layers.py
                              train_sae_expansions.py
                                      │
                                      ▼
                                 train_xae.py
                                 bootstrap_xae_ci.py
                                 compare_xae.py
                                      │
                                      ▼
                              build_app_cache.py
                              build_taxonomy_cache.py
                              build_steering_cache.py
                                      │
                                      ▼
                                run_tcav.py
                                analyze_monosemanticity.py
                                probe_layer_sweep_kfold.py
                                probe_table_finetune_sae.py
                                run_kfold_native_head.py
                                      │
                                      ▼
                              tools/paper_figures/Figure {1..7}/plot.py
```

## Script → paper section

| Script | Produces | Paper |
|---|---|---|
| `finetune_sleepfm_v1_binary.py` | SleepFM binary-finetuned checkpoint (`sleepfm1.ckpt`) | §3 (encoders) |
| `finetune_reve_binary.py` | REVE binary-finetuned checkpoint | §3 |
| `finetune_labram_binary.py` | LaBraM binary-finetuned checkpoint | §3 |
| `train_sae_layers.py` | TopK SAE on each encoder layer | §3.1, Fig 2 |
| `train_sae_expansions.py` | Expansion sweep (E=1..64) at a single layer | §3.2, Fig 3, Fig 7 |
| `train_xae.py` | Spectral decoder (XAE) per encoder | §2 |
| `bootstrap_xae_ci.py` | Per-token bootstrap CIs for XAE R² | §2 |
| `compare_xae.py` | Cross-model XAE R² comparison | §2 (helper for `bootstrap_xae_ci`) |
| `build_app_cache.py` | Per-experiment feature-enrichment + spectral cache | infrastructure (used by steering / taxonomy caches) |
| `build_taxonomy_cache.py` | Pre-computed monosemanticity taxonomy fractions | Fig 3 |
| `build_steering_cache.py` | Per-experiment steering tokens + metadata | §5, Fig 4/5/6 |
| `run_tcav.py` | TCAV scores (3 variants — see `src/sae4eeg/tcav.py`) | §4, Fig 4 |
| `analyze_monosemanticity.py` | Per-feature category-discrimination test | §3.1 (epileptiform null result) |
| `probe_layer_sweep_kfold.py` | 5-fold AUROC of linear probe under SAE substitution at each layer | §3.2, Fig 2 |
| `probe_table_finetune_sae.py` | Operating-layer SAE-faithfulness table (per encoder) | §3.2 (table) |
| `run_kfold_native_head.py` | No-SAE baseline AUROC (per encoder × fold) | §3.2 (baseline column) |
| `run_expansion_sweep.sh` | Convenience wrapper for the expansion sweep | shell helper |
| `run_steering_sweep.sh` | Convenience wrapper for the steering sweep | shell helper |

## What's *not* here (and why)

The development repo has dozens of additional scripts for: the Streamlit explorer app, codebook clustering, the Stage-2 granular-label experiments, exploratory plots, monosemanticity baselines (PCA/ICA, embedding direct), feature-activation maximisation, and various intermediate diagnostics. None of those feed paper figures or tables directly, so they are not part of this companion repo.

If you need any of them: the development snapshot at git tag `paper-submission-2026-05` in the development repo preserves the full state.

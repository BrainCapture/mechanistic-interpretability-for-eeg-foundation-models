# Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders

Companion code repository for the preprint **"Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders"** by William Lehn-Schiøler, Magnus Ruud Kjær, Rahul Thapa, Magnus Guldberg Pedersen, Anton Storgaard Mosquera, Nick Williams, Andreas Brink-Kjær, Tue Lehn-Schiøler, Sándor Beniczky, Radu Gatej, Lars Kai Hansen, and Sadasivan Puthusserypady (BrainCapture A/S and collaborators).

The paper PDF, LaTeX source, and final figures are in [`paper/`](paper/).

**Interactive demo:** <https://sae4eeg-app-506594542723.europe-west6.run.app/neurips-2026/>

## What this repo contains

```
src/sae4eeg/          Core library: SAE, XAE, TCAV, encoder wrappers, dataset
tools/                Analysis pipeline scripts (SAE/XAE/TCAV training, steering, probes)
tools/paper_figures/  Scripts that generated every figure in the paper
paper/                The submitted LaTeX package + compiled PDF
docs/                 Methodology notes (figure registry, cross-model plan)
```

## What this repo does *not* contain (and why)

This repo is **not end-to-end reproducible**, by design. The components that would be required to retrain or rerun the pipeline are private to BrainCapture and cannot be released:

- **EEG data** — the BrainCapture clinical EEG dataset (~2,900 subjects) is not publicly available
- **Fine-tuned encoder weights** — the binary-finetuned SleepFM, REVE, and LaBraM checkpoints used in the paper are not released
- **Pre-built result caches** — SAE/XAE checkpoints, TCAV caches, steering caches

**Pre-trained encoder weights** (the starting point for the fine-tuned variants) are publicly available from their original authors:

| Encoder | Source |
|---|---|
| SleepFM | Thapa et al., 2024 — see paper §3 for the repository link |
| REVE | <https://huggingface.co/brain-bzh/reve-base> (gated; requires EDPB agreement) |
| LaBraM | Jiang et al., 2024 — see paper §3 for the repository link |

What is included is sufficient to (a) inspect the methodology, (b) audit the analysis code, and (c) port the pipeline to your own EEG corpus and encoder. See `docs/new_model_guide.md` for per-encoder integration notes.

## Pipeline overview

```
frozen encoder  →  SAE  →  XAE (spectral decoder)
                    ↓        ↓
                   TCAV   feature explanations
                    ↓
              concept steering
```

| Step | Script | Output |
|---|---|---|
| 1. Train SAEs on encoder layers | `tools/train_sae_layers.py` | `results/features/{encoder}/sae_*.pt` |
| 2. Train XAE (spectral decoder) | `tools/train_xae.py` | `results/xae/{encoder}/` |
| 3. Compute TCAV (concept attribution) | `tools/run_tcav.py` | `results/tcav/{experiment}/tcav_cache.pt` |
| 4. Build steering cache | `tools/build_steering_cache.py` | `results/steering_cache/{experiment}/` |
| 5. Compute monosemanticity taxonomy | `tools/analyze_monosemanticity.py` | per-experiment |
| 6. Generate paper figures | `tools/paper_figures/plot_*.py` | `paper/figures/*.{png,pdf}` |

Full pipeline documentation: `docs/paper_figures.md` (figure → script registry), `docs/concept_steering_cross_model_plan.md` (cross-model steering methodology), `docs/granular_training_plan.md` (granular-label experiments).

## Paper figures

The scripts that produced each figure in the paper are in [`tools/paper_figures/`](tools/paper_figures/):

| Paper figure | Script |
|---|---|
| Pipeline diagram | (hand-drawn, not script-generated) |
| Dictionary size | `plot_dictionary_size.py` |
| Layer-sweep faithfulness | `plot_layer_sweep_kfold_wide.py` |
| Taxonomy grid | `plot_paper_taxonomy_grid.py` |
| Concept steering curves (3×3) | `plot_concept_steering_curves_3x3_paper.py` |
| Cross-model steering | `plot_cross_model_steering_concepts_v2_optimalE.py` |
| Perfect steering (3-panel) | `plot_perfect_steering_three_panel.py` |

## Installation

```bash
# Python 3.12 required
uv sync          # installs the dependencies declared in pyproject.toml
```

Most scripts accept a `--help` flag describing their inputs and outputs.

## Citation

```bibtex
@misc{lehnschioler2026sae4eeg,
  title  = {Mechanistic Interpretability of {EEG} Foundation Models via Sparse Autoencoders},
  author = {Lehn-Schi{\o}ler, William and Kj{\ae}r, Magnus Ruud and Thapa, Rahul and
            Pedersen, Magnus Guldberg and Mosquera, Anton Storgaard and Williams, Nick and
            Brink-Kj{\ae}r, Andreas and Lehn-Schi{\o}ler, Tue and Beniczky, S{\'a}ndor and
            Gatej, Radu and Hansen, Lars Kai and Puthusserypady, Sadasivan},
  year   = {2026},
  note   = {Preprint},
}
```

See `CITATION.cff` for machine-readable metadata.

## License

This code is released under the **PolyForm Noncommercial License 1.0.0** — see [`LICENSE`](LICENSE). Use for academic research, evaluation, and personal study is permitted; commercial use is not. Copyright (c) 2026 BrainCapture.

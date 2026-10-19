# Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders

Companion code repository for **"Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders"** (Lehn-Schiøler et al., NeurIPS 2026).

Authors: William Lehn-Schiøler, Magnus Ruud Kjær, Rahul Thapa, Magnus Guldberg Pedersen, Anton Mosquera Storgaard, Nick Williams, Andreas Brink-Kjær, Tue Lehn-Schiøler, Sadasivan Puthusserypady, James Zou, Lars Kai Hansen.

Affiliations: BrainCapture (Kongens Lyngby, Denmark); DTU Health Tech and DTU Compute (Technical University of Denmark, Kongens Lyngby); Stanford University; Seer Medical (Melbourne).

## What this repo contains

```
src/mecheeg/          Core library: SAE, spectral decoder, TCAV, encoder wrappers, dataset
tools/                Analysis pipeline scripts (SAE/spectral decoder/TCAV training, steering, probes)
tools/paper_figures/  Per-figure self-contained directories (plot.py + data + rendered figure)
app/                  Interactive Streamlit explorer (browse SAE features, TCAV scores, steering)
```

## What this repo does *not* contain (and why)

This repo is **not end-to-end reproducible**, by design. The components that would be required to retrain or rerun the pipeline are private to BrainCapture and cannot be released:

- **EEG data** — the BrainCapture clinical EEG dataset (3,036 subjects, collected with BrainCapture's BC-1 system) used in the paper is not publicly available
- **Fine-tuned encoder weights** — the binary-finetuned SleepFM, REVE, and LaBraM checkpoints used in the paper are not released
- **Pre-built result caches** — SAE/spectral decoder checkpoints, TCAV caches, steering caches

**Pre-trained encoder weights** (the starting point for the fine-tuned variants) are publicly available from their original authors:

| Encoder | Source |
|---|---|
| SleepFM | original SleepFM publication — see paper §3 for the citation and repository link |
| REVE | <https://huggingface.co/brain-bzh/reve-base> (gated; requires EDPB agreement) |
| LaBraM | Jiang et al., 2024 — see paper §3 for the repository link |

What is included is sufficient to (a) inspect the methodology and (b) audit the analysis code. Researchers porting the pipeline to their own EEG corpus and encoder can follow the existing encoder wrappers in `src/mecheeg/encoders.py` (`SleepFMBackend`, `REVEBackend`, `LaBraMBackend`) as templates.

## Reproducing results on public data

The central steering and entanglement results are replicated on public corpora from the
Temple University Hospital EEG resource: TUAB, TUEV and TUSL. A data use agreement is
required. These replications can be rerun end to end with the code here. See
[`tools/rebuttal/README.md`](tools/rebuttal/README.md) for the full script → paper map.

```bash
uv run tools/rebuttal/preprocess_tuab_rebuttal.py            # TUAB → 128 Hz, 60 s windows
uv run tools/rebuttal/finetune_sleepfm_tuab.py               # finetune SleepFM on TUAB
uv run tools/rebuttal/build_tuab_steering_cache.py
uv run tools/rebuttal/rebuttal_tuab_entanglement.py --sae-seeds 8
```

All steering numbers in the camera-ready are means over independently seeded SAEs.
`tools/rebuttal/run_camera_ready_seeds.sh` and `tools/rebuttal/run_fig4_multiseed.sh`
reproduce them.

## Pipeline overview

```
frozen encoder  →  SAE  →  spectral decoder
                    ↓        ↓
                   TCAV   feature explanations
                    ↓
              concept steering
```

| Step | Script | Output |
|---|---|---|
| 1. Train SAEs on encoder layers | `tools/train_sae_layers.py` | `results/features/{encoder}/sae_*.pt` |
| 2. Train spectral decoder | `tools/train_spectral_decoder.py` | `results/spectral_decoder/{encoder}/` |
| 3. Compute TCAV (concept attribution) | `tools/run_tcav.py` | `results/tcav/{experiment}/tcav_cache.pt` |
| 4. Build steering cache | `tools/build_steering_cache.py` | `results/steering_cache/{experiment}/` |
| 5. Compute monosemanticity taxonomy | `tools/analyze_monosemanticity.py` | per-experiment |
| 6. Generate paper figures | `tools/paper_figures/Figure N/plot.py` | `tools/paper_figures/Figure N/figure.{png,pdf}` |

See `tools/README.md` for the full script → paper section map.

## Paper

The NeurIPS 2026 camera-ready paper (PDF and LaTeX source) will be added to `paper/` once it is final.

## Paper figures

Each figure has its own self-contained directory under [`tools/paper_figures/`](tools/paper_figures/) — a `plot.py`, the bundled input `data.{json,npz}` where applicable, the rendered `figure.{png,pdf}`, and a `README.md` describing the figure and its data provenance.

```bash
uv run python "tools/paper_figures/Figure N/plot.py"
```

regenerates Figure N. Every matplotlib figure (2, 3, 4, 5, 6, 7) is self-contained — no `results/`, encoder weights, or EEG data needed. Figure 1 is an interactive React figure; see [its README](tools/paper_figures/Figure%201/CLAUDE.md).

## Installation

```bash
# Python 3.12 required
uv sync                # core dependencies
uv sync --group app    # + Streamlit & Plotly (for the explorer)
```

Most scripts accept a `--help` flag describing their inputs and outputs.

## Interactive explorer

```bash
uv run streamlit run app/main.py
```

The app walks through the SAE features, TCAV concept attributions, and concept steering interactively. Without the result caches (which require the private dataset + encoder weights — see the section above), each tab will display a "build this cache first" message and the matching `tools/...` command. The development repo runs against full caches; from this public repo the value is in (a) reading how the analyses are presented and (b) verifying the code is self-consistent. A hosted instance of the app running against the full caches is linked from the preprint.

## Citation

See [`CITATION.cff`](CITATION.cff) for the full author list and BibTeX-compatible metadata.

## License

This code is released under the **PolyForm Noncommercial License 1.0.0** — see [`LICENSE`](LICENSE). Use for academic research, evaluation, and personal study is permitted; commercial use is not.

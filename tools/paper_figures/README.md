# Paper figures

Each subdirectory `Figure N/` is self-contained and corresponds to one figure in the paper:

| # | Title | Data file | Notes |
|---|---|---|---|
| 1 | Pipeline overview | — | Interactive React figure (Babel-standalone, browser-rendered) |
| 2 | SAE-faithfulness layer sweep | `data.json` (14 KB) | byte-identical regen |
| 3 | Monosemanticity taxonomy across SAE expansion and encoder depth | `data.json` (30 KB) | visually identical (matplotlib version-drift in PNG bytes) |
| 4 | Concept encoding strength and steering selectivity | `data.npz` (2.9 MB) | byte-identical regen |
| 5 | Steering sweeps across the encoding-selectivity landscape | `data.npz` (2.9 MB, same source as Fig 4) | byte-identical regen |
| 6 | Spectrum-level concept steering (abnormal → normal) | `data.npz` (7 KB) | byte-identical regen |
| 7 | SAE dictionary size (appendix) | — (hardcoded in script) | byte-identical regen |

## Conventions

Inside each `Figure N/`:

```
plot.py             canonical script — reads from this directory, writes here too
data.{json,npz}     pre-computed input data (where applicable)
figure.png          submitted version (and what `plot.py` overwrites on rerun)
figure.pdf          same in PDF
README.md           what the figure shows + provenance of data.{...}
```

## Regenerating any one figure

```bash
uv run python "tools/paper_figures/Figure N/plot.py"
```

Every matplotlib-based figure (2, 3, 4, 5, 6, 7) is self-contained — running the script overwrites `figure.png` and `figure.pdf` from the bundled data alone. No `results/`, no encoder weights, no EEG data required.

Figure 1 is an interactive HTML figure; see [`Figure 1/CLAUDE.md`](Figure%201/CLAUDE.md) for run instructions.

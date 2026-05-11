# Paper figures

Each subdirectory `Figure N/` is self-contained and corresponds to one figure in the paper:

| # | Title | Self-contained? |
|---|---|---|
| 1 | Pipeline overview (interactive React figure) | yes |
| 2 | SAE-faithfulness layer sweep | yes — `data.json` bundled |
| 3 | Monosemanticity taxonomy across SAE expansion and encoder depth | TODO (caches GB-scale, need pre-computed summary) |
| 4 | Concept encoding strength and steering selectivity | yes — `data.npz` bundled |
| 5 | Steering sweeps across the encoding-selectivity landscape | yes — `data.npz` bundled |
| 6 | Spectrum-level concept steering (abnormal → normal) | TODO (needs steering / SAE / app caches) |
| 7 | SAE dictionary size (appendix) | yes — values hardcoded in script |

## Conventions

Inside each `Figure N/`:

```
plot.py        canonical script — reads from this directory, writes here too
data.{json,npz,...}   pre-computed input data
figure.png     submitted version (and what `plot.py` overwrites on rerun)
figure.pdf     same in PDF
README.md      what the figure shows + provenance of data.{...}
```

For figures marked TODO, `plot.py` is included verbatim from the development repo but reads from `results/` paths that are not shipped. See each figure's README for what needs to be bundled.

## Regenerating any one figure

```bash
uv run python "tools/paper_figures/Figure N/plot.py"
```

For figures marked self-contained above, this overwrites `figure.png` and `figure.pdf` with a byte-identical copy of the submitted version.

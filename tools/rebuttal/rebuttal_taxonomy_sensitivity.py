"""Rebuttal T1 — taxonomy operational definition + threshold-sensitivity sweep.

Answers reviewer HdqW Q2 / AC point 4: state the exact Separable / Entangled / Dead
decision rule and show the fractions are stable to the three thresholds it depends on.

The rule (reproduced exactly from tools/paper_figures/plot_taxonomy_expansion.py):
  For each SAE feature, count n = number of metadata FIELDS
  (age_group, gender, indication_group, medication_group) with at least one
  ENRICHED category, where a category counts iff  p < P_THRESH  AND  fold > FOLD_MIN
  (p = raw Mann-Whitney U on in-group vs out-group firing; fold = fire-rate ratio).
    n == 1            -> Separable
    n >= 2            -> Entangled
    n == 0 and fires below DEAD_FIRE_FRAC x (uniform TopK rate k/N) -> Dead
    n == 0 otherwise  -> Uncharacterized  (collapsed into "Dead" for the 3-way display)

Defaults: P_THRESH = 0.05, FOLD_MIN = 1.0, DEAD_FIRE_FRAC = 0.25.

This script sweeps each threshold around its default at the three encoders'
operating cells (E*=8) and writes a markdown table + a Nature-style figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))

import plot_taxonomy_expansion as tx  # noqa: E402  (the reference classifier)

import torch  # noqa: E402

# Operating cells (E*=8) at each encoder's operating layer.
CELLS = [
    ("SleepFM", "sleepfm_finetuned_exp8_layer2"),
    ("REVE",    "reve_qjbe08_exp8_layer8"),
    ("LaBraM",  "labram_layer11_exp8"),
]

EXP_DIR = ROOT / "results" / "experiments"
OUT_DIR = ROOT / "results" / "rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cell(exp: str):
    """Mirror plot_paper_taxonomy_grid._load_taxonomy's inputs (enr, fire rates, expected)."""
    exp_dir = EXP_DIR / exp
    cache_path = None
    for cand in ("taxonomy_cache.pt", "app_cache.pt"):
        p = exp_dir / cand
        if p.exists():
            cache_path = p
            break
    if cache_path is None:
        raise FileNotFoundError(f"no taxonomy/app cache for {exp}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    enr = cache.get("feature_meta_enrichment")
    if not enr:
        raise ValueError(f"{exp}: cache has no feature_meta_enrichment")
    feature_stats = cache.get("feature_stats") or []
    meta = cache.get("meta") or {}
    n_features = meta.get("n_features", len(enr))
    k = meta.get("k", 8)
    expected = 100.0 * k / max(n_features, 1)
    fr = (
        np.array([s.get("fire_rate_pct", 0.0) for s in feature_stats], dtype=float)
        if feature_stats else None
    )
    if fr is not None and len(fr) != len(enr):
        fr = None
    return enr, fr, expected, n_features, k


def classify(enr, fr, expected, *, p_thresh, fold_min, dead_frac):
    """Call the REFERENCE _classify with the three thresholds overridden, collapse to 3 bins."""
    orig_p, orig_dead, orig_sig = tx.P_THRESH, tx.DEAD_FIRE_FRAC, tx._sig_cats
    try:
        tx.P_THRESH = p_thresh
        tx.DEAD_FIRE_FRAC = dead_frac
        tx._sig_cats = lambda feat, field, fold_min=fold_min: orig_sig(feat, field, fold_min)
        fr5 = tx._classify(enr, fire_rates_pct=fr, expected_rate_pct=expected)
    finally:
        tx.P_THRESH, tx.DEAD_FIRE_FRAC, tx._sig_cats = orig_p, orig_dead, orig_sig
    return {
        "sep": fr5["separable"],
        "ent": fr5["entangled"],
        "dead": fr5["dead"] + fr5["inactive"] + fr5["uncharacterized"],
    }


DEFAULT = dict(p_thresh=0.05, fold_min=1.0, dead_frac=0.25)
SWEEP = {
    "p_thresh":  [0.01, 0.05, 0.10],
    "fold_min":  [1.0, 1.5, 2.0],
    "dead_frac": [0.10, 0.25, 0.50],
}


def main() -> None:
    lines: list[str] = []
    lines.append("# T1 — Taxonomy threshold-sensitivity sweep\n")
    lines.append(
        "Fractions are **Separable / Entangled / Dead** (Dead = dead + uncharacterized + "
        "inactive). Rule reproduced verbatim from `plot_taxonomy_expansion._classify`; only "
        "the swept threshold is overridden per column, the other two held at default "
        "(p<0.05, fold>1.0, dead<0.25x TopK rate).\n"
    )

    cell_data = {}
    for name, exp in CELLS:
        enr, fr, expected, n_features, k = load_cell(exp)
        cell_data[name] = (enr, fr, expected, n_features, k)

        # Validate: our default recompute must equal a direct call to the reference
        # _classify (i.e. no monkeypatch leakage / restoration is clean).
        ref5 = tx._classify(enr, fire_rates_pct=fr, expected_rate_pct=expected)
        ref = {
            "sep": ref5["separable"],
            "ent": ref5["entangled"],
            "dead": ref5["dead"] + ref5["inactive"] + ref5["uncharacterized"],
        }
        ours = classify(enr, fr, expected, **DEFAULT)
        for kk in ("sep", "ent", "dead"):
            assert abs(ours[kk] - ref[kk]) < 1e-12, (
                f"{exp} MISMATCH {kk}: ours={ours[kk]:.6f} ref={ref[kk]:.6f}"
            )
        lines.append(
            f"\n## {name}  (`{exp}`, N={n_features}, k={k})  "
            f"— default: sep {ours['sep']:.2f} / ent {ours['ent']:.2f} / dead {ours['dead']:.2f}"
            f"  ✓ reproduces reference `_classify`\n"
        )
        for param, vals in SWEEP.items():
            lines.append(f"\n**Vary {param}** (others at default):\n")
            lines.append("| " + param + " | Separable | Entangled | Dead |")
            lines.append("|---|---|---|---|")
            for v in vals:
                kw = dict(DEFAULT)
                kw[param] = v
                f = classify(enr, fr, expected, **kw)
                star = "  ← default" if abs(v - DEFAULT[param]) < 1e-9 else ""
                lines.append(
                    f"| {v}{star} | {f['sep']:.3f} | {f['ent']:.3f} | {f['dead']:.3f} |"
                )
        lines.append("")

    md_path = OUT_DIR / "taxonomy_sensitivity.md"
    md_path.write_text("\n".join(lines))
    print(f"[T1] wrote {md_path}")

    _make_figure(cell_data)


def _make_figure(cell_data) -> None:
    from mecheeg.nature_fig_style import (  # noqa: E402
        set_nature_style, nature_figure, nature_palette, save_nature_figure,
    )

    set_nature_style()
    pal = nature_palette(3)
    fig, axes = nature_figure(width="double", height_mm=55, ncols=3)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    params = ["p_thresh", "fold_min", "dead_frac"]
    xlabels = {"p_thresh": "p threshold", "fold_min": "fold-change min", "dead_frac": "dead fire frac"}
    for ax, param in zip(axes, params):
        vals = SWEEP[param]
        for ci, (name, (enr, fr, expected, _, _)) in enumerate(cell_data.items()):
            seps, ents, deads = [], [], []
            for v in vals:
                kw = dict(DEFAULT)
                kw[param] = v
                f = classify(enr, fr, expected, **kw)
                seps.append(f["sep"]); ents.append(f["ent"]); deads.append(f["dead"])
            ax.plot(vals, seps, "-o", color=pal[ci], ms=3, lw=1.0,
                    label=name if param == "p_thresh" else None)
            ax.plot(vals, ents, "--s", color=pal[ci], ms=3, lw=1.0, alpha=0.7)
        ax.axvline(DEFAULT[param], color="0.6", lw=0.6, ls=":")
        ax.set_xlabel(xlabels[param])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction")
    axes[0].legend(frameon=False, fontsize=5, loc="upper right")
    fig.suptitle("Taxonomy fractions vs. thresholds  (solid = Separable, dashed = Entangled)",
                 fontsize=6)
    out = OUT_DIR / "taxonomy_sensitivity.pdf"
    save_nature_figure(fig, str(out))
    save_nature_figure(fig, str(out.with_suffix(".png")))
    print(f"[T1] wrote {out} (+.png)")


if __name__ == "__main__":
    main()

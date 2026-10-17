"""Clean taxonomy figure: expansion sweep + co-occurrence heatmap.

Two-panel figure per encoder (SleepFM finetuned layer 2, REVE finetuned layer 21):
  Left  — Line chart: Separable / Entangled / Spurious fraction vs expansion ratio.
  Right — Cross-field co-occurrence heatmap at E=1.

Saves:
    results/monosemanticity/taxonomy_expansion.pdf/.png         (SleepFM)
    results/monosemanticity/taxonomy_expansion_reve.pdf/.png    (REVE)

Usage::

    uv run tools/plot_taxonomy_expansion.py [--encoder sleepfm|reve|both]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

ROOT          = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR   = ROOT / "results" / "experiments"
OUT_DIR       = ROOT / "results" / "monosemanticity"
BASELINE_DIR  = ROOT / "results" / "monosemanticity"  # embedding baselines live here

# Default 4-field set for BrainCapture-trained encoders (paper). Each per-encoder
# block in SWEEPS may override via `fields` + `field_labels` keys.
FIELDS = ["age_group", "gender", "indication_group", "medication_group"]
FIELD_LABELS = {"age_group": "Age", "gender": "Sex",
                "indication_group": "Indication", "medication_group": "Medication"}

# MESA / SleepFM v2 fields — 6-field taxonomy.
# Default set: Age / Sex / OSA / Sleep stage / Race / has_respiratory_event.
# Rationale (2026-05-25 audit of feature_meta_enrichment at L=1):
#   sleep_medication had 0/128 BH-sig hits (silently dropping a slot);
#   replaced with race (48/128 hits) + the OSA-event flag has_respiratory_event
#   (53/128 hits). mesa_site (84% hits) is deliberately excluded as a
#   site/scanner confound.
MESA_FIELDS = ["age_group", "gender", "osa_severity", "sleep_stage",
               "race", "has_respiratory_event"]
MESA_FIELD_LABELS = {
    "age_group":                "Age",
    "gender":                   "Sex",
    "osa_severity":             "OSA",
    "sleep_stage":              "Stage",
    "sleep_medication":         "SleepMed",
    "race":                     "Race",
    "mesa_site":                "Site",
    "arousal_index_group":      "ArI",
    "spindle_density_group":    "SpindleDens",
    "slow_wave_density_group":  "SWDens",
    "so_spindle_coupling_group":"SO-Sp coupling",
    "has_arousal":              "Arousal",
    "has_spindle":              "Spindle",
    "has_slow_wave":            "SlowWave",
    "has_eye_movement":         "EyeMov",
    "has_respiratory_event":    "Resp",
    "has_so_up_state":          "SO up-state",
}
# Symmetric taxonomy: any 2+ fields BH-sig → "entangled" (no cluster distinction).
# Subject-level Cramér's V across these 4 fields is ≤0.36 in both encoders'
# datasets, so no pair is meaningfully more "clustered" than the others —
# justifies treating them symmetrically.
P_THRESH = 0.05
AGE_COLLAPSE = {
    "0-3": "child", "4-9": "child", "10-19": "child",
    "20-29": "adult", "30-39": "adult", "40-49": "adult", "50-59": "adult", "60+": "adult",
    # MESA age bins (54-95y cohort) — collapsed into mid/elder/oldest.
    "55-64": "mid", "65-74": "elder", "75+": "oldest",
}

COL_SEP    = "#2ca02c"
COL_ENT    = "#ff7f0e"
COL_DEAD   = "#aaaaaa"  # truly silent — low firing rate (SAE) or no raw signal (embedding)
COL_UNCHAR = "#7777aa"  # active but not field-selective — sub-BH or unmeasured concept
GRAY       = "#888888"

# A dim/feature is "dead" if it fires below this fraction of the uniform TopK
# expectation (k/n). 0.25 means "fires at <¼ of the rate a random feature
# would under TopK" — captures genuinely under-used features at high E.
DEAD_FIRE_FRAC = 0.25

SWEEPS = {
    "sleepfm": {
        "experiments": [
            ("sleepfm_finetuned_layer2",       1),
            ("sleepfm_finetuned_exp2_layer2",  2),
            ("sleepfm_finetuned_exp4_layer2",  4),
            ("sleepfm_finetuned_exp8_layer2",  8),
            ("sleepfm_finetuned_exp16_layer2", 16),
            ("sleepfm_finetuned_exp32_layer2", 32),
            ("sleepfm_finetuned_exp64_layer2", 64),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (SleepFM finetuned, layer 2, fixed k=8)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 2)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16, 32, 64],
        "xlim":     (0.75, 180),
        "out_stem": "taxonomy_expansion",
        "annot_spur_xytext":  (16, 4),
        "annot_ent_xytext":   (40, 58),
        "annot_ent_xy_idx":   4,   # index into expansions for arrow tip
        "baseline_experiment": "sleepfm_finetuned_layer2",
    },
    "sleepfm_kscaled": {
        "experiments": [
            ("sleepfm_finetuned_layer2",            1),    # E=1 baseline (k=8 already correct)
            ("sleepfm_finetuned_exp2_k16_layer2",   2),
            ("sleepfm_finetuned_exp4_k32_layer2",   4),
            ("sleepfm_finetuned_exp8_k64_layer2",   8),
            ("sleepfm_finetuned_exp16_k128_layer2", 16),
            ("sleepfm_finetuned_exp32_k256_layer2", 32),
            ("sleepfm_finetuned_exp64_k512_layer2", 64),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (SleepFM finetuned, layer 2, k=8·E, k/n=6.25%)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 2, k/n=6.25%)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16, 32, 64],
        "xlim":     (0.75, 180),
        "out_stem": "taxonomy_expansion_kscaled",
        "annot_spur_xytext":  (16, 4),
        "annot_ent_xytext":   (40, 58),
        "annot_ent_xy_idx":   4,
        "baseline_experiment": "sleepfm_finetuned_layer2",
    },
    "reve": {
        "experiments": [
            ("reve_qjbe08_layer21",        1),
            ("reve_qjbe08_exp2_layer21",   2),
            ("reve_qjbe08_exp4_layer21",   4),
            ("reve_qjbe08_exp8_layer21",   8),
            ("reve_qjbe08_exp16_layer21",  16),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (REVE finetuned, layer 21)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 21)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16],
        "xlim":     (0.75, 45),
        "out_stem": "taxonomy_expansion_reve",
        "annot_spur_xytext":  None,              # spurious ~0% — skip annotation
        "annot_ent_xytext":   (7, 38),
        "annot_ent_xy_idx":   1,                 # E=2 (23%)
        "baseline_experiment": "reve_qjbe08_layer21",
    },
    "reve_l8": {
        "experiments": [
            ("reve_qjbe08_layer8",        1),
            ("reve_qjbe08_exp2_layer8",   2),
            ("reve_qjbe08_exp4_layer8",   4),
            ("reve_qjbe08_exp8_layer8",   8),
            ("reve_qjbe08_exp16_layer8",  16),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (REVE finetuned, layer 8)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 8)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16],
        "xlim":     (0.75, 45),
        "out_stem": "taxonomy_expansion_reve_l8",
        "annot_spur_xytext":  None,
        "annot_ent_xytext":   (7, 38),
        "annot_ent_xy_idx":   1,
        "baseline_experiment": "reve_qjbe08_layer8",
    },
    "sleepfm_v2": {
        # MESA + SleepFM v2 (MAE+CL). Per-layer block populated post-Phase-C
        # so the headline-layer choice is data-driven. The script also
        # supports expansion sweeps once a headline layer is fixed (see
        # mesa_sleepfm_v2_plan.md §4 Phase D).
        "experiments": [
            ("sleepfm_v2_layer0", 1),
            ("sleepfm_v2_layer1", 1),
            ("sleepfm_v2_layer2", 1),
            ("sleepfm_v2_layer3", 1),
        ],
        "title":      "Monosemanticity taxonomy across layers  (SleepFM v2 MAE+CL, MESA, E=1, k=8)",
        "cooc_title": "Cross-field co-occurrence  (E=1)\nDiagonal = single-field prevalence",
        "xticks":     [1],
        "xlim":       (0.75, 2.0),
        "out_stem":   "10_taxonomy_expansion_perlayer",
        "out_dir":    "results/mesa",
        "annot_spur_xytext":  None,
        "annot_ent_xytext":   (1.1, 38),
        "annot_ent_xy_idx":   0,
        "baseline_experiment": "sleepfm_v2_layer3",
        # 5-field taxonomy: Age / Sex / OSA / Sleep stage / Race.
        "fields":       MESA_FIELDS,
        "field_labels": MESA_FIELD_LABELS,
    },
    "sleepfm_v2_kscaled": {
        # MESA + SleepFM v2, k-scaled expansion sweep at L=1 (k = 8·E,
        # keeping k/n = 6.25% constant). Matches the paper's main SleepFM
        # finetuned panel and removes the dead-neuron explosion artefact
        # that fixed-k sweeps produce at high E. Headline figure.
        "experiments": [
            ("sleepfm_v2_layer1",                1),    # E=1 baseline (k=8)
            ("sleepfm_v2_exp2_k16_layer1",       2),
            ("sleepfm_v2_exp4_k32_layer1",       4),
            ("sleepfm_v2_exp8_k64_layer1",       8),
            ("sleepfm_v2_exp16_k128_layer1",     16),
            ("sleepfm_v2_exp32_k256_layer1",     32),
            ("sleepfm_v2_exp64_k512_layer1",     64),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (SleepFM v2 MAE+CL, MESA, layer 1, k=8·E, k/n=6.25%)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 1)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16, 32, 64],
        "xlim":     (0.75, 80.0),
        "out_stem": "09_taxonomy_expansion_scaled",
        "out_dir":  "results/mesa",
        "annot_spur_xytext":  None,
        "annot_ent_xytext":   (1.1, 38),
        "annot_ent_xy_idx":   0,
        "baseline_experiment": "sleepfm_v2_layer1",
        "fields":       MESA_FIELDS,
        "field_labels": MESA_FIELD_LABELS,
    },
    "labram": {
        "experiments": [
            ("labram_layer11",        1),
            ("labram_layer11_exp2",   2),
            ("labram_layer11_exp4",   4),
            ("labram_layer11_exp8",   8),
            ("labram_layer11_exp16",  16),
        ],
        "title":    "Monosemanticity taxonomy vs. expansion  (LaBraM finetuned, layer 11)",
        "cooc_title": "Cross-field co-occurrence  (E=1, layer 11)\nDiagonal = single-field prevalence",
        "xticks":   [1, 2, 4, 8, 16],
        "xlim":     (0.75, 45),
        "out_stem": "taxonomy_expansion_labram",
        "annot_spur_xytext":  None,
        "annot_ent_xytext":   (7, 38),
        "annot_ent_xy_idx":   1,
        "baseline_experiment": "labram_layer11",
    },
}

BASELINE_X = 0.5  # x-position for the embedding baseline column on the log axis


# ── helpers ──────────────────────────────────────────────────────────────────

def _sig_cats(feat: dict, field: str, fold_min: float = 1.0) -> set[str]:
    sig: set[str] = set()
    for cat, fold, p in feat.get(field, []):
        if p < P_THRESH and fold > fold_min:  # only enriched categories
            # For binary event fields (yes/no), skip "no" — it is the trivial
            # complement and marks only anti-selective features, not event features.
            if cat == "no" and field.endswith("_per_token"):
                continue
            if field == "age_group":
                c = AGE_COLLAPSE.get(cat)
                if c:
                    sig.add(c)
            else:
                sig.add(cat)
    return sig


def _classify(
    enr_list: list,
    *,
    fire_rates_pct: np.ndarray | None = None,
    expected_rate_pct: float | None = None,
    enr_raw: list | None = None,
    fields: list[str] | None = None,
) -> dict[str, float]:
    """Five-bin classification.

    Bins: separable / entangled / spurious / dead / uncharacterized.

    Bins 1-3 are determined by BH-significant per-field hits (unchanged from
    the original 4-bin taxonomy).

    "Dead" vs "uncharacterized" splits the old "inactive" bin:
      - dead: cannot be assigned to any field by any reasonable test
              · fire_rate < DEAD_FIRE_FRAC × expected_rate (when fire rates are
                provided — i.e. SAE features), OR
              · 0 fields hit even at raw p<0.05 (when raw enrichment is
                provided — i.e. embedding baseline under top-k binarization,
                where every dim has constant fire_rate=k/D).
      - uncharacterized: 0 fields BH-significant but the dim/feature is
        clearly active (high fire rate or sub-BH multi-field signal).

    If neither fire rates nor raw enrichment are supplied, the old
    "inactive" label is reported instead (3-way split for the inactive bin
    is suppressed).
    """
    fs = fields or FIELDS
    counts = {"separable": 0, "entangled": 0,
              "dead": 0, "uncharacterized": 0, "inactive": 0}
    have_split = (
        (fire_rates_pct is not None and expected_rate_pct is not None)
        or enr_raw is not None
    )
    for i, feat in enumerate(enr_list):
        active = {f for f in fs if _sig_cats(feat, f)}
        n = len(active)
        if n == 0:
            if not have_split:
                counts["inactive"] += 1
                continue
            is_dead = False
            if fire_rates_pct is not None and expected_rate_pct is not None:
                is_dead = fire_rates_pct[i] < DEAD_FIRE_FRAC * expected_rate_pct
            elif enr_raw is not None:
                raw_active = {f for f in fs if _sig_cats(enr_raw[i], f)}
                is_dead = (len(raw_active) == 0)
            counts["dead" if is_dead else "uncharacterized"] += 1
        elif n == 1:
            counts["separable"] += 1
        else:
            counts["entangled"] += 1
    total = max(len(enr_list), 1)
    return {k: v / total for k, v in counts.items()}


def _cooccurrence(enr_list: list, fields: list[str] | None = None) -> np.ndarray:
    fs = fields or FIELDS
    n_fields = len(fs)
    active   = np.zeros((len(enr_list), n_fields), dtype=bool)
    for i, feat in enumerate(enr_list):
        for j, f in enumerate(fs):
            active[i, j] = bool(_sig_cats(feat, f))
    mat = np.zeros((n_fields, n_fields))
    for a in range(n_fields):
        for b in range(n_fields):
            either = active[:, a] | active[:, b]
            both   = active[:, a] & active[:, b]
            mat[a, b] = both.sum() / max(either.sum(), 1)
    return mat


def _load_sweep(cfg: dict) -> tuple:
    expansions, f_sep, f_ent, f_dead, f_unchar = [], [], [], [], []
    enr_ref = None
    missing = []
    cfg_fields = cfg.get("fields", FIELDS)
    for exp_name, E in cfg["experiments"]:
        cache_path = RESULTS_DIR / exp_name / "app_cache.pt"
        if not cache_path.exists():
            missing.append((exp_name, E))
            continue
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        enr   = cache.get("feature_meta_enrichment")
        if not enr:
            print(f"  [skip] {exp_name}: no feature_meta_enrichment (rebuild app cache)")
            missing.append((exp_name, E))
            continue
        feature_stats = cache.get("feature_stats") or []
        meta = cache.get("meta") or {}
        n_features = meta.get("n_features", len(enr))
        k_topk     = meta.get("k", 8)
        expected_rate_pct = 100.0 * k_topk / max(n_features, 1)
        fr = np.array(
            [s.get("fire_rate_pct", 0.0) for s in feature_stats],
            dtype=float,
        ) if feature_stats else None
        if fr is not None and len(fr) != len(enr):
            fr = None
        fracs = _classify(enr, fire_rates_pct=fr,
                          expected_rate_pct=expected_rate_pct,
                          fields=cfg_fields)
        expansions.append(E)
        f_sep.append(fracs["separable"])
        f_ent.append(fracs["entangled"])
        f_dead.append(fracs["dead"] + fracs["inactive"])
        f_unchar.append(fracs["uncharacterized"])
        if E == 1:
            enr_ref = enr
    if missing:
        print(f"  Missing/incomplete: {[e for e, _ in missing]}")
    return (np.array(expansions, dtype=float),
            np.array(f_sep), np.array(f_ent),
            np.array(f_dead), np.array(f_unchar),
            enr_ref)


def _load_baseline(cfg: dict) -> dict | None:
    """Load embedding-baseline cache produced by embedding_monosemanticity_baseline.py."""
    exp = cfg.get("baseline_experiment")
    if not exp:
        return None
    p = BASELINE_DIR / f"embedding_baseline_{exp}.pt"
    if not p.exists():
        print(f"  [info] no baseline at {p}; run embedding_monosemanticity_baseline.py first")
        return None
    bl = torch.load(p, map_location="cpu", weights_only=False)
    bf = bl["fractions"]
    dead     = bf.get("dead",            bf.get("inactive", 0.0))
    unchar   = bf.get("uncharacterized", 0.0)
    ent_total = bf["entangled"] + bf.get("spurious", 0.0)
    print(f"  Loaded baseline: {p.name}  "
          f"sep={bf['separable']:.0%}  ent={ent_total:.0%}  "
          f"dead={dead:.0%}  unchar={unchar:.0%}")
    return bl


def _plot_encoder(cfg: dict, with_baseline: bool = False) -> None:
    print(f"\nLoading sweep: {cfg['out_stem']}")
    (expansions, f_sep, f_ent, f_dead, f_unchar,
     enr_ref) = _load_sweep(cfg)
    if len(expansions) == 0:
        print("  No data — skipping.")
        return

    print("Expansion sweep:")
    for E, s, e, d, u in zip(expansions, f_sep, f_ent, f_dead, f_unchar):
        print(f"  E={int(E):3d}: sep={s:.0%}  ent={e:.0%}  "
              f"dead={d:.0%}  unchar={u:.0%}")

    baseline = _load_baseline(cfg) if with_baseline else None

    if baseline is not None:
        fig = plt.figure(figsize=(13, 5.2))
        gs  = fig.add_gridspec(1, 3, width_ratios=[3, 1.6, 1.6], wspace=0.42)
        ax_l  = fig.add_subplot(gs[0, 0])
        ax_m  = fig.add_subplot(gs[0, 1])
        ax_r  = fig.add_subplot(gs[0, 2])
        heatmap_axes = (ax_m, ax_r)
    else:
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(11, 5.2),
            gridspec_kw={"width_ratios": [3, 2], "wspace": 0.38},
        )
        heatmap_axes = (ax_r,)

    fig.patch.set_facecolor("white")
    for ax in (ax_l, *heatmap_axes):
        ax.set_facecolor("white")

    ax_l.plot(expansions, f_sep    * 100, "o-",  color=COL_SEP,    lw=2.2, ms=7, zorder=3,
              label="Separable (1 field)")
    ax_l.plot(expansions, f_ent    * 100, "s-",  color=COL_ENT,    lw=2.2, ms=7, zorder=3,
              label="Entangled (≥2 fields)")
    ax_l.plot(expansions, f_dead   * 100, "D--", color=COL_DEAD,   lw=2.2, ms=6, zorder=2,
              label="Dead (fire rate < ¼·k/n)")
    ax_l.plot(expansions, f_unchar * 100, "v:",  color=COL_UNCHAR, lw=2.0, ms=6, zorder=2,
              label="Uncharacterized (active, no field)")

    if baseline is not None:
        bf = baseline["fractions"]
        bf_dead   = bf.get("dead",            bf.get("inactive", 0.0))
        bf_unchar = bf.get("uncharacterized", 0.0)
        # Symmetric taxonomy: collapse legacy spurious into entangled
        bf_ent_total = bf["entangled"] + bf.get("spurious", 0.0)
        # Discrete reference column on the log axis (left of E=1)
        ax_l.plot([BASELINE_X], [bf["separable"] * 100], "o", color=COL_SEP, ms=8, zorder=4,
                  markeredgecolor="white", markeredgewidth=1.2)
        ax_l.plot([BASELINE_X], [bf_ent_total    * 100], "s", color=COL_ENT, ms=8, zorder=4,
                  markeredgecolor="white", markeredgewidth=1.2)
        ax_l.plot([BASELINE_X], [bf_dead         * 100], "D", color=COL_DEAD, ms=7, zorder=3,
                  markeredgecolor="white", markeredgewidth=1.2)
        ax_l.plot([BASELINE_X], [bf_unchar       * 100], "v", color=COL_UNCHAR, ms=7, zorder=3,
                  markeredgecolor="white", markeredgewidth=1.2)
        # Dotted connectors from baseline to E=1, showing what the SAE buys you
        for y_bl, y_e1, col in (
            (bf["separable"] * 100, f_sep[0]    * 100, COL_SEP),
            (bf_ent_total    * 100, f_ent[0]    * 100, COL_ENT),
            (bf_dead         * 100, f_dead[0]   * 100, COL_DEAD),
            (bf_unchar       * 100, f_unchar[0] * 100, COL_UNCHAR),
        ):
            ax_l.plot([BASELINE_X, expansions[0]], [y_bl, y_e1],
                      ":", color=col, lw=1.2, alpha=0.55, zorder=1)
        # Visual separator between baseline and SAE sweep
        sep_x = (BASELINE_X * expansions[0]) ** 0.5  # geometric mean on log axis
        ax_l.axvline(sep_x, color=GRAY, linestyle="--", lw=0.8, alpha=0.55, zorder=0)
        ax_l.text(BASELINE_X, 67, "raw\nembedding", ha="center", va="top",
                  fontsize=8, color="#555555", fontstyle="italic")

    ent_idx = min(cfg["annot_ent_xy_idx"], len(expansions) - 1)
    ax_l.annotate(
        "partially biological\n(slow decline)",
        xy=(expansions[ent_idx], f_ent[ent_idx] * 100), xytext=cfg["annot_ent_xytext"],
        ha="center", fontsize=8, color=COL_ENT,
        arrowprops=dict(arrowstyle="-|>", color=COL_ENT, lw=1.0),
    )

    ax_l.set_xscale("log", base=2)
    ax_l.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_l.set_xticks(cfg["xticks"])
    ax_l.set_xlabel("SAE expansion ratio  (E)", fontsize=10, color="#333333")
    ax_l.set_ylabel("% of features", fontsize=10, color="#333333")
    ax_l.set_ylim(0, 70)
    if baseline is not None:
        # Extend lower bound to fit the baseline column at x=0.5
        ax_l.set_xlim(BASELINE_X / 1.4, cfg["xlim"][1])
    else:
        ax_l.set_xlim(*cfg["xlim"])
    ax_l.legend(fontsize=8.5, framealpha=0.0, facecolor="white", edgecolor="none",
                loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=3, columnspacing=1.6, handletextpad=0.6, borderaxespad=0.0)
    ax_l.set_title(cfg["title"], fontsize=9.5, color="#333333", pad=8)
    ax_l.spines[["top", "right"]].set_visible(False)
    for sp in ["left", "bottom"]:
        ax_l.spines[sp].set_color(GRAY)
    ax_l.tick_params(colors="#333333")
    ax_l.yaxis.grid(True, alpha=0.25, linestyle="--", color=GRAY)
    ax_l.set_axisbelow(True)
    ax_l.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    cfg_fields = cfg.get("fields", FIELDS)
    cfg_field_labels = cfg.get("field_labels", FIELD_LABELS)

    def _draw_heatmap(ax, cooc_mat, title, *, attach_cbar):
        labels = [cfg_field_labels[f] for f in cfg_fields]
        im = ax.imshow(cooc_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cfg_fields)))
        ax.set_yticks(range(len(cfg_fields)))
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right", color="#333333")
        ax.set_yticklabels(labels, fontsize=9, color="#333333")
        ax.tick_params(length=0)
        for i in range(len(cfg_fields)):
            for j in range(len(cfg_fields)):
                v = cooc_mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if v > 0.55 else "#333333")
        if attach_cbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Co-activation prevalence", fontsize=8.5, color="#333333")
            cbar.ax.tick_params(colors="#333333", labelsize=8)
        ax.set_title(title, fontsize=9.5, color="#333333", pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)

    if baseline is not None:
        bl_cooc = baseline["cooccurrence"]
        bl_title = (f"Raw embedding (top-{baseline['k']})\n"
                    f"{baseline['embed_dim']} dims, {baseline['n_subjects']} subjects")
        _draw_heatmap(ax_m, bl_cooc, bl_title, attach_cbar=False)
        if enr_ref:
            sae_cooc = _cooccurrence(enr_ref, fields=cfg_fields)
            sae_title = "SAE  E=1\nDiagonal = single-field prevalence"
            _draw_heatmap(ax_r, sae_cooc, sae_title, attach_cbar=True)
        else:
            ax_r.set_visible(False)
    else:
        if enr_ref:
            _draw_heatmap(ax_r, _cooccurrence(enr_ref, fields=cfg_fields), cfg["cooc_title"], attach_cbar=True)
        else:
            ax_r.set_visible(False)

    out_dir = ROOT / cfg["out_dir"] if "out_dir" in cfg else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    stem = f"{cfg['out_stem']}_with_baseline" if baseline is not None else cfg["out_stem"]
    for suffix in (".pdf", ".png"):
        p = out_dir / f"{stem}{suffix}"
        plt.savefig(p, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved {p}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="both",
                        choices=["sleepfm", "sleepfm_kscaled", "reve", "reve_l8", "labram", "sleepfm_v2", "sleepfm_v2_kscaled", "both"])
    parser.add_argument("--with-baseline", action="store_true",
                        help="Add raw-embedding baseline column + heatmap "
                             "(requires embedding_baseline_<exp>.pt in results/monosemanticity/)")
    args = parser.parse_args()

    encoders = list(SWEEPS.keys()) if args.encoder == "both" else [args.encoder]
    for enc in encoders:
        _plot_encoder(SWEEPS[enc], with_baseline=args.with_baseline)

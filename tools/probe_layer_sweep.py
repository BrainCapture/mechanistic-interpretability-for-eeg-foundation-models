"""Layer-by-layer SAE-faithfulness probe sweep.

Same protocol as `probe_table_finetune_sae.py`, but iterates over a list of
layers per encoder instead of just the operating layer. Output is a curve
(test AUROC vs. layer) rather than a single row.

For each (encoder, layer) we:
  1. Use the SAME finetuned encoder + the SAME baseline mean-pooled embeddings
     as in the operating-point probe.
  2. Inject the layer-`l` SAE (encode → decode in normalised activation
     space) and re-extract embeddings.
  3. Train a 3-seed linear probe and report test AUROC + Δ to baseline.

Outputs
-------
  results/probe_reconstruction/layer_sweep.json
  results/probe_reconstruction/layer_sweep.png

Usage
-----
  uv run tools/probe_layer_sweep.py
  uv run tools/probe_layer_sweep.py --encoders sleepfm reve --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse the building blocks from the headline probe script
sys.path.insert(0, str(ROOT / "tools"))
from probe_table_finetune_sae import (   # noqa: E402
    DEVICE, ENCODERS, _SAEInjector, evaluate, extract_embeddings,
    load_sae, train_probe,
)
import probe_table_finetune_sae as _probe_mod  # for `injector_target` global

from sae4eeg.dataset import StandardizeLabel, get_dataloaders
from sae4eeg.encoders import load_encoder

OUT_DIR = ROOT / "results" / "probe_reconstruction"

# ──────────────────────────────────────────────────────────────────────────────
# Layer subset per encoder + SAE checkpoint pattern
# ──────────────────────────────────────────────────────────────────────────────
SWEEP: dict[str, dict] = {
    "sleepfm": {
        "layers":     [0, 1, 2],
        "sae_dir":    "results/features/sleepfm_finetuned_local",
        "sae_pattern": "sae_sleepfm_exp1.0_k8_layer{L}.pt",
    },
    "reve": {
        "layers":     list(range(22)),  # 0..21 — full sweep (requires all SAEs trained)
        "sae_dir":    "results/features/reve_local",
        "sae_pattern": "sae_reve_exp1.0_k8_layer{L}.pt",
    },
    "labram": {
        "layers":     list(range(12)),  # 0..11 — full sweep (all SAEs already trained)
        "sae_dir":    "results/features/labram",
        "sae_pattern": "sae_labram_exp1.0_k8_layer{L}.pt",
    },
}


def run_encoder_sweep(name: str, args) -> dict:
    cfg = ENCODERS[name]
    sweep = SWEEP[name]
    print(f"\n{'='*72}\nEncoder: {name}\n  layers: {sweep['layers']}\n{'='*72}")

    weights = ROOT / cfg["weights"]
    data_path = ROOT / cfg["data"]
    splits_path = ROOT / cfg["splits"]
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. encoder
    backend = load_encoder(cfg["load_name"], weights_path=weights)
    backend.to(DEVICE).eval()

    # 2. data loaders (same splits as the operating-layer probe)
    transform = StandardizeLabel()
    folds = list(get_dataloaders(
        train_path=str(data_path),
        transformer=transform,
        batch_size=cfg["batch_size"],
        num_workers=2,
        seed=42,
        n_splits=1,
        split_info_path=str(splits_path),
    ))
    _, train_loader, val_loader, test_loader = folds[0]
    print(f"  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

    # 3. baseline (encoder-only, no SAE) — extracted once per encoder
    print("\n  → baseline")
    t0 = time.time()
    train_emb,  train_y = extract_embeddings(backend, train_loader, None, args.max_samples)
    val_emb,    val_y   = extract_embeddings(backend, val_loader,   None, args.max_samples)
    test_emb,   test_y  = extract_embeddings(backend, test_loader,  None, args.max_samples)
    print(f"    embeddings extracted ({time.time()-t0:.0f}s)  "
          f"train={tuple(train_emb.shape)} test={tuple(test_emb.shape)}")

    seed_metrics_b = []
    for s in range(args.seeds):
        head = train_probe(
            train_emb, train_y, val_emb, val_y,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.probe_batch_size, seed=s,
        )
        seed_metrics_b.append(evaluate(head, test_emb, test_y))
    base_auc_mean = round(statistics.mean(m["auc"] for m in seed_metrics_b), 4)
    base_auc_std  = round(statistics.stdev(m["auc"] for m in seed_metrics_b)
                           if args.seeds > 1 else 0.0, 4)
    base_ci_lo    = round(statistics.mean(m["auc_ci_lo"] for m in seed_metrics_b), 4)
    base_ci_hi    = round(statistics.mean(m["auc_ci_hi"] for m in seed_metrics_b), 4)
    print(f"    AUROC (baseline) = {base_auc_mean:.4f} ± {base_auc_std:.4f}  "
          f"95% boot CI [{base_ci_lo:.4f}, {base_ci_hi:.4f}]")

    # 4. one SAE-injected condition per layer
    layer_results: dict[int, dict] = {}
    for layer in sweep["layers"]:
        sae_path = ROOT / sweep["sae_dir"] / sweep["sae_pattern"].format(L=layer)
        if not sae_path.exists():
            print(f"  ⚠ skipping layer {layer}: missing {sae_path.relative_to(ROOT)}")
            continue
        print(f"\n  → sae_layer{layer}  ({sae_path.name})")
        sae, act_mean, act_std = load_sae(sae_path)
        injector = _SAEInjector(sae, act_mean, act_std, DEVICE)
        _probe_mod.injector_target = layer

        t0 = time.time()
        tr_emb,  tr_y = extract_embeddings(backend, train_loader, injector, args.max_samples)
        va_emb,  va_y = extract_embeddings(backend, val_loader,   injector, args.max_samples)
        te_emb,  te_y = extract_embeddings(backend, test_loader,  injector, args.max_samples)
        print(f"    embeddings extracted ({time.time()-t0:.0f}s)")

        seed_metrics = []
        for s in range(args.seeds):
            head = train_probe(
                tr_emb, tr_y, va_emb, va_y,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.probe_batch_size, seed=s,
            )
            seed_metrics.append(evaluate(head, te_emb, te_y))
        m_auc  = round(statistics.mean(m["auc"]  for m in seed_metrics), 4)
        s_auc  = round(statistics.stdev(m["auc"] for m in seed_metrics)
                        if args.seeds > 1 else 0.0, 4)
        m_aupr = round(statistics.mean(m["auprc"] for m in seed_metrics), 4)
        m_ba   = round(statistics.mean(m["ba"]    for m in seed_metrics), 4)
        m_ci_lo = round(statistics.mean(m["auc_ci_lo"] for m in seed_metrics), 4)
        m_ci_hi = round(statistics.mean(m["auc_ci_hi"] for m in seed_metrics), 4)
        delta  = round(base_auc_mean - m_auc, 4)
        print(f"    AUROC = {m_auc:.4f} ± {s_auc:.4f}  "
              f"95% boot CI [{m_ci_lo:.4f}, {m_ci_hi:.4f}]  Δ = {delta:+.4f}")
        layer_results[layer] = {
            "auc_mean":   m_auc, "auc_std": s_auc,
            "auc_ci_lo":  m_ci_lo, "auc_ci_hi": m_ci_hi,
            "auprc_mean": m_aupr,
            "ba_mean":    m_ba,
            "delta_auc":  delta,
        }

    return {
        "baseline": {"auc_mean": base_auc_mean, "auc_std": base_auc_std,
                     "auc_ci_lo": base_ci_lo,   "auc_ci_hi": base_ci_hi},
        "by_layer": layer_results,
        "n_layers_total": len(backend.get_hookable_layers()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+",
                   default=list(SWEEP.keys()),
                   choices=list(SWEEP.keys()))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def plot(summary: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"sleepfm": "#1f77b4", "reve": "#ff7f0e", "labram": "#2ca02c"}
    n_total = {n: r["n_layers_total"] for n, r in summary.items()}

    for name, res in summary.items():
        layers = sorted(int(l) for l in res["by_layer"])

        def _by(L, key):
            entry = res["by_layer"][str(L) if str(L) in res["by_layer"] else L]
            return entry[key]

        aucs    = [_by(L, "auc_mean")  for L in layers]
        ci_lo   = [_by(L, "auc_ci_lo") for L in layers]
        ci_hi   = [_by(L, "auc_ci_hi") for L in layers]
        # x-axis: relative layer position so the three encoders share one curve scale
        xs = [L / max(n_total[name] - 1, 1) for L in layers]
        ax.plot(xs, aucs, "o-", color=colors[name],
                label=f"{name} (L*={cfg_op(name)})")
        ax.fill_between(xs, ci_lo, ci_hi, color=colors[name], alpha=0.15,
                        linewidth=0)
        # baseline (no SAE) — point estimate + shaded CI
        b = res["baseline"]
        ax.axhline(b["auc_mean"], color=colors[name],
                   linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhspan(b["auc_ci_lo"], b["auc_ci_hi"], color=colors[name],
                   alpha=0.05, linewidth=0)

    ax.set_xlabel("Relative layer (0 = first, 1 = last)")
    ax.set_ylabel("Test AUROC after SAE substitution")
    ax.set_title("SAE-faithfulness layer sweep (95% bootstrap CI shaded)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "layer_sweep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nWrote plot → {out.relative_to(ROOT)}")


def cfg_op(name: str) -> int:
    """Operating layer per encoder (for plot labels only)."""
    return {"sleepfm": 2, "reve": 8, "labram": 11}[name]


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}\nEncoders: {args.encoders}")

    summary: dict = {}
    for name in args.encoders:
        summary[name] = run_encoder_sweep(name, args)
        # checkpoint after each encoder so partial results are persisted
        out_path = OUT_DIR / "layer_sweep.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote partial summary to {out_path.relative_to(ROOT)}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY (test AUROC, mean ± std over seeds)")
    print("=" * 72)
    for name, res in summary.items():
        b = res["baseline"]
        print(f"\n{name}: baseline {b['auc_mean']:.4f} ± {b['auc_std']:.4f}  "
              f"(of {res['n_layers_total']} layers)")
        for layer, r in sorted(res["by_layer"].items(), key=lambda kv: int(kv[0])):
            print(f"  layer {int(layer):2d}  AUROC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
                  f"Δ = {r['delta_auc']:+.4f}")

    try:
        plot(summary)
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()

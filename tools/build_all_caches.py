"""build_all_caches.py — Orchestrate the full Codebook + App-Cache pipeline.

For every discovered SAE run (encoder × variant × layer) that does not yet
have an app_cache.pt, this script runs:

  1. build_codebook.py      (once per encoder × variant — skipped if exists)
  2. explain_features.py    (once per encoder × variant × layer — skipped if exists)
  3. Writes metadata.json   (once per experiment — auto-generated from checkpoint)
  4. build_app_cache.py     (once per experiment)

The app expects caches at:
  results/experiments/{encoder}_{variant}_layer{L}/app_cache.pt

Usage
-----
  # All discovered runs (last layer only by default):
  uv run tools/build_all_caches.py

  # All layers for a specific run:
  uv run tools/build_all_caches.py --encoder sleepfm --variant pretrained --all-layers

  # Single specific combination:
  uv run tools/build_all_caches.py --encoder sleepfm --variant finetuned --layer 2

  # Force rebuild even if cache exists:
  uv run tools/build_all_caches.py --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent

_ENCODER_DATA = {
    "sleepfm": "data/D4-v3-preprocessed-v2",
    "reve":    "data/D4-v3-preprocessed-v1",
}
_ENCODER_FS = {"sleepfm": 128, "reve": 200}
_ENCODER_EMBED = {"sleepfm": 128, "reve": 512}

_DEFAULT_WEIGHTS = {
    "sleepfm": str(ROOT / "checkpoints" / "pretrained" / "sleepfm_weights.pt"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Discovery (mirrors app/main.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_encoder(folder_name: str) -> str:
    return "reve" if folder_name.startswith("reve") else "sleepfm"


def discover_runs():
    """Return list of (encoder, variant_label, tag, weights_path, layers, folder_name)."""
    features_root = ROOT / "results" / "features"
    # {(encoder, folder_name): [layers]}
    found = {}

    for folder in sorted(features_root.iterdir()):
        if not folder.is_dir():
            continue
        ckpts = sorted(folder.glob("sae_*.pt"))
        if not ckpts:
            continue

        encoder = _parse_encoder(folder.name)
        tag: Optional[str] = None
        encoder_weights: Optional[str] = None
        try:
            meta = torch.load(str(ckpts[0]), map_location="cpu", weights_only=False)
            encoder        = meta.get("encoder", encoder)
            tag            = meta.get("tag")
            encoder_weights = meta.get("encoder_weights")
        except Exception:
            pass

        is_finetuned = (
            "finetuned" in folder.name
            or tag == "finetuned"
            or (encoder_weights is not None and encoder_weights.endswith(".ckpt"))
            or (tag is not None and tag not in ("pretrained", "base"))
        )
        variant = "finetuned" if is_finetuned else "pretrained"

        layers = []
        for ckpt in ckpts:
            try:
                layers.append(int(ckpt.stem.split("_layer")[1]))
            except (IndexError, ValueError):
                continue

        key = (encoder, folder.name)
        found[key] = {
            "encoder":        encoder,
            "variant":        variant,
            "tag":            tag,
            "encoder_weights": encoder_weights,
            "layers":         sorted(set(layers)),
            "folder_name":    folder.name,
        }

    return list(found.values())


def _xae_path(folder_name: str) -> Optional[Path]:
    candidates = [
        ROOT / "results" / "xae" / folder_name / "xae_checkpoint.pt",
        ROOT / "results" / "xae" / "xae_checkpoint.pt",
    ]
    return next((p for p in candidates if p.exists()), None)


def _codebook_path(folder_name: str) -> Path:
    return ROOT / "results" / "xae" / folder_name / "codebook" / "codebook.pt"


def _explanations_path(folder_name: str) -> Path:
    return ROOT / "results" / "xae" / folder_name / "explanations" / "feature_explanations.json"


def _exp_name(folder_name: str, layer: int) -> str:
    return f"{folder_name}_layer{layer}"


def _app_cache_path(folder_name: str, layer: int) -> Path:
    return ROOT / "results" / "experiments" / _exp_name(folder_name, layer) / "app_cache.pt"


def _metadata_path(folder_name: str, layer: int) -> Path:
    return ROOT / "results" / "experiments" / _exp_name(folder_name, layer) / "metadata.json"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], step: str):
    print(f"\n{'─'*60}")
    print(f"  {step}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [ERROR] {step} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def build_codebook(run_info: dict, force: bool = False):
    folder_name = run_info["folder_name"]
    encoder     = run_info["encoder"]
    tag         = run_info["tag"]
    weights     = run_info["encoder_weights"]

    out = _codebook_path(folder_name)
    if out.exists() and not force:
        print(f"  [skip] Codebook already exists: {out.relative_to(ROOT)}")
        return

    cmd = [
        "uv", "run", "tools/build_codebook.py",
        "--encoder", encoder,
    ]
    if tag:
        cmd += ["--tag", tag]
    if weights:
        cmd += ["--weights-path", weights]
    elif encoder == "sleepfm" and _DEFAULT_WEIGHTS.get(encoder):
        cmd += ["--weights-path", _DEFAULT_WEIGHTS[encoder]]

    run(cmd, f"Build codebook: {folder_name}")


def run_explain_features(run_info: dict, layer: int, force: bool = False):
    folder_name = run_info["folder_name"]
    encoder     = run_info["encoder"]
    tag         = run_info["tag"]
    weights     = run_info["encoder_weights"]

    out = _explanations_path(folder_name)
    if out.exists() and not force:
        print(f"  [skip] Explanations already exist: {out.relative_to(ROOT)}")
        return

    cmd = [
        "uv", "run", "tools/explain_features.py",
        "--encoder", encoder,
        "--layer", str(layer),
    ]
    if tag:
        cmd += ["--tag", tag]
    if weights:
        cmd += ["--weights-path", weights]
    elif encoder == "sleepfm" and _DEFAULT_WEIGHTS.get(encoder):
        cmd += ["--weights-path", _DEFAULT_WEIGHTS[encoder]]

    run(cmd, f"Explain features: {folder_name} layer {layer}")


def _resolve_weights(weights: Optional[str]) -> Optional[str]:
    """Return a ROOT-relative path for encoder weights, searching subdirs if needed."""
    if not weights:
        return None
    p = Path(weights)
    # Absolute path: make relative if under ROOT
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    # Relative path: check as-is first, then under checkpoints/finetuned|pretrained
    if (ROOT / p).exists():
        return str(p)
    name = p.name
    for subdir in ("finetuned", "pretrained"):
        candidate = ROOT / "checkpoints" / subdir / name
        if candidate.exists():
            return str(candidate.relative_to(ROOT))
    return str(p)  # best effort — will surface a clear error at runtime


def write_metadata(run_info: dict, layer: int):
    folder_name = run_info["folder_name"]
    encoder     = run_info["encoder"]
    weights     = run_info["encoder_weights"]

    # Read embed_dim, expansion, k from the SAE checkpoint
    sae_glob = list(
        (ROOT / "results" / "features" / folder_name)
        .glob(f"sae_{encoder}_*_layer{layer}.pt")
    )
    if not sae_glob:
        print(f"  [warn] No SAE checkpoint found for {folder_name} layer {layer}")
        return
    sae_path = sae_glob[0]
    ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)

    embed_dim  = ckpt.get("embed_dim",   _ENCODER_EMBED.get(encoder, 128))
    expansion  = ckpt.get("expansion",   1.0)
    k          = ckpt.get("k",           8)
    n_features = int(embed_dim * float(expansion))
    fs         = _ENCODER_FS.get(encoder, 128)
    patch_size = fs

    xae_p      = _xae_path(folder_name)
    codebook_p = _codebook_path(folder_name)
    expl_p     = _explanations_path(folder_name)
    exp_name   = _exp_name(folder_name, layer)

    meta = {
        "name":          exp_name,
        "display_name":  (
            f"{encoder.upper()} {'finetuned' if run_info['variant'] == 'finetuned' else 'pretrained'}"
            f" · layer {layer}"
        ),
        "encoder":       encoder,
        "embed_dim":     embed_dim,
        "fs":            fs,
        "patch_size":    patch_size,
        "target_layer":  layer,
        "expansion":     expansion,
        "k":             k,
        "n_features":    n_features,
        "n_clusters":    200,
        "sae_checkpoint":   str(sae_path.relative_to(ROOT)),
        "xae_checkpoint":   str(xae_p.relative_to(ROOT)) if xae_p else None,
        "codebook_path":    str(codebook_p.relative_to(ROOT)) if codebook_p.exists() else None,
        "feature_explanations": (
            str(expl_p.relative_to(ROOT)) if expl_p.exists() else None
        ),
        "weights_path": _resolve_weights(weights),
    }

    meta_p = _metadata_path(folder_name, layer)
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote metadata → {meta_p.relative_to(ROOT)}")


def build_app_cache(run_info: dict, layer: int, force: bool = False):
    folder_name = run_info["folder_name"]
    exp_name    = _exp_name(folder_name, layer)
    cache_p     = _app_cache_path(folder_name, layer)

    if cache_p.exists() and not force:
        print(f"  [skip] App cache already exists: {cache_p.relative_to(ROOT)}")
        return

    run(
        ["uv", "run", "tools/build_app_cache.py", "--experiment", exp_name],
        f"Build app cache: {exp_name}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate full Codebook + App-Cache pipeline")
    p.add_argument("--encoder",  default=None, help="Filter by encoder (sleepfm|reve)")
    p.add_argument("--variant",  default=None, help="Filter by variant (pretrained|finetuned)")
    p.add_argument("--layer",    type=int, default=None,
                   help="Build only this layer (default: last layer per run)")
    p.add_argument("--all-layers", action="store_true",
                   help="Build all layers instead of just the last one")
    p.add_argument("--force",    action="store_true",
                   help="Rebuild even if outputs already exist")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print what would be run without executing")
    return p.parse_args()


def main():
    args = parse_args()

    all_runs = discover_runs()

    # Apply filters
    if args.encoder:
        all_runs = [r for r in all_runs if r["encoder"] == args.encoder]
    if args.variant:
        all_runs = [r for r in all_runs if r["variant"] == args.variant]

    if not all_runs:
        print("No matching runs found.")
        sys.exit(0)

    print(f"\nDiscovered {len(all_runs)} run(s):")
    for r in all_runs:
        print(f"  {r['encoder']:10s} | {r['variant']:12s} | "
              f"folder={r['folder_name']:25s} | layers={r['layers']}")

    if args.dry_run:
        print("\n[dry-run] No commands executed.")
        return

    for run_info in all_runs:
        folder_name = run_info["folder_name"]
        all_layers  = run_info["layers"]

        # Determine which layers to build
        if args.layer is not None:
            if args.layer not in all_layers:
                print(f"\n  [skip] Layer {args.layer} not in {folder_name}")
                continue
            target_layers = [args.layer]
        elif args.all_layers:
            target_layers = all_layers
        else:
            target_layers = [all_layers[-1]]   # last layer only by default

        print(f"\n{'='*60}")
        print(f"  Run: {folder_name}  (layers: {target_layers})")
        print(f"{'='*60}")

        # ── Step 1: codebook (once per encoder×variant) ──────────────
        if _xae_path(folder_name):
            build_codebook(run_info, force=args.force)
        else:
            print(f"  [skip codebook] No XAE found for {folder_name}")

        # ── Steps 2–4: per layer ──────────────────────────────────────
        for layer in target_layers:
            print(f"\n  --- Layer {layer} ---")

            if _xae_path(folder_name):
                run_explain_features(run_info, layer, force=args.force)

            write_metadata(run_info, layer)

            if _codebook_path(folder_name).exists() and _xae_path(folder_name):
                build_app_cache(run_info, layer, force=args.force)
            else:
                print(f"  [skip app_cache] Codebook or XAE missing for {folder_name}")

    print(f"\n{'='*60}")
    print("  All done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Post-encoder M2 metric for concept steering.

The default M2 metric in ``plot_concept_steering_xae_all.py`` trains a logistic
probe on **SAE z-features at the steered layer**. That answers the question
"are the SAE features at this layer separable for target vs source?", which
becomes a confound for cross-encoder comparison: when the SAE sits at the
classifier-adjacent last layer (e.g. SleepFM L2), z-feature separability is
trivially high; when it sits mid-encoder (e.g. REVE L8), z-features may be
concept-aligned but not classifier-aligned.

This tool computes a fairer cross-encoder M2 by training the probe on the
**post-encoder (final-layer)** embeddings, and evaluating on the final-layer
embeddings of *propagated* steered tokens. Specifically:

    1. Train probe on (final-layer src, final-layer tgt) embeddings.
    2. For each steering step, compute steered z at the target layer.
    3. Decode steered z back to a target-layer embedding.
    4. Forward-propagate the steered target-layer embedding through layers
       L+1 ... L_final via a forward-hook that overrides the layer-L block
       output.
    5. Capture the resulting final-layer embeddings.
    6. Evaluate the probe on those.

Saves: appends ``m2_post_real`` per (concept, step) to the per-experiment
``steering_metrics_<experiment>.json`` (re-reads, augments, re-writes).

Usage::

    uv run tools/compute_m2_post_encoder.py --experiment reve_qjbe08_layer8
    uv run tools/compute_m2_post_encoder.py --experiment reve_qjbe08_layer8 \\
        --concepts age_group classification --steps 0 20 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from sae4eeg.dataset import H5PYDatasetLabeled, StandardizeLabel
from sae4eeg.encoders import EncoderBackend
from sae4eeg.sae import SparseAutoencoder

import build_app_cache as _bac
from build_app_cache import _load_model

# Mirror the concept configs from plot_concept_steering_xae_all.py — same keys.
# We only need (key, cls_filter, src_field, src_vals, tgt_vals, [age_filter])
# for the M2 probe; we ignore plotting / spectral parts here.
_AGE_CHILD = {"0-3", "4-9", "10-19"}
_AGE_ADULT = {"20-29", "30-39", "40-49", "50-59", "60+"}
_ABN_CATS  = ["Abnormal - Epileptiform", "Abnormal - Other"]
_NORMAL    = ["Normal"]

# Each concept: src_indices, tgt_indices, sorted_feats determine steering
# replacements; we use the same logic as plot_concept_steering_xae_all.py
# but reproduce it here for self-containedness.

DEFAULT_STEPS = [0, 20, 50]


# ── helpers ──────────────────────────────────────────────────────────────────

def _split_indices(token_meta: dict, src_field: str, src_vals: list[str],
                   tgt_vals: list[str], cls_filter: str | None,
                   cls_arr: np.ndarray, age_arr: np.ndarray | None = None,
                   age_filter_src: set | None = None,
                   age_filter_tgt: set | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    arr = token_meta[src_field]
    src_mask = np.isin(arr, src_vals)
    tgt_mask = np.isin(arr, tgt_vals)
    if cls_filter == "abnormal":
        cm = np.isin(cls_arr, _ABN_CATS)
        src_mask &= cm; tgt_mask &= cm
    elif cls_filter == "normal":
        cm = cls_arr == "Normal"
        src_mask &= cm; tgt_mask &= cm
    # cls_filter in (None, "all") → no class mask (unconditional)
    if age_filter_src is not None and age_arr is not None:
        src_mask &= np.isin(age_arr, list(age_filter_src))
    if age_filter_tgt is not None and age_arr is not None:
        tgt_mask &= np.isin(age_arr, list(age_filter_tgt))
    return np.where(src_mask)[0], np.where(tgt_mask)[0]


def _rank_concept_feats(z_all: np.ndarray, idx_src: np.ndarray,
                        idx_tgt: np.ndarray) -> np.ndarray:
    """Rank features by |coef| of a logistic probe trained on z-features
    src vs tgt (mirrors the fallback branch of plot_concept_steering_xae_all)."""
    X = np.vstack([z_all[idx_src], z_all[idx_tgt]])
    y = np.array([0] * len(idx_src) + [1] * len(idx_tgt))
    p = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs").fit(X, y)
    return np.argsort(np.abs(p.coef_[0]))[::-1]


def _substitute(z: torch.Tensor, z_donor: torch.Tensor,
                feats: np.ndarray) -> torch.Tensor:
    out = z.clone()
    if len(feats) == 0:
        return out
    feats_t = torch.tensor(np.ascontiguousarray(feats), dtype=torch.long)
    out[:, feats_t] = z_donor[:, feats_t].mean(dim=0, keepdim=True)
    return out


def _decode_z_to_layerL(z_steered: torch.Tensor, sae: SparseAutoencoder,
                        act_mean: torch.Tensor, act_std: torch.Tensor) -> torch.Tensor:
    """SAE.decode → un-normalise back to raw layer-L embedding space."""
    with torch.no_grad():
        x_norm = sae.decode(z_steered.to(DEVICE))
    return (x_norm * act_std + act_mean).cpu()


# ── encoder propagation with layer-L injection ──────────────────────────────

def _propagate_with_injection(model, layer_idx: int, raw_eeg: torch.Tensor,
                               injected_layerL: torch.Tensor,
                               token_pos_idx: list[int]) -> torch.Tensor:
    """Run encoder on `raw_eeg` (single window, B=1), but at layer `layer_idx`
    replace the listed token positions' output with `injected_layerL`. Return
    the final-layer embedding (B, N_tok, E_final).

    `injected_layerL` has shape (len(token_pos_idx), E_layer_L). We assume
    layer-L's hooked output is shape (1, N_tok_flat, E) after flattening any
    REVE-style 4D output. We override token_pos_idx along the N_tok_flat axis.
    """
    hookable = model.get_hookable_layers()
    target   = hookable[layer_idx]
    final    = hookable[-1]

    captured_final: dict[str, torch.Tensor] = {}

    def inject_hook(module, inp, out):
        # Output may be (B, C, S, E) or (B, S, E). Flatten C×S → tokens.
        if out.dim() == 4:
            B2, C, S, E = out.shape
            flat = out.reshape(B2, C * S, E)
            for k, pos in enumerate(token_pos_idx):
                flat[0, pos] = injected_layerL[k].to(out.device, dtype=flat.dtype)
            return flat.reshape(B2, C, S, E)
        else:
            for k, pos in enumerate(token_pos_idx):
                out[0, pos] = injected_layerL[k].to(out.device, dtype=out.dtype)
            return out

    def capture_final_hook(module, inp, out):
        if out.dim() == 4:
            B2, C, S, E = out.shape
            captured_final["v"] = out.reshape(B2, C * S, E).detach().cpu()
        else:
            captured_final["v"] = out.detach().cpu()

    h_inj  = target.register_forward_hook(inject_hook)
    h_fin  = final.register_forward_hook(capture_final_hook)
    try:
        with torch.no_grad():
            _ = model.encode(raw_eeg.to(DEVICE))
    finally:
        h_inj.remove()
        h_fin.remove()
    return captured_final["v"]   # (1, N_tok_flat, E_final)


# ── concept configs (parallel to plot_concept_steering_xae_all.py) ──────────

CONCEPTS = [
    dict(key="age", src_field="age_group", src_vals=list(_AGE_CHILD),
         tgt_vals=list(_AGE_ADULT), cls_filter="normal"),
    dict(key="age", src_field="age_group", src_vals=list(_AGE_CHILD),
         tgt_vals=list(_AGE_ADULT), cls_filter="abnormal"),
    dict(key="classification_adult", src_field="classification",
         src_vals=_ABN_CATS, tgt_vals=_NORMAL, cls_filter="all",
         age_filter_src=_AGE_ADULT, age_filter_tgt=_AGE_ADULT),
    dict(key="classification_child", src_field="classification",
         src_vals=_ABN_CATS, tgt_vals=_NORMAL, cls_filter="all",
         age_filter_src=_AGE_CHILD, age_filter_tgt=_AGE_CHILD),
    dict(key="medication_asm", src_field="medication_group",
         src_vals=["ASM"], tgt_vals=["No Current Medication"],
         cls_filter="normal"),
    dict(key="medication_asm", src_field="medication_group",
         src_vals=["ASM"], tgt_vals=["No Current Medication"],
         cls_filter="abnormal"),
]

# Unconditional erasure variants (used by the v8/v9/v10 unconditional pipeline).
# All cls_filter="all"; population-mean donor (donor-mode="population") gives
# concept ERASURE rather than substitution.
CONCEPTS_UNCOND = [
    dict(key="age", src_field="age_group", src_vals=["0-3"],
         tgt_vals=["50-59", "60+"], cls_filter="all"),
    dict(key="gender", src_field="gender", src_vals=["female"],
         tgt_vals=["male"], cls_filter="all"),
    dict(key="medication_asm", src_field="medication_group",
         src_vals=["ASM"], tgt_vals=["No Current Medication"],
         cls_filter="all"),
    dict(key="classification", src_field="classification",
         src_vals=_ABN_CATS, tgt_vals=_NORMAL, cls_filter="all"),
    dict(key="classification_child", src_field="classification",
         src_vals=_ABN_CATS, tgt_vals=_NORMAL, cls_filter="all",
         age_filter_src=_AGE_CHILD, age_filter_tgt=_AGE_CHILD),
    dict(key="classification_adult", src_field="classification",
         src_vals=_ABN_CATS, tgt_vals=_NORMAL, cls_filter="all",
         age_filter_src=_AGE_ADULT, age_filter_tgt=_AGE_ADULT),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--n-subsample", type=int, default=200,
                        help="Per-group token subsample for M2_post (<<500 to keep "
                             "encoder forwards tractable; default 200).")
    parser.add_argument("--metrics-path", type=Path, default=None,
                        help="Custom path to steering_metrics.json. Default = "
                             "paper/concept_steering_figures/steering_metrics_<exp>.json")
    parser.add_argument("--unconditional", action="store_true",
                        help="Use CONCEPTS_UNCOND (cls_filter=all, no class mask) "
                             "with population-mean donor for concept erasure.")
    parser.add_argument("--donor-mode", choices=["target", "population"], default=None,
                        help="Override donor mode. Default: 'population' if "
                             "--unconditional, else 'target'.")
    args = parser.parse_args()

    concepts = CONCEPTS_UNCOND if args.unconditional else CONCEPTS
    donor_mode = args.donor_mode or ("population" if args.unconditional else "target")

    sc_path = ROOT / "results" / "steering_cache" / args.experiment / "steering_cache.pt"
    metrics_path = (args.metrics_path
                    or ROOT / "paper" / "concept_steering_figures"
                       / f"steering_metrics_{args.experiment}.json")

    print(f"[load] {sc_path.name}")
    sc = torch.load(sc_path, map_location="cpu", weights_only=False)
    target_layer_pre = int(sc["target_layer"])
    final_layer_pre  = int(sc["final_layer"])
    is_final = target_layer_pre == final_layer_pre
    if "window_file_idx" not in sc and not is_final:
        raise SystemExit(
            "Steering cache lacks window indices — rebuild with the patched "
            "tools/build_steering_cache.py first.")
    if "window_file_idx" not in sc:
        # Final-layer case: no propagation needed, no window reload either.
        sc["window_file_idx"] = np.arange(0)
        sc["window_local_idx"] = np.arange(0)
        print("  [info] no window indices; final-layer fast path will be used")
    # When target_layer == final_layer (e.g. SleepFM L2 = last layer), the
    # post-encoder M2 is identical to the local M2 — no propagation needed,
    # the layer-L embedding *is* the final-layer embedding.
    if sc.get("embeddings_final") is None:
        sc["embeddings_final"] = sc["embeddings"]
        print("  [info] target_layer == final_layer; M2_post will equal M2_local")

    target_layer = int(sc["target_layer"])
    final_layer  = int(sc["final_layer"])
    n_tokens     = sc["n_tokens"]
    tpw          = sc["tokens_per_window"]
    print(f"  target_layer={target_layer}  final_layer={final_layer}  "
          f"n_tokens={n_tokens}  tokens_per_window={tpw}")

    print(f"[load] experiment metadata + SAE")
    meta_path = ROOT / "results" / "experiments" / args.experiment / "metadata.json"
    meta = json.loads(meta_path.read_text())
    sae_ckpt = torch.load(ROOT / meta["sae_checkpoint"],
                          map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"],
                            mode="topk", k=meta["k"]).to(DEVICE).eval()
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    act_mean = sae_ckpt["act_mean"].to(DEVICE)
    act_std  = sae_ckpt["act_std"].to(DEVICE)

    print(f"[load] encoder")
    weights = meta.get("weights_path")
    weights = ROOT / weights if weights else None
    model   = _load_model(meta["encoder"], weights_path=weights)

    # Compute SAE z for all tokens (used for ranking + steering substitution)
    print(f"[compute] SAE z for {n_tokens} tokens (batched)")
    embeddings = sc["embeddings"]
    BATCH = 4096
    z_chunks = []
    with torch.no_grad():
        for s in range(0, n_tokens, BATCH):
            x = embeddings[s:s + BATCH].to(DEVICE)
            xn = (x - act_mean) / act_std.clamp(min=1e-8)
            z_chunks.append(sae.encode(xn).cpu())
    z_all = torch.cat(z_chunks, dim=0)
    nf = z_all.shape[1]
    print(f"  z_all shape: {tuple(z_all.shape)}")

    # Token → window mapping (default DataLoader order, no permutation)
    file_idx_arr  = sc["window_file_idx"]
    local_idx_arr = sc["window_local_idx"]
    n_windows_used = len(file_idx_arr)
    print(f"  {n_windows_used} windows used for {n_tokens} tokens")

    # Open dataset for window reload
    data_path = sc.get("data_path", "data/D4-v3-preprocessed-v1")
    full_ds = H5PYDatasetLabeled(str(ROOT / data_path), transform=StandardizeLabel())

    # Token meta needed for src/tgt selection
    token_meta = {f: sc[f] for f in
                  ["age_group", "gender", "classification", "medication_group"]}
    cls_arr = sc["classification"]
    age_arr = sc["age_group"]

    # Final-layer probe inputs already collected
    embeddings_final = sc["embeddings_final"]
    final_dim = embeddings_final.shape[1]

    rng = np.random.default_rng(42)

    # Load existing per-experiment metrics file (if any) to augment
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        ent_by_key = {(r["concept"], r["cls_filter"]): r for r in existing}
    else:
        existing = []
        ent_by_key = {}

    out_rows: list[dict] = []
    for cfg in concepts:
        key, cls_filter = cfg["key"], cfg["cls_filter"]
        idx_src, idx_tgt = _split_indices(
            token_meta, cfg["src_field"], cfg["src_vals"], cfg["tgt_vals"],
            cls_filter, cls_arr, age_arr,
            cfg.get("age_filter_src"), cfg.get("age_filter_tgt"),
        )
        if len(idx_src) < 50 or len(idx_tgt) < 50:
            print(f"\n[skip] {key}/{cls_filter}: too few tokens "
                  f"(src={len(idx_src)} tgt={len(idx_tgt)})")
            continue

        # Subsample for tractability
        n_sub = min(args.n_subsample, len(idx_src))
        idx_src_sub = rng.choice(idx_src, size=n_sub, replace=False)

        print(f"\n=== {key} / {cls_filter}  src={len(idx_src)} tgt={len(idx_tgt)}  "
              f"subsample n={n_sub} ===")

        # Rank features (same logic as the plotting tool's fallback)
        sorted_feats = _rank_concept_feats(z_all.numpy(), idx_src, idx_tgt)

        # Train probe on **final-layer** embeddings of all src/tgt tokens
        Xf = np.vstack([embeddings_final[idx_src].numpy(),
                        embeddings_final[idx_tgt].numpy()])
        yf = np.array([0] * len(idx_src) + [1] * len(idx_tgt))
        probe = LogisticRegression(max_iter=300, C=0.1, solver="lbfgs").fit(Xf, yf)
        probe_acc = probe.score(Xf, yf)
        print(f"  final-layer probe acc: {probe_acc:.3f}")

        # For each step, compute steered z, decode to layer-L emb, propagate
        # through encoder, capture final-layer outputs, evaluate probe.
        m2_post: dict[int, float] = {}

        # Group sub-sampled tokens by window for batched forward
        window_idx_per_token = idx_src_sub // tpw
        token_pos_per_token  = idx_src_sub %  tpw
        unique_windows       = np.unique(window_idx_per_token)
        print(f"  spans {len(unique_windows)} windows (avg "
              f"{n_sub / len(unique_windows):.1f} tokens/window)")

        # Donor for substitution. For 'population' (unconditional erasure), use
        # the global token mean — that drives M2 toward 0.5 if successful. For
        # 'target', use the target-group mean — that drives M2 toward 1.
        if donor_mode == "population":
            z_donor = z_all  # full pool — _substitute uses .mean(dim=0) below
        else:
            z_donor = z_all[idx_tgt]
        for n in args.steps:
            if n == 0:
                # Step 0: no steering; just evaluate probe on final-layer src.
                p1 = probe.predict_proba(embeddings_final[idx_src_sub].numpy())[:, 1]
                m2_post[n] = float(p1.mean())
                print(f"  step={n:>3d}  M2_post={m2_post[n]:.3f}")
                continue

            # Compute steered z for this step
            feats = sorted_feats[:n]
            z_src_sub = z_all[idx_src_sub]
            z_steered = _substitute(z_src_sub, z_donor, feats)
            # Decode steered z back to layer-L embedding space (un-normalised)
            layerL_steered = _decode_z_to_layerL(z_steered, sae, act_mean, act_std)

            final_steered_emb = np.zeros((n_sub, final_dim), dtype=np.float32)
            # Fast path: if target_layer == final_layer, the SAE-decoded layer-L
            # embedding *is* the final-layer embedding — no propagation needed.
            if target_layer == final_layer:
                final_steered_emb[:] = layerL_steered.numpy()
                p_target = probe.predict_proba(final_steered_emb)[:, 1]
                m2_post[n] = float(p_target.mean())
                print(f"  step={n:>3d}  M2_post={m2_post[n]:.3f} (no propagation)")
                continue

            pbar = tqdm(unique_windows, desc=f"  step={n} propagate",
                        leave=False)
            for w_id in pbar:
                # Tokens in this window that we need to inject
                token_mask = window_idx_per_token == w_id
                pos_list   = token_pos_per_token[token_mask].tolist()
                inj_layerL = layerL_steered[token_mask]   # (k_window, E_L)
                # Reload the EEG window
                fi = int(file_idx_arr[w_id])
                li = int(local_idx_arr[w_id])
                # use dataset's __getitem__ via a single-window subset
                # (ensures the same standardisation that build_steering_cache used)
                # H5PYDatasetLabeled stores windows by [file_idx][local_idx]
                eeg = full_ds[int(w_id)][0]
                if not torch.is_tensor(eeg):
                    eeg = torch.as_tensor(eeg)
                # Forward with injection at layer L, capture final-layer output
                final_out = _propagate_with_injection(
                    model, target_layer, eeg.unsqueeze(0), inj_layerL,
                    pos_list,
                )   # (1, N_tok, E_final)
                # Pull out the same token positions from the final-layer output
                for k, pos in enumerate(pos_list):
                    src_idx_within = int(np.where(token_mask)[0][k])
                    final_steered_emb[src_idx_within] = (
                        final_out[0, pos].numpy()
                    )

            p_target = probe.predict_proba(final_steered_emb)[:, 1]
            m2_post[n] = float(p_target.mean())
            print(f"  step={n:>3d}  M2_post={m2_post[n]:.3f}")

        # Append to metrics
        row_key = (cfg["key"], cfg["cls_filter"])
        if row_key in ent_by_key:
            ent_by_key[row_key]["m2_post_real"] = {str(k): float(v)
                                                    for k, v in m2_post.items()}
            ent_by_key[row_key]["m2_post_meta"] = {
                "n_subsample": n_sub,
                "final_layer": final_layer,
                "probe_acc":   float(probe_acc),
            }
        else:
            new_row = {
                "experiment":   args.experiment,
                "concept":      cfg["key"],
                "cls_filter":   cfg["cls_filter"],
                "n_src_tokens": int(len(idx_src)),
                "n_tgt_tokens": int(len(idx_tgt)),
                "m2_post_real": {str(k): float(v) for k, v in m2_post.items()},
                "m2_post_meta": {
                    "n_subsample": n_sub,
                    "final_layer": final_layer,
                    "probe_acc":   float(probe_acc),
                },
            }
            existing.append(new_row)
            ent_by_key[row_key] = new_row
        out_rows.append({"concept": cfg["key"], "cls_filter": cfg["cls_filter"],
                         "m2_post": m2_post, "probe_acc": probe_acc})

    # Write metrics file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(existing, indent=2))
    try:
        rel = metrics_path.relative_to(ROOT)
    except ValueError:
        rel = metrics_path
    print(f"\nSaved {rel}  (rows={len(existing)})")
    for r in out_rows:
        print(f"  {r['concept']}/{r['cls_filter']}: M2_post = "
              f"{', '.join(f'{k}={v:.3f}' for k, v in r['m2_post'].items())}  "
              f"(probe_acc={r['probe_acc']:.2f})")


if __name__ == "__main__":
    main()

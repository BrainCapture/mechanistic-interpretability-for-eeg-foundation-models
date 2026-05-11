"""Probe SAE reconstruction quality via linear classification.

Trains a linear classifier head on frozen SleepFM embeddings under four
conditions:
  baseline    — original encoder, no SAE intervention
  sae_layer0  — layer-0 output replaced by SAE reconstruction, rest runs on
  sae_layer1  — layer-1 output replaced by SAE reconstruction, rest runs on
  sae_layer2  — layer-2 output replaced by SAE reconstruction

The encoder is completely frozen in all cases; only the linear head is
trained. This tests whether the SAE bottleneck at each layer destroys
class-discriminative information.

Pre-computes mean-pooled embeddings once per condition (fast, encoder frozen),
then trains a logistic regression head on those embeddings.

Results are printed and saved to results/probe_reconstruction/results.json.

Usage::

    uv run tools/probe_sae_reconstruction.py
    uv run tools/probe_sae_reconstruction.py --epochs 30 --lr 1e-3
    uv run tools/probe_sae_reconstruction.py --max-samples 5000
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sae4eeg.sae import SparseAutoencoder
from sae4eeg.encoders import load_encoder
from sae4eeg.dataset import StandardizeLabel, get_dataloaders

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "D4-v3-preprocessed-v2"
WEIGHTS_PATH = ROOT / "checkpoints" / "finetuned" / "sleepfm1.ckpt"
SAE_DIR = ROOT / "results" / "features" / "sleepfm_finetuned"
OUT_DIR = ROOT / "results" / "probe_reconstruction"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAE hyperparameters (must match training)
EMBED_DIM = 128
EXPANSION = 1.0
K = 8
N_LAYERS = 3


# ─────────────────────────────────────────────────────────────────────────────
# SAE injection hook
# ─────────────────────────────────────────────────────────────────────────────

class _SAEInjector:
    """Forward hook that replaces a transformer layer output with its SAE
    reconstruction.

    The SAE was trained on normalised activations, so we:
      1. Normalise the raw layer output using the stored act_mean / act_std.
      2. Run it through the SAE (encode → decode).
      3. Unnormalise back to the original activation space.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        act_mean: torch.Tensor,
        act_std: torch.Tensor,
        device: str,
    ):
        self.sae = sae.to(device).eval()
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device
        self._handle = None

    def _hook_fn(self, module, input, output):
        out = output.to(self.device)
        z_norm = (out - self.act_mean) / self.act_std
        with torch.no_grad():
            z_hat = self.sae.decode(self.sae.encode(z_norm))
        return z_hat * self.act_std + self.act_mean

    def register(self, layer: nn.Module):
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────────────────
# SAE loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sae(layer: int) -> tuple[SparseAutoencoder, torch.Tensor, torch.Tensor]:
    """Load SAE checkpoint for a given layer.

    Returns (sae, act_mean, act_std).
    """
    path = SAE_DIR / f"sae_sleepfm_exp{EXPANSION}_k{K}_layer{layer}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(
        input_dim=int(ckpt["embed_dim"]),
        expansion=int(ckpt["expansion"]),
        k=int(ckpt["k"]),
        mode="topk",
    )
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()
    return sae, ckpt["act_mean"], ckpt["act_std"]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder,
    loader: DataLoader,
    injections: list[tuple[_SAEInjector, nn.Module]],
    max_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder over a dataloader and return (embeddings, labels).

    embeddings  : (N, embed_dim)  — mean-pooled over token dimension
    labels      : (N,)            — integer class labels
    injections  : list of (injector, layer) pairs — all registered before the
                  forward pass and removed after.
    """
    for injector, layer_module in injections:
        injector.register(layer_module)

    emb_list, label_list = [], []
    n_seen = 0

    for batch in loader:
        x, y, _ = batch          # (B, C, T), (B,), channels
        x = x.to(DEVICE)

        tokens = encoder.encode(x)          # (B, S, E)
        pooled = tokens.mean(dim=1).cpu()   # (B, E)

        emb_list.append(pooled)
        label_list.append(y.long().cpu())
        n_seen += x.shape[0]

        if max_samples is not None and n_seen >= max_samples:
            break

    for injector, _ in injections:
        injector.remove()

    return torch.cat(emb_list), torch.cat(label_list)


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

def train_probe(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    n_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
) -> nn.Linear:
    """Train a linear classification head on pre-computed embeddings."""
    torch.manual_seed(seed)
    head = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(train_emb, train_labels)
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        head.train()
        for xb, yb in ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(head, val_emb, val_labels)["ba"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  val_ba={val_acc:.4f}", flush=True)

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


@torch.no_grad()
def evaluate(head: nn.Linear, emb: torch.Tensor, labels: torch.Tensor) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
    head.eval()
    logits = head(emb.to(DEVICE))
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.numpy()
    return {
        "acc":  float((preds == labels_np).mean()),
        "ba":   float(balanced_accuracy_score(labels_np, preds)),
        "auc":  float(roc_auc_score(labels_np, probs)),
        "f1":   float(f1_score(labels_np, preds, average="weighted")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--seeds",         type=int,   default=10,
                        help="Number of random seeds for the probe (default: 10)")
    parser.add_argument("--weights-path",  type=str,   default=None,
                        help="Override encoder weights path (default: finetuned checkpoint)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the baseline condition (no SAE injection)")
    parser.add_argument("--max-samples",   type=int,   default=None,
                        help="Cap per split (for fast testing)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    weights_path = Path(args.weights_path) if args.weights_path else WEIGHTS_PATH
    print(f"Encoder weights: {weights_path}")
    encoder = load_encoder("sleepfm", weights_path=weights_path)
    encoder.to(DEVICE).eval()

    transformer_layers = encoder.model.transformer_encoder.layers
    print(f"Transformer layers: {len(transformer_layers)}")

    # ── Load SAEs ─────────────────────────────────────────────────────────────
    saes = {}
    if not args.baseline_only:
        for layer in range(N_LAYERS):
            sae, act_mean, act_std = load_sae(layer)
            saes[layer] = _SAEInjector(sae, act_mean, act_std, DEVICE)
            print(f"Loaded SAE for layer {layer}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data loaders ─────────────────────────────────────────────────────────
    transform = StandardizeLabel(mean=0.0, std=1e-5, clip=5e-5)
    folds = list(get_dataloaders(
        str(DATA_PATH),
        transformer=transform,
        batch_size=32,
        num_workers=4,
        n_splits=1,
        split_info_path=str(OUT_DIR / "splits.json"),
    ))
    assert len(folds) == 1, "Expected single fold"
    _, train_loader, val_loader, test_loader = folds[0]

    # ── Conditions ────────────────────────────────────────────────────────────
    # Each condition is (name, [(injector, layer), ...])
    def _inj(layer_idx):
        return (saes[layer_idx], transformer_layers[layer_idx])

    conditions: list[tuple[str, list]] = [("baseline", [])]
    if not args.baseline_only:
        for layer_idx in range(N_LAYERS):
            conditions.append((f"sae_layer{layer_idx}", [_inj(layer_idx)]))
        # Multi-layer combinations
        conditions.append(("sae_layer0+1",   [_inj(0), _inj(1)]))
        conditions.append(("sae_layer1+2",   [_inj(1), _inj(2)]))
        conditions.append(("sae_layer0+1+2", [_inj(0), _inj(1), _inj(2)]))

    results = {}

    for cond_name, injections in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        # Pre-compute embeddings (encoder frozen, one pass per split)
        print("  Extracting train embeddings …")
        train_emb, train_labels = extract_embeddings(
            encoder, train_loader, injections, args.max_samples
        )
        print("  Extracting val embeddings …")
        val_emb, val_labels = extract_embeddings(
            encoder, val_loader, injections, args.max_samples
        )
        print("  Extracting test embeddings …")
        test_emb, test_labels = extract_embeddings(
            encoder, test_loader, injections, args.max_samples
        )

        n_classes = int(train_labels.max().item()) + 1
        print(f"  n_classes={n_classes}  "
              f"train={len(train_emb)}  val={len(val_emb)}  test={len(test_emb)}")

        # Train linear head with multiple seeds and report mean ± std
        print(f"  Training probe ({args.seeds} seeds) …")
        import statistics
        seed_metrics: list[dict] = []
        for seed in range(args.seeds):
            head = train_probe(
                train_emb, train_labels,
                val_emb, val_labels,
                n_classes=n_classes,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
            )
            seed_metrics.append(evaluate(head, test_emb, test_labels))

        def _agg(key):
            vals = [m[key] for m in seed_metrics]
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        ba_mean,  ba_std  = _agg("ba")
        auc_mean, auc_std = _agg("auc")
        f1_mean,  f1_std  = _agg("f1")
        acc_mean, acc_std = _agg("acc")
        print(f"  acc={acc_mean:.4f}±{acc_std:.4f}  ba={ba_mean:.4f}±{ba_std:.4f}  "
              f"auc={auc_mean:.4f}±{auc_std:.4f}  f1={f1_mean:.4f}±{f1_std:.4f}")
        results[cond_name] = {
            "acc_mean":  round(acc_mean, 4), "acc_std":  round(acc_std,  4),
            "ba_mean":   round(ba_mean,  4), "ba_std":   round(ba_std,   4),
            "auc_mean":  round(auc_mean, 4), "auc_std":  round(auc_std,  4),
            "f1_mean":   round(f1_mean,  4), "f1_std":   round(f1_std,   4),
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Condition':<20}  {'acc':>7}  {'BA':>7}  {'AUC':>7}  {'F1':>7}")
    for cond, r in results.items():
        print(f"{cond:<20}  "
              f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
              f"{r['ba_mean']:.4f}±{r['ba_std']:.4f}  "
              f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
              f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}")

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

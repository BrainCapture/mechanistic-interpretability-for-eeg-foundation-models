import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sae4eeg.encoders import EncoderBackend


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Activation Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ActivationExtractor:
    """
    Registers forward hooks on a list of ``nn.Module`` layers and collects
    the *output* of each layer.

    Each stored activation has shape  (B, S, E)  where
        B = batch size
        S = temporal sequence length (tokens per window)
        E = embed_dim

    Can be constructed in two ways:

    1.  From a plain ``nn.Module`` that has a ``transformer_encoder.layers``
        attribute (legacy path, preserves backward compatibility with
        ``SetTransformer``)::

            extractor = ActivationExtractor(set_transformer_model)

    2.  From an explicit ``layers`` list (generic path, works with any
        encoder including REVE)::

            extractor = ActivationExtractor(model, layers=backend.get_hookable_layers())
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[nn.Module]] = None,
    ):
        self.model = model
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # If an explicit layer list is provided, use it; otherwise fall back
        # to the legacy SetTransformer attribute path.
        if layers is not None:
            self._layer_list = layers
        else:
            self._layer_list = list(model.transformer_encoder.layers)

        self._register_hooks()

    # ------------------------------------------------------------------
    def _register_hooks(self):
        layers = self._layer_list
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)
        print(f"[ActivationExtractor] Registered hooks on {len(layers)} layers.")

    def _make_hook(self, idx: int):
        def hook_fn(module, input, output):
            # TransformerEncoderLayer returns a tensor (batch_first=True)
            # shape: (B, S, E)
            self._activations[idx] = output.detach().cpu()
        return hook_fn

    # ------------------------------------------------------------------
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Returns cached activations from the most recent forward pass."""
        return dict(self._activations)

    def clear(self):
        self._activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("[ActivationExtractor] All hooks removed.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sparse Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    SAE with two sparsity modes:

      mode='l1'  :  x → encode → ReLU → decode → x̂
                    Loss = MSE(x, x̂) + λ·L1(z)

      mode='topk':  x → encode → ReLU → TopK mask → decode → x̂
                    Loss = MSE(x, x̂)   (sparsity is structural, no L1 needed)

    TopK directly controls L0 by zeroing all but the k largest activations
    per token.  This avoids the failure mode where L1 shrinks magnitudes
    without actually turning neurons off.

    Parameters
    ----------
    input_dim   : dimensionality of the transformer activation (embed_dim)
    expansion   : dictionary size = input_dim * expansion
    l1_coef     : sparsity penalty weight  (only used when mode='l1')
    mode        : 'l1' or 'topk'
    k           : number of active features per token (only used when mode='topk')
    """

    def __init__(
        self,
        input_dim: int,
        expansion: int = 8,
        l1_coef: float = 1e-3,
        mode: str = "topk",
        k: int = 32,
    ):
        super().__init__()
        dict_size = int(input_dim * expansion)
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.mode = mode
        self.k = k

        # Pre-encoder bias: subtract mean activation direction before encoding
        # so the encoder doesn't waste capacity on the shared mean.
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        self.l1_coef = l1_coef

        # Unit-norm decoder columns (important for SAE stability)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self):
        """Keep decoder columns unit-norm to avoid degenerate solutions."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def init_b_pre(self, mean: torch.Tensor):
        """Initialise b_pre to the mean activation direction."""
        self.b_pre.copy_(mean.to(self.b_pre.device))

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x - self.b_pre))
        if self.mode == "topk":
            z = self._topk_mask(z)
        return z

    @staticmethod
    def _topk_mask_fn(z: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out everything except the top-k activations per token."""
        topk_vals, topk_idx = z.topk(k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)
        return z * mask

    def _topk_mask(self, z: torch.Tensor) -> torch.Tensor:
        return self._topk_mask_fn(z, self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor):
        x_hat, z = self(x)
        recon = F.mse_loss(x_hat, x)
        if self.mode == "topk":
            # TopK: sparsity is structural, no L1 penalty needed
            sparse = torch.tensor(0.0, device=x.device)
            total = recon
        else:
            sparse = self.l1_coef * z.abs().mean()
            total = recon + sparse
        return total, recon, sparse, z


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAE Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SAETrainer:
    """
    Collects activations from a SetTransformer over a dataset, then trains
    one SAE per requested layer (or just one target layer for MVP).

    Parameters
    ----------
    embed_dim       : transformer embed_dim
    expansion       : SAE expansion factor
    l1_coef         : sparsity penalty
    target_layers   : list of layer indices to train SAEs on (None = all)
    lr              : Adam learning rate
    batch_size      : SAE training batch size
    epochs          : number of SAE training epochs
    device          : 'cuda' / 'cpu'
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int = 8,
        l1_coef: float = 1e-3,
        mode: str = "topk",
        k: int = 32,
        target_layers: Optional[List[int]] = None,
        lr: float = 3e-4,
        batch_size: int = 2048,
        epochs: int = 20,
        resample_every: int = 5,
        resample_min_alive: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.embed_dim     = embed_dim
        self.expansion     = expansion
        self.l1_coef       = l1_coef
        self.mode          = mode
        self.k             = k
        self.target_layers = target_layers
        self.lr            = lr
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.resample_every     = resample_every
        self.resample_min_alive = resample_min_alive
        self.device        = device
        self.saes: Dict[int, SparseAutoencoder] = {}
        # Per-layer normalisation stats (set during train())
        self.act_means: Dict[int, torch.Tensor] = {}
        self.act_stds:  Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_activations(
        self,
        encoder,  # nn.Module (legacy) OR EncoderBackend
        dataloader: DataLoader,
        max_tokens: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Run full dataset through an encoder and store layer activations.

        Parameters
        ----------
        encoder : nn.Module or EncoderBackend
            Either a plain ``SetTransformer`` (legacy path) or any
            ``EncoderBackend`` instance.  When an ``EncoderBackend`` is
            passed its ``get_hookable_layers()`` method provides the hook
            targets, and its ``model`` attribute is used for the forward pass.
        dataloader : DataLoader

        Returns
        -------
        dict mapping layer_idx → Tensor of shape (N_tokens, E).
        N_tokens is capped at ``max_tokens`` if provided.
        """
        from sae4eeg.encoders import EncoderBackend as _EB  # local import

        n_prefix_tokens = getattr(encoder, "n_prefix_tokens", 0)
        if isinstance(encoder, _EB):
            model      = encoder.model
            all_layers = encoder.get_hookable_layers()
            # Only hook the target layers to avoid accumulating unused activations
            if self.target_layers:
                target_sorted = sorted(self.target_layers)
                hook_layers  = [all_layers[i] for i in target_sorted if i < len(all_layers)]
                # Map enumerate-index → real layer index for later lookup
                _idx_map: Dict[int, int] = {
                    ei: li for ei, li in enumerate(target_sorted) if li < len(all_layers)
                }
            else:
                hook_layers = all_layers
                _idx_map = {i: i for i in range(len(all_layers))}
            call_fn = lambda x: encoder.encode(x)   # noqa: E731
        else:
            # Legacy path: plain SetTransformer
            model      = encoder
            hook_layers = None   # ActivationExtractor will use transformer_encoder.layers
            _idx_map    = None
            call_fn = lambda x: model(x)               # noqa: E731

        model.eval()
        if isinstance(encoder, _EB):
            encoder.to(self.device)
        else:
            model.to(self.device)

        extractor = ActivationExtractor(model, layers=hook_layers)
        buffers: Dict[int, List[torch.Tensor]] = {}
        tokens_collected = 0

        n_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            # Support (x,) or (x, label) tuples
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            extractor.clear()
            _ = call_fn(x)

            for enum_idx, act in extractor.get_activations().items():
                # Remap enumerate-index → real layer index when _idx_map is set
                layer_idx = _idx_map[enum_idx] if _idx_map is not None else enum_idx
                if self.target_layers and layer_idx not in self.target_layers:
                    continue
                # act: (B, S, E) → flatten to (B*S, E) so each token is one sample
                if n_prefix_tokens:
                    act = act[:, n_prefix_tokens:, :]
                B, S, E = act.shape
                flat = act.reshape(B * S, E)
                if max_tokens is not None:
                    remaining = max_tokens - tokens_collected
                    flat = flat[:remaining]
                buffers.setdefault(layer_idx, []).append(flat)

            tokens_collected += x.shape[0] * (flat.shape[0] // x.shape[0] if buffers else 1)

            suffix = f"  ({tokens_collected}/{max_tokens} tokens)" if max_tokens else ""
            print(f"    batch {batch_idx+1}/{n_batches}{suffix}", flush=True)

            if max_tokens is not None and tokens_collected >= max_tokens:
                print(f"    reached max_tokens={max_tokens}, stopping early", flush=True)
                break

        extractor.remove_hooks()
        return {k: torch.cat(v, dim=0) for k, v in buffers.items()}

    # ------------------------------------------------------------------
    def train(
        self,
        encoder,  # nn.Module (legacy) OR EncoderBackend
        dataloader: DataLoader,
        max_tokens: Optional[int] = None,
    ) -> Dict[int, SparseAutoencoder]:
        """
        Full pipeline: collect activations → normalise → train SAEs → return trained SAEs.

        Parameters
        ----------
        encoder : nn.Module or EncoderBackend
            The encoder to extract activations from.
        dataloader : DataLoader
        max_tokens : int, optional
            Cap on tokens to collect (passed through to collect_activations).
        """
        print("=== Collecting activations ===")
        all_acts = self.collect_activations(encoder, dataloader, max_tokens=max_tokens)
        print(f"Collected activations for layers: {sorted(all_acts.keys())}")

        # Normalise activations to zero-mean / unit-std per dimension
        for layer_idx in list(all_acts.keys()):
            acts = all_acts[layer_idx]
            mean = acts.mean(dim=0)
            std  = acts.std(dim=0).clamp(min=1e-8)
            self.act_means[layer_idx] = mean
            self.act_stds[layer_idx]  = std
            all_acts[layer_idx] = (acts - mean) / std
            print(f"  Layer {layer_idx}: normalised {acts.shape[0]} tokens  "
                  f"(mean-norm={mean.norm():.3f}, std-mean={std.mean():.4f})")

        for layer_idx, acts in sorted(all_acts.items()):
            print(f"\n=== Training SAE on layer {layer_idx} | "
                  f"tokens={acts.shape[0]}, dim={acts.shape[1]} ===")
            sae = self._train_single_sae(acts, layer_idx)
            self.saes[layer_idx] = sae

        return self.saes

    # ------------------------------------------------------------------
    def _train_single_sae(
        self, acts: torch.Tensor, layer_idx: int
    ) -> SparseAutoencoder:
        sae = SparseAutoencoder(
            self.embed_dim, self.expansion, self.l1_coef,
            mode=self.mode, k=self.k,
        ).to(self.device)

        # Initialise b_pre to the mean of the (already normalised) activations
        sae.init_b_pre(acts.mean(dim=0))

        opt = torch.optim.Adam(sae.parameters(), lr=self.lr)
        dataset = TensorDataset(acts)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Track firing counts: per-epoch for dead detection, cumulative for info
        total_fires = torch.zeros(sae.dict_size, device=self.device)

        for epoch in range(self.epochs):
            epoch_total = epoch_recon = epoch_sparse = 0.0
            epoch_fires = torch.zeros(sae.dict_size, device=self.device)

            for (x,) in loader:
                x = x.to(self.device)
                loss, recon, sparse, z = sae.loss(x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae._normalise_decoder()   # keep columns unit-norm
                epoch_total  += loss.item()
                epoch_recon  += recon.item()
                epoch_sparse += sparse.item()

                # Accumulate which neurons fired this epoch
                epoch_fires += (z.detach() > 0).float().sum(dim=0)

            total_fires += epoch_fires
            n = len(loader)
            n_dead_epoch = (epoch_fires == 0).sum().item()
            n_dead_total = (total_fires == 0).sum().item()
            print(f"  Layer {layer_idx} | Epoch {epoch+1:3d}/{self.epochs} | "
                  f"loss={epoch_total/n:.4f}  recon={epoch_recon/n:.4f}  "
                  f"sparse={epoch_sparse/n:.4f}  "
                  f"dead_epoch={n_dead_epoch}/{sae.dict_size}  "
                  f"dead_total={n_dead_total}/{sae.dict_size}", flush=True)

            # ── Dead neuron resampling (per-epoch detection) ──────────
            # Use epoch_fires (not cumulative) so we catch neurons that
            # died recently rather than waiting for a long window.
            if (self.resample_every > 0
                    and (epoch + 1) % self.resample_every == 0
                    and epoch + 1 < self.epochs):
                n_resampled = self._resample_dead_neurons(
                    sae, acts, epoch_fires, opt
                )

        return sae.cpu()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _resample_dead_neurons(
        self,
        sae: SparseAutoencoder,
        acts: torch.Tensor,
        neuron_fires: torch.Tensor,
        opt: torch.optim.Optimizer,
    ) -> int:
        """
        Anthropic-style dead neuron resampling.
        Dead neurons get their encoder weight replaced by a high-loss input
        direction, and decoder column reset to match.
        """
        dead_mask = neuron_fires == 0
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Find the inputs with the highest reconstruction error
        sample_size = min(len(acts), 8192)
        idx = torch.randperm(len(acts))[:sample_size]
        x_sample = acts[idx].to(self.device)
        x_hat, _ = sae(x_sample)
        losses = (x_sample - x_hat).pow(2).sum(dim=-1)  # per-token MSE

        # Sample dead-neuron replacement directions proportional to loss
        probs = losses / losses.sum()
        replace_idx = torch.multinomial(probs, n_dead, replacement=True)
        replacement_dirs = x_sample[replace_idx] - sae.b_pre  # subtract b_pre

        # Normalise replacement directions and add slight noise to encourage diversity
        norms = replacement_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        replacement_dirs = replacement_dirs / norms
        replacement_dirs = replacement_dirs + 1e-3 * torch.randn_like(replacement_dirs)

        # Assign to dead neurons
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_enc_norm = sae.encoder.weight[~dead_mask].norm(dim=-1).mean()

        # Give resampled neurons a healthy initial magnitude to encourage them to compete
        sae.encoder.weight[dead_indices] = replacement_dirs * alive_enc_norm * 0.8
        sae.encoder.bias[dead_indices] = 0.0
        sae.decoder.weight[:, dead_indices] = replacement_dirs.T
        sae._normalise_decoder()

        # Reset Adam state for resampled neurons
        for param in [sae.encoder.weight, sae.encoder.bias,
                      sae.decoder.weight, sae.decoder.bias]:
            state = opt.state.get(param)
            if state:
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()

        print(f"    ↳ Resampled {n_dead} dead neurons")
        return n_dead

    # ------------------------------------------------------------------
    def get_feature_activations(
        self,
        sae: SparseAutoencoder,
        acts: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        Encode a full activation tensor through a trained SAE.
        Returns latent codes  (N, dict_size)  on CPU.
        """
        sae.eval().to(self.device)
        codes = []
        for i in range(0, len(acts), batch_size):
            chunk = acts[i:i+batch_size].to(self.device)
            with torch.no_grad():
                codes.append(sae.encode(chunk).cpu())
        return torch.cat(codes, dim=0)

    def save(self, path: str):
        state = {
            "saes": {k: v.state_dict() for k, v in self.saes.items()},
            "act_means": self.act_means,
            "act_stds":  self.act_stds,
        }
        torch.save(state, path)
        print(f"Saved SAEs to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        sae_states = state.get("saes", state)  # backwards compat
        for k_layer, sd in sae_states.items():
            sae = SparseAutoencoder(
                self.embed_dim, self.expansion, self.l1_coef,
                mode=self.mode, k=self.k,
            )
            sae.load_state_dict(sd)
            self.saes[k_layer] = sae
        if "act_means" in state:
            self.act_means = state["act_means"]
            self.act_stds  = state["act_stds"]
        print(f"Loaded SAEs for layers: {sorted(self.saes.keys())}")

    # ------------------------------------------------------------------
    def sweep_l1(
        self,
        encoder,  # nn.Module (legacy) OR EncoderBackend
        dataloader: DataLoader,
        l1_values: Optional[List[float]] = None,
        max_tokens: int = 50_000,
        sweep_epochs: int = 10,
        target_layer: Optional[int] = None,
    ) -> List[dict]:
        """
        Quick l1_coef sweep on a token subset (mode='l1' only).
        """
        if l1_values is None:
            l1_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

        acts = self._prepare_sweep_acts(encoder, dataloader,
                                        max_tokens, target_layer)
        layer = target_layer or (max(self._sweep_layer_cache.keys()) if hasattr(self, '_sweep_layer_cache') else 0)

        print(f"Sweep on layer {layer}, {len(acts)} tokens, "
              f"l1 values: {l1_values}\n")

        results = []
        for l1 in l1_values:
            print(f"--- l1={l1:.0e} ---")
            sae = SparseAutoencoder(
                self.embed_dim, self.expansion, l1, mode="l1",
            ).to(self.device)
            sae.init_b_pre(acts.mean(dim=0))
            metrics = self._sweep_train_eval(sae, acts, sweep_epochs)
            metrics["l1_coef"] = l1
            results.append(metrics)
            print()

        print("=== Sweep Summary ===")
        print(f"{'l1':>10s}  {'L0':>8s}  {'Dead%':>8s}  {'R²':>8s}")
        for r in results:
            print(f"{r['l1_coef']:>10.0e}  {r['l0']:>8.1f}  "
                  f"{r['dead_frac']*100:>7.1f}%  {r['r2']:>8.4f}")
        return results

    # ------------------------------------------------------------------
    def sweep_k(
        self,
        encoder,  # nn.Module (legacy) OR EncoderBackend
        dataloader: DataLoader,
        k_values: Optional[List[int]] = None,
        max_tokens: int = 50_000,
        sweep_epochs: int = 10,
        target_layer: Optional[int] = None,
    ) -> List[dict]:
        """
        Quick k sweep on a token subset (mode='topk').
        Trains a small SAE for each candidate k and reports L0 / R² / dead%.
        """
        if k_values is None:
            k_values = [8, 16, 32, 48, 64, 96]

        acts = self._prepare_sweep_acts(encoder, dataloader,
                                        max_tokens, target_layer)
        layer = target_layer or max(self.target_layers) if self.target_layers else 0

        print(f"Sweep on layer {layer}, {len(acts)} tokens, "
              f"k values: {k_values}\n")

        results = []
        for k_val in k_values:
            print(f"--- k={k_val} ---")
            sae = SparseAutoencoder(
                self.embed_dim, self.expansion, self.l1_coef,
                mode="topk", k=k_val,
            ).to(self.device)
            sae.init_b_pre(acts.mean(dim=0))
            metrics = self._sweep_train_eval(sae, acts, sweep_epochs)
            metrics["k"] = k_val
            results.append(metrics)
            print()

        print("=== K Sweep Summary ===")
        print(f"{'k':>6s}  {'L0':>8s}  {'Dead%':>8s}  {'R²':>8s}")
        for r in results:
            print(f"{r['k']:>6d}  {r['l0']:>8.1f}  "
                  f"{r['dead_frac']*100:>7.1f}%  {r['r2']:>8.4f}")
        return results

    # ------------------------------------------------------------------
    def _prepare_sweep_acts(
        self,
        encoder,  # nn.Module (legacy) OR EncoderBackend
        dataloader: DataLoader,
        max_tokens: int,
        target_layer: Optional[int],
    ) -> torch.Tensor:
        """Collect, normalise, and subsample activations for a sweep."""
        print("=== Sweep: collecting activations ===")
        all_acts = self.collect_activations(encoder, dataloader)
        layer = target_layer or (max(all_acts.keys()) if all_acts else 0)
        acts = all_acts[layer]

        # Normalise
        mean = acts.mean(dim=0)
        std  = acts.std(dim=0).clamp(min=1e-8)
        acts = (acts - mean) / std

        # Subsample
        if len(acts) > max_tokens:
            idx = torch.randperm(len(acts))[:max_tokens]
            acts = acts[idx]
        return acts

    def _sweep_train_eval(
        self,
        sae: SparseAutoencoder,
        acts: torch.Tensor,
        sweep_epochs: int,
    ) -> dict:
        """Train an SAE for a few epochs and return diagnostics."""
        opt = torch.optim.Adam(sae.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(acts), batch_size=self.batch_size, shuffle=True)

        for ep in range(sweep_epochs):
            for (x,) in loader:
                x = x.to(self.device)
                loss, *_ = sae.loss(x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae._normalise_decoder()

        return sae_diagnostics(sae.cpu(), acts, device=self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Quick diagnostic utilities
# ─────────────────────────────────────────────────────────────────────────────

def sae_diagnostics(sae: SparseAutoencoder, acts: torch.Tensor, device="cpu"):
    """
    Print basic SAE health metrics on a held-out activation tensor.

    Metrics
    -------
    L0          : average number of active features per token  (want ~1–50)
    Dead neurons: features that never fire across the eval set (want ~0%)
    R²          : reconstruction quality                       (want >0.95)
    """
    sae.eval().to(device)
    # Chunk forward pass to bound peak GPU memory: a single forward on N tokens
    # ×  n_features (= embed_dim × E) can exceed GPU RAM at high E (e.g. E=16
    # gives 8192-dim z and the topk mask doubles peak usage).  Stream batches
    # and accumulate scalars instead.
    chunk = 16384
    n = acts.shape[0]
    l0_sum = 0.0
    feat_fire_count = torch.zeros(int(sae.encoder.weight.shape[0]), device=device)
    ss_res = 0.0
    x_mean = acts.mean(dim=0).to(device)
    ss_tot = 0.0
    with torch.no_grad():
        for i in range(0, n, chunk):
            x_b = acts[i:i + chunk].to(device, non_blocking=True)
            x_hat_b, z_b = sae(x_b)
            l0_sum += (z_b > 0).float().sum(dim=-1).sum().item()
            feat_fire_count += (z_b > 0).float().sum(dim=0)
            ss_res += ((x_b - x_hat_b) ** 2).sum().item()
            ss_tot += ((x_b - x_mean) ** 2).sum().item()
            del x_b, x_hat_b, z_b
        l0        = l0_sum / n
        dead_frac = (feat_fire_count == 0).float().mean().item()
        r2        = 1 - ss_res / (ss_tot + 1e-8)

    print(f"  L0 (avg active features): {l0:.1f}")
    print(f"  Dead neurons:             {dead_frac*100:.1f}%")
    print(f"  R²:                       {r2:.4f}")
    return {"l0": l0, "dead_frac": dead_frac, "r2": r2}
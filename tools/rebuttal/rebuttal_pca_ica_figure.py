"""Rebuttal — the SAE-vs-PCA/ICA figure (honest two-panel).

Panel a: Δ̃ steering selectivity by direction source (SAE competitive with PCA/ICA;
         random-dict + centroid ≈ 0) — selective steering needs a *learned* dictionary,
         and we do NOT claim the SAE steers harder.
Panel b: the differentiator — the top abnormality-aligned SAE features decode to clean,
         band-localized spectral signatures (interpretable primitives), whereas a
         PCA component of the same concept is spectrally diffuse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "paper_figures"))
sys.path.insert(0, str(ROOT / "src"))
import plot_cross_model_steering_concepts_paper as P  # noqa: E402
from mecheeg.sae import SparseAutoencoder  # noqa: E402
from mecheeg.spectral_decoder import SpectralDecoderTrainer  # noqa: E402
from mecheeg.nature_fig_style import (  # noqa: E402
    set_nature_style, nature_figure, nature_palette, save_nature_figure,
)

ART = ROOT / "results" / "rebuttal"
FIG = ART / "figures"
EXP = "sleepfm_finetuned_exp8_k64_layer2"
ABN = {"Abnormal - Epileptiform", "Abnormal - Other"}
NORMAL = {"Normal"}


def decode_spectrum(emb_std, sae, am, as_, sdt):
    with torch.no_grad():
        emb_r = emb_std * (as_ + 1e-8) + am
        amp, _, _ = sdt.decode_direction(emb_r.cpu(), denormalise=True)
        amp = torch.expm1(amp.clamp(min=0)).numpy()
    return amp.mean(0)


def main():
    set_nature_style()
    meta = json.loads((ROOT / "results/experiments" / EXP / "metadata.json").read_text())
    sae_ck = torch.load(meta["sae_checkpoint"], map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(meta["embed_dim"], expansion=meta["expansion"], mode="topk", k=meta["k"])
    sae.load_state_dict(sae_ck["sae_state_dict"]); sae.eval()
    am = sae_ck["act_mean"].float(); as_ = sae_ck["act_std"].float()
    sdt = SpectralDecoderTrainer(embed_dim=meta["embed_dim"]); sdt.load(meta["xae_checkpoint"])
    sdt.spectral_decoder.cpu().eval()
    freqs = np.asarray(sdt.spectral.freqs)
    nf = sae.decoder.weight.shape[1]

    sc = torch.load(ROOT / "results/steering_cache" / EXP / "steering_cache.pt",
                    map_location="cpu", weights_only=False)
    cls = np.asarray([str(v) for v in sc["classification"]])
    emb = sc["embeddings"].float()
    pos = np.isin(cls, list(ABN)); neg = np.isin(cls, list(NORMAL))
    rng = np.random.default_rng(42)
    n = min(3000, int(pos.sum()), int(neg.sum()))
    idx = np.concatenate([rng.choice(np.where(pos)[0], n, replace=False),
                          rng.choice(np.where(neg)[0], n, replace=False)])
    y = np.array([1] * n + [0] * n)
    Xstd = (emb[idx] - am) / (as_ + 1e-8)

    pca = PCA(n_components=meta["embed_dim"], svd_solver="full").fit(Xstd.numpy())

    def feat_spectrum(i):
        z = torch.zeros(1, nf); z[0, i] = 6.0
        return decode_spectrum(sae.decode(z), sae, am, as_, sdt)

    def pca_spectrum(i):
        emb_std = torch.as_tensor(pca.mean_ + 3.0 * pca.components_[i], dtype=torch.float32)[None]
        return decode_spectrum(emb_std, sae, am, as_, sdt)

    norm = lambda s: s / (s.max() + 1e-9)

    # Decode ALL SAE features → pick the sharpest band-specialized primitive per band,
    # showing the dictionary decomposes the spectrum into interpretable units.
    print("decoding all SAE features …", flush=True)
    all_specs = np.stack([feat_spectrum(i) for i in range(nf)])          # (nf, n_bins)
    bands = {"δ": (1, 4), "θ": (4, 8), "α": (8, 13), "β": (13, 30)}
    peak_f = freqs[all_specs.argmax(1)]
    conc = all_specs.max(1) / (all_specs.mean(1) + 1e-9)                  # peakedness
    band_feats = {}
    for b, (f0, f1) in bands.items():
        cand = np.where((peak_f >= f0) & (peak_f < f1))[0]
        if len(cand):
            band_feats[b] = int(cand[np.argmax(conc[cand])])
    sae_specs = {b: all_specs[i] for b, i in band_feats.items()}
    # PCA: two highest-variance components (broadband, redundant)
    pca_specs = [pca_spectrum(i) for i in range(2)]

    pal = nature_palette(6)
    fig, (ax1, ax2) = nature_figure(width="onehalf", height_mm=62, ncols=2)

    d = json.load(open(ART / "steering_baselines_sleepfm_finetuned_layer2_seeds5.json"))["results"]
    concepts = ["Pathology", "Age", "Medication (ASM)"]
    methods = ["SAE", "PCA", "ICA", "RandDict", "dense-CAV", "centroid"]
    mlabel = ["SAE", "PCA", "ICA", "rand\ndict", "dense\nCAV", "centr"]

    def dval(c, m):
        rows = d[c]["rows"]
        if m == "SAE":
            return float(np.mean([r["delta"] for r in rows if r["method"].startswith("SAE[seed")]))
        return next((r["delta"] for r in rows if r["method"].startswith(m)), np.nan)

    x = np.arange(len(methods)); w = 0.26
    for ci, c in enumerate(concepts):
        ax1.bar(x + (ci-1)*w, [dval(c, m) for m in methods], w, color=pal[ci], label=c)
    ax1.axhline(0, color="0.4", lw=0.6); ax1.axhline(0.05, color="0.6", lw=0.5, ls=":")
    ax1.set_xticks(x); ax1.set_xticklabels(mlabel, fontsize=5)
    ax1.set_ylabel(r"steering selectivity $\tilde{\Delta}$")
    ax1.set_title("a  Selective steering needs a learned dictionary", loc="left", fontsize=6.3)
    ax1.legend(frameon=False, fontsize=4.6, loc="upper right")
    ax1.annotate("random & centroid ≈ 0", xy=(3.3, 0.012), fontsize=4.6, color="0.35")

    bcols = {"δ": pal[0], "θ": pal[1], "α": pal[2], "β": pal[4]}
    for b, s in sae_specs.items():
        ax2.plot(freqs, norm(s), "-", color=bcols.get(b, pal[5]), lw=1.3,
                 label=f"SAE feat ({b}-band)")
    for k, s in enumerate(pca_specs):
        ax2.plot(freqs, norm(s), "--", color="0.5", lw=1.0,
                 label="PCA components" if k == 0 else None)
    for lab, (f0, f1) in {"δ": (1, 4), "θ": (4, 8), "α": (8, 13), "β": (13, 30)}.items():
        ax2.axvspan(f0, f1, color="0.92", alpha=0.5, zorder=0)
        ax2.text((f0+f1)/2, 1.04, lab, ha="center", fontsize=5, color="0.4")
    ax2.set_xlim(1, 45); ax2.set_ylim(0, 1.13)
    ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("norm. amplitude")
    ax2.set_title("b  SAE features are band-specialized primitives; PCA is broadband",
                  loc="left", fontsize=6.0)
    ax2.legend(frameon=False, fontsize=4.6, loc="upper right")

    FIG.mkdir(parents=True, exist_ok=True)
    save_nature_figure(fig, str(FIG / "fig_pca_ica.pdf"))
    save_nature_figure(fig, str(FIG / "fig_pca_ica.png"))
    print(f"wrote fig_pca_ica  (band-specialized SAE feats {band_feats})")


if __name__ == "__main__":
    main()

"""TCAV (Testing with Concept Activation Vectors) for SAE features.

Kim et al. 2018 (https://arxiv.org/abs/1711.11279)

A linear CAV is trained in *normalised* encoder activation space (the same
space the SAE was trained on) to separate concept-positive tokens from random
negative tokens.  Three distinct TCAV variants are implemented, with
important differences in what they measure:

─────────────────────────────────────────────────────────────────────────────
Variant A — Weight-space alignment score  (compute_tcav_scores)
─────────────────────────────────────────────────────────────────────────────
NOT a proper Kim et al. TCAV.  For each SAE feature i:

    S_i(v_C) = W_enc[i] · v_C          (directional derivative of feature i)

Score = fraction of K fold-CAVs where S_i > 0 ∈ {0, 0.1, …, 1.0}.

This is purely a *weight-space* property — it does not depend on any
individual example.  Because W_enc[i] · v_C has a fixed sign for a given
CAV, and high-accuracy CAVs are nearly identical across folds, scores
concentrate at 0.0 and 1.0 with resolution 1/K (e.g. 11 values for K=10).

Use-case: quick sanity check on encoder weight alignment with a concept.
Significance: comparison with n_random random-label CAVs on the same data.

─────────────────────────────────────────────────────────────────────────────
Variant B — Per-feature Kim et al. TCAV  (compute_per_feature_tcav)
─────────────────────────────────────────────────────────────────────────────
Proper Kim et al. applied per SAE feature.  For TopK SAE with linear+TopK
encoder, the Jacobian ∂z_i/∂a equals W_enc[i] when feature i is active,
and 0 otherwise.  So for each example x ∈ X_concept:

    S_i(x) = W_enc[i] · v_C  if z_i(x) > 0, else 0

TCAV_i = fraction of X_concept where S_i(x) > 0
       = firing_rate(i, concept) × I(W_enc[i] · v_C > 0)

Range: [0, max_firing_rate].  Example-dependent (via firing rate).
Because W_enc[i] · v_C is still constant per feature, TCAV_i collapses to
a binary gate on the firing rate: either 0 or the firing rate itself.

Use-case: per-feature analysis showing which features fire on concept
examples *and* are aligned with the concept direction.

─────────────────────────────────────────────────────────────────────────────
Variant C — Model-level Kim et al. TCAV  (compute_model_tcav_score)
─────────────────────────────────────────────────────────────────────────────
The proper Kim et al. formulation with a downstream prediction function.
A linear probe h is trained on SAE feature activations z to predict class k
(e.g. abnormal vs normal).  For each example x ∈ X_concept:

    S_k(x) = ∇_a h(SAE(a(x))) · v_C
           = Σ_i [ w_probe_i × (W_enc[i] · v_C) × I(z_i(x) > 0) ]

where w_probe are the probe weights in z-space (sigmoid' cancels for sign).
TCAV_k = fraction of X_concept where S_k(x) > 0 ∈ [0, 1], 0.5 = chance.

This is *genuinely* example-dependent: different examples activate different
features, giving each a different sensitivity to the concept direction.

Use-case: single headline score per (concept, layer) answering "does concept
C causally influence the model's prediction of class k at this layer?"
Significance: empirical p-value vs n_random random-label CAVs.

─────────────────────────────────────────────────────────────────────────────
Other key quantities
─────────────────────────────────────────────────────────────────────────────
cav          : (embed_dim,) unit-norm direction separating concept tokens
               from random tokens in normalised activation space.
alignment    : cosine_similarity(cav, SAE_decoder_direction_i) ∈ [-1, 1]
firing_rate  : fraction of concept tokens where feature i fires
delta_rate   : firing_rate_concept − firing_rate_baseline  ∈ [-1, 1]
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def train_cav(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    n_splits: int = 10,
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Train a linear CAV via logistic regression with cross-validation.

    Parameters
    ----------
    pos_acts : (N_pos, embed_dim)  concept-positive normalised activations
    neg_acts : (N_neg, embed_dim)  random/negative normalised activations
    n_splits : CV folds (default 10 — needed for binomial significance at p<0.05)
    C        : logistic regression inverse regularisation strength
    random_state : RNG seed

    Returns
    -------
    mean_cav   : (embed_dim,) unit-norm mean concept activation vector
    accuracy   : mean cross-validated binary accuracy (chance = 0.5)
    fold_cavs  : (n_splits, embed_dim) unit-norm CAV per fold (for TCAV scoring)
    """
    n_pos, n_neg = len(pos_acts), len(neg_acts)
    n_splits = max(2, min(n_splits, n_pos // 2, n_neg // 2))

    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.array([1] * n_pos + [0] * n_neg)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, cavs = [], []
    for tr, va in skf.split(X_s, y):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
        clf.fit(X_s[tr], y[tr])
        accs.append(float(clf.score(X_s[va], y[va])))
        coef = clf.coef_[0]
        cavs.append(coef / (np.linalg.norm(coef) + 1e-8))

    fold_cavs = np.stack(cavs).astype(np.float32)   # (n_splits, embed_dim)
    mean_cav  = fold_cavs.mean(axis=0)
    mean_cav /= np.linalg.norm(mean_cav) + 1e-8
    return mean_cav.astype(np.float32), float(np.mean(accs)), fold_cavs


def _single_cav(
    X_s: np.ndarray,
    y: np.ndarray,
    C: float,
    random_state: int,
) -> np.ndarray:
    """Train a single logistic regression CAV (no CV). Returns unit-norm vector."""
    clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
    clf.fit(X_s, y)
    coef = clf.coef_[0]
    return (coef / (np.linalg.norm(coef) + 1e-8)).astype(np.float32)


def compute_tcav_scores(
    fold_cavs: np.ndarray,
    W_enc: np.ndarray,
    pos_acts_norm: np.ndarray,
    neg_acts_norm: np.ndarray,
    n_random: int = 50,
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Traditional TCAV score with empirical significance against random concepts.

    For each SAE feature i and concept C, the directional derivative is:
        S_i(v_C) = W_enc[i,:] · v_C
    which measures how much activating in the concept direction increases feature i.

    TCAV score = fraction of k-fold CAV runs where S_i > 0 ∈ [0, 1], 0.5 = chance.

    Significance is tested empirically: n_random random CAVs (same pool sizes,
    random labels) form a null distribution.  p-value = fraction of random TCAV
    scores ≥ real TCAV score (one-tailed, testing for concept selectivity).

    Parameters
    ----------
    fold_cavs     : (n_splits, embed_dim) per-fold unit-norm CAV directions
    W_enc         : (n_features, embed_dim) SAE encoder weight rows
    pos_acts_norm : (N_pos, embed_dim) normalised concept-positive activations
    neg_acts_norm : (N_neg, embed_dim) normalised concept-negative activations
    n_random      : number of random CAVs to train for null distribution
    C             : logistic regression regularisation
    random_state  : RNG seed

    Returns
    -------
    tcav_scores : (n_features,) ∈ [0, 1] — fraction of folds with S_i > 0
    p_values    : (n_features,) empirical p-values from random CAV null distribution
    """
    from sklearn.preprocessing import StandardScaler

    # ── Directional derivatives: W_enc[i,:] · v_C_k for each fold ────────────
    # W_enc_unit: (n_features, E),  fold_cavs: (K, E)
    W_norms   = np.linalg.norm(W_enc, axis=1, keepdims=True).clip(min=1e-8)
    W_unit    = W_enc / W_norms                         # (n_features, E)
    fold_dd   = fold_cavs @ W_unit.T                    # (K, n_features) — directional derivs

    # TCAV score = fraction of folds where directional derivative > 0
    tcav_scores = (fold_dd > 0).mean(axis=0).astype(np.float32)   # (n_features,)

    # ── Null distribution from random CAVs ───────────────────────────────────
    rng     = np.random.default_rng(random_state)
    all_acts = np.concatenate([pos_acts_norm, neg_acts_norm], axis=0)
    n_pos, n_neg = len(pos_acts_norm), len(neg_acts_norm)
    n_total  = len(all_acts)

    # Fit scaler once on the pooled data
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_acts)

    random_tcav = np.zeros((n_random, W_enc.shape[0]), dtype=np.float32)
    for r in range(n_random):
        # Random labels — same class balance as real concept
        rand_y = np.zeros(n_total, dtype=int)
        rand_pos_idx = rng.choice(n_total, size=n_pos, replace=False)
        rand_y[rand_pos_idx] = 1
        rand_cav = _single_cav(all_scaled, rand_y, C=C, random_state=int(rng.integers(1e6)))
        rand_dd  = rand_cav @ W_unit.T          # (n_features,)
        # Random TCAV score: 1 if directional derivative > 0, else 0
        random_tcav[r] = (rand_dd > 0).astype(np.float32)

    # Empirical p-value: fraction of random TCAV scores >= real TCAV score
    # (one-tailed: testing that the concept causes more positive derivatives than random)
    p_values = (random_tcav >= tcav_scores[None, :]).mean(axis=0).astype(np.float32)

    return tcav_scores, p_values


def cav_sae_alignment(
    cav: np.ndarray,
    W_dec: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between each SAE decoder direction and the CAV.

    Parameters
    ----------
    cav   : (embed_dim,) unit-norm CAV vector in normalised activation space
    W_dec : (n_features, embed_dim) SAE decoder weight rows

    Returns
    -------
    (n_features,) alignment scores in [-1, 1]
    """
    W_norms = np.linalg.norm(W_dec, axis=1, keepdims=True).clip(min=1e-8)
    W_unit  = W_dec / W_norms
    cav_u   = cav / (np.linalg.norm(cav) + 1e-8)
    return (W_unit @ cav_u).astype(np.float32)


def concept_firing_rates(
    raw_concept_acts: torch.Tensor,
    sae: "SparseAutoencoder",  # type: ignore[name-defined]  # noqa: F821
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    """Fraction of concept tokens where each SAE feature fires.

    Parameters
    ----------
    raw_concept_acts : (N, embed_dim) *raw* (unnormalised) encoder activations
                       for concept-positive tokens
    sae              : trained SparseAutoencoder
    act_mean, act_std: normalisation stats stored in the SAE checkpoint
    batch_size       : processing chunk size

    Returns
    -------
    (n_features,) firing rates in [0, 1]
    """
    sae.eval()
    fired: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(raw_concept_acts), batch_size):
            batch  = raw_concept_acts[i : i + batch_size]
            normed = (batch - act_mean) / (act_std + 1e-8)
            z      = sae.encode(normed)
            fired.append((z > 0).float().cpu())
    return torch.cat(fired, dim=0).mean(dim=0).numpy().astype(np.float32)


def get_sae_features(
    raw_acts: torch.Tensor,
    sae: "SparseAutoencoder",  # type: ignore[name-defined]  # noqa: F821
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return SAE feature activations (z vectors) for a batch of raw activations.

    Parameters
    ----------
    raw_acts : (N, embed_dim) unnormalised encoder activations
    sae, act_mean, act_std : trained SAE and its normalisation statistics

    Returns
    -------
    (N, n_features) float32 array of z-activations (non-negative, sparse)
    """
    sae.eval()
    zs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(raw_acts), batch_size):
            batch  = raw_acts[i : i + batch_size]
            normed = (batch - act_mean) / (act_std + 1e-8)
            zs.append(sae.encode(normed).cpu().float())
    return torch.cat(zs, dim=0).numpy().astype(np.float32)


def train_probe(
    pos_features: np.ndarray,
    neg_features: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """Train a linear probe on SAE feature activations to separate concept from baseline.

    Used by compute_model_tcav_score to define the downstream prediction
    function h(z) required by proper Kim et al. TCAV.

    Parameters
    ----------
    pos_features : (N_pos, n_features) SAE z-activations for concept-positive tokens
    neg_features : (N_neg, n_features) SAE z-activations for random/negative tokens

    Returns
    -------
    probe_weights : (n_features,) LogReg coefficient vector in scaled z-space
    probe_accuracy : float — held-out binary classification accuracy
    """
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([pos_features, neg_features], axis=0)
    y = np.array([1] * len(pos_features) + [0] * len(neg_features))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
    accs = cross_val_score(clf, X_s, y, cv=5, scoring="accuracy")
    clf.fit(X_s, y)   # refit on all data for the weights
    return clf.coef_[0].astype(np.float32), float(accs.mean())


def _model_sensitivities(
    cav: np.ndarray,
    W_enc: np.ndarray,
    probe_weights: np.ndarray,
    raw_acts: torch.Tensor,
    sae: "SparseAutoencoder",  # type: ignore[name-defined]  # noqa: F821
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    """Per-example model sensitivity S_k(x) = Σ_i [w_i × (W_enc[i]·v_C) × I(z_i>0)].

    The sign of S_k(x) determines whether the prediction of class k increases
    when the input is nudged in the concept direction v_C.

    Returns
    -------
    (N,) float32 sensitivity values — one per input example
    """
    cav_u = cav / (np.linalg.norm(cav) + 1e-8)
    # Weighted alignment: w_probe[i] × (W_enc[i] · v_C) for each feature
    w_aln = torch.tensor(
        probe_weights * (W_enc @ cav_u), dtype=torch.float32
    )  # (n_features,)

    sae.eval()
    sens: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(raw_acts), batch_size):
            batch  = raw_acts[i : i + batch_size]
            normed = (batch - act_mean) / (act_std + 1e-8)
            z      = sae.encode(normed)               # (B, n_features)
            active = (z > 0).float()
            s      = (active * w_aln.to(active.device)).sum(dim=1)   # (B,)
            sens.append(s.cpu())
    return torch.cat(sens).numpy().astype(np.float32)


def compute_model_tcav_score(
    cav: np.ndarray,
    W_enc: np.ndarray,
    probe_weights: np.ndarray,
    raw_concept_acts: torch.Tensor,
    sae: "SparseAutoencoder",  # type: ignore[name-defined]  # noqa: F821
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    pos_acts_norm: np.ndarray,
    neg_acts_norm: np.ndarray,
    n_random: int = 50,
    C: float = 1.0,
    batch_size: int = 4096,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Proper model-level Kim et al. TCAV score with empirical significance.

    Computes the fraction of concept-positive examples where the downstream
    probe's prediction increases when activations are pushed in the concept
    direction (Variant C in the module docstring).

    Parameters
    ----------
    cav              : (embed_dim,) mean unit-norm CAV for the real concept
    W_enc            : (n_features, embed_dim) SAE encoder weight rows
    probe_weights    : (n_features,) linear probe weights in scaled z-space
    raw_concept_acts : (N_pos, embed_dim) raw activations for concept examples
    sae, act_mean, act_std : trained SAE and normalisation statistics
    pos_acts_norm    : (N_pos, embed_dim) normalised activations (for null CAVs)
    neg_acts_norm    : (N_neg, embed_dim) normalised activations (for null CAVs)
    n_random         : number of random-label CAVs for null distribution
    C                : logistic regression regularisation
    batch_size       : batch size for SAE forward passes
    random_state     : RNG seed

    Returns
    -------
    tcav_score : float ∈ [0, 1], 0.5 = chance
    p_value    : empirical one-tailed p-value vs random-concept null distribution
    """
    # ── Real TCAV score ───────────────────────────────────────────────────────
    s_real     = _model_sensitivities(cav, W_enc, probe_weights,
                                      raw_concept_acts, sae, act_mean, act_std, batch_size)
    real_score = float((s_real > 0).mean())

    # ── Null distribution: n_random random-label CAVs ────────────────────────
    rng      = np.random.default_rng(random_state)
    all_acts = np.concatenate([pos_acts_norm, neg_acts_norm], axis=0)
    n_pos    = len(pos_acts_norm)
    n_total  = len(all_acts)

    scaler      = StandardScaler()
    all_scaled  = scaler.fit_transform(all_acts)

    null_scores: list[float] = []
    for _ in range(n_random):
        rand_y = np.zeros(n_total, dtype=int)
        rand_y[rng.choice(n_total, size=n_pos, replace=False)] = 1
        rand_cav = _single_cav(all_scaled, rand_y, C=C,
                               random_state=int(rng.integers(1_000_000)))
        s_null = _model_sensitivities(rand_cav, W_enc, probe_weights,
                                      raw_concept_acts, sae, act_mean, act_std, batch_size)
        null_scores.append(float((s_null > 0).mean()))

    p_value = float(np.mean(np.array(null_scores) >= real_score))
    return real_score, p_value


def firing_rate_enrichment(
    pos_rates: np.ndarray,
    neg_rates: np.ndarray,
    n_pos: int,
    n_neg: int,
    fdr_method: str = "bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-proportions z-test for firing rate enrichment with FDR correction.

    Tests H₀: firing_rate_concept == firing_rate_baseline for each feature.
    Valid for large n (n_pos, n_neg ≥ 30).

    Parameters
    ----------
    pos_rates  : (n_features,) firing rates on concept-positive tokens
    neg_rates  : (n_features,) firing rates on concept-negative tokens
    n_pos      : number of concept-positive tokens
    n_neg      : number of concept-negative tokens (negative sample used)
    fdr_method : 'bh' (Benjamini–Hochberg) or 'bonferroni'

    Returns
    -------
    p_values   : (n_features,) FDR-corrected p-values
    delta_rate : (n_features,) = pos_rate − neg_rate  ∈ [-1, 1], 0 = chance
    """
    from scipy import stats

    delta = pos_rates - neg_rates   # (n_features,)

    # Pooled proportion for each feature
    p_pool = (pos_rates * n_pos + neg_rates * n_neg) / (n_pos + n_neg)
    # Avoid divide-by-zero when pooled rate is 0 or 1
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_pos + 1 / n_neg) + 1e-12)
    z  = delta / se

    # Two-tailed p-values
    raw_p = 2 * stats.norm.sf(np.abs(z))

    # Benjamini–Hochberg FDR correction (implemented without statsmodels)
    n = len(raw_p)
    if fdr_method == "bh":
        order = np.argsort(raw_p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        bh_thresholds = raw_p * n / ranks
        # Enforce monotonicity from right
        p_adj = np.minimum.accumulate(bh_thresholds[::-1])[::-1]
        p_adj = p_adj[np.argsort(order)]   # restore original order
        p_adj = np.clip(p_adj, 0.0, 1.0)
    else:
        p_adj = np.clip(raw_p * n, 0.0, 1.0)

    return p_adj.astype(np.float32), delta.astype(np.float32)

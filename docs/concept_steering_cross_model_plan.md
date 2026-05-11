# Cross-model concept steering comparison — plan

**Goal:** Characterise how SAE-based concept steering transfers across EEG
foundation models. Three models: SleepFM (done — strong steering), REVE (done
— concepts present but orthogonal to spectral manifold), LaBraM (medium effort
— weights available). A random baseline anchors the comparison statistically.

## Diagnostic finding (2026-04-29) — REVE concepts are spectrally orthogonal

The original "architecture independence" framing turned out to be an
oversimplification. Three steering pathways were tested on REVE layer 21:

| Pathway | CAV accuracy | M1 (α=1, n=50) | M2 (α=1, n=50) |
|---|---|---|---|
| SAE feature substitution (top-N enriched) | — | 0.97–1.01 | 0.30–0.37 |
| Direction in encoder space, decoded via SAE+XAE | 0.77 | 1.01 | 0.34 |
| Direction in encoder space, **decoded XAE-only** | **1.00** | 1.01 | **0.78–0.96** |

**Interpretation:** REVE represents abnormality/age/classification in
encoder dimensions that are perfectly linearly separable (CAV=1.00) — the
concepts are present and discriminable. But these dimensions are nearly
**orthogonal to the XAE's spectrally-decodable directions**. Steering along
the concept axis moves the embedding into a different concept class (M2 → 1)
without changing the spectrum (M1 ≈ 1).

This contrasts with SleepFM where concept directions are spectrally grounded
(M1 drops to 0.50–0.60 with strong M2 increase). Likely cause: SleepFM was
finetuned end-to-end on binary labels, forcing concept ↔ spectrum alignment;
REVE was pretrained for masked reconstruction and the layer-21 finetune may
have routed concepts through non-spectral channels.

---

## 1. Scientific claim

> SAE features capture model-internal concept directions that can be used to
> predictably shift a recording's neural representation toward a target clinical
> state. The effect is architecture-independent and significantly stronger than
> steering in a random direction.

This reframes steering from a qualitative demo into a quantitative,
cross-architecture result.

---

## 2. Steering protocol (identical across models)

For each model + concept pair:

1. **Source group** (e.g. child, N=50 recordings)  
   **Target group** (e.g. adult-normal reference)

2. Load SAE features for source tokens. Identify the top-K SAE features most
   enriched in source vs target (from the TCAV/enrichment cache). The steering
   direction is derived from SAE feature space (CAV trained on SAE z-activations,
   projected back to encoder space via the SAE decoder weight matrix).

3. **Real steering:** Project source embeddings by subtracting `α × v_concept`
   (CAV direction, scaled so the projected embedding lies on the target-group
   manifold). Vary α = {0.25, 0.5, 0.75, 1.0 × ‖shift needed‖}.

4. **Random baseline:** Replace `v_concept` with a random unit vector sampled
   from the null space of `v_concept` (same magnitude, orthogonal to the real
   direction). N=100 directions — sufficient given concentration of measure in
   high-dimensional spaces (512D for REVE/LaBraM). Report mean ± SD.

5. Decode steered embedding through XAE → spectral profile.
   **REVE and LaBraM:** average across channels before reporting (Option A —
   matches SleepFM's channel-averaged output, keeps comparison apples-to-apples).

6. Score with metrics (see §5).

---

## 3. Concepts to steer (all four, all models)

| Concept | Direction | Models |
|---|---|---|
| **Child → adult-normal** | Age subspace projection | SleepFM ✅, REVE, LaBraM |
| **Abnormal → normal** | Abnormality CAV | SleepFM ✅, REVE, LaBraM |
| **ASM medication removal** | ASM CAV | SleepFM ✅, REVE, LaBraM |
| **Seizure indication** | Seizure CAV | SleepFM ✅, REVE, LaBraM |
| **Child-abnormal → child-normal** | Pathological δ removal | SleepFM ✅, REVE, LaBraM |

The age steering + child-abnormal case together reproduce the asymmetric
entanglement finding — the most novel result.

---

## 4. Per-model status and effort

### SleepFM (done)
- Steering cache: `results/steering_cache/sleepfm_finetuned_layer2/`
- All scripts exist: `build_steering_cache.py`, `plot_concept_steering_xae_all.py`
- Figures: `paper/figures/concept_steering_xae_age_combined.png`
- **Remaining:** add random baseline metrics to existing script; compute
  concept AUROC change (currently only spectral plots, no scalar metric).

### REVE (medium effort ~1 week)

What already exists:
- SAE + XAE + app caches + TCAV (complete across all 6 layers)
- `build_steering_cache.py` needs `--experiment reve_qjbe08_layer21` support

What needs to change:
- `plot_concept_steering_xae_all.py` needs encoder-agnostic experiment flag
- XAE output: average across 19 channels before computing spectral distance
- Steering for child → adult requires metadata age labels (present in app cache)

### LaBraM (medium effort ~1–2 weeks)

Weights are available (finetuned), but the backend is not yet in the codebase.
Steps:
1. Implement `LaBraMBackend` in `src/sae4eeg/encoders.py`.
2. Register in all tools (`_ENCODER_DATA`, `_V2_CHECKPOINTS`, etc.) — ~8 files.
3. Train SAE, XAE, build codebook, app cache, run TCAV.
4. Build steering cache and run steering scripts.

---

## 5. Evaluation metrics

### M1 — Spectral distance ratio

```
R = d(steered → target_mean) / d(source → target_mean)
```
Measures how much the spectral gap to the target group closes. R=0 = perfect
steering; R=1 = no change. Random baseline gives R_rand ~ 1.0 ± small SD.

### M2 — Concept probe AUROC change (SAE-feature-based)

Train a linear probe on SAE z-features (already done for SleepFM in
`probe_sae_reconstruction.py`). Report AUROC before/after steering.
- Real steering should push abnormal samples toward normal probe outputs.
- Random baseline: AUROC change ≈ 0.

### M3 — Entanglement signature (age steering only)

For child-abnormal steering: after removing age direction, how much residual
pathological δ remains above the child-normal reference band? Metric: mean δ
power excess (steered − child-normal mean), normalised by baseline excess.

---

## 6. Paper figure structure

**Figure A — Bar chart: scalar metrics across models and concepts**

X-axis: concepts (child→adult, abnormal→normal, ASM, seizure, child-abnormal).
For each concept: 3 grouped bars (SleepFM, REVE, LaBraM), showing M1 and M2.
Error bars: ± 1 SD of the random baseline. Real steering bar clearly above
random baseline demonstrates architecture-independence.

**Figure B — Steering spectra (qualitative, representative)**

One row per model, one column per concept. Each cell: XAE spectra at n=0,
n=20, n=50 steering steps + target-group reference band (shaded). Random
baseline shown as grey shaded region.

**Figure C — Entanglement comparison**

M3 for SleepFM, REVE, LaBraM side by side. The key asymmetry claim:
child-abnormal cannot be fully steered regardless of model.

---

## 7. Random baseline implementation

Add `--random-baseline N` flag (default N=100) to `build_steering_cache.py`
and the steering plot scripts. N=100 is sufficient — in 512-D space, random
directions concentrate tightly near the sphere equator so variance is small
regardless of N. N=1000 would give ~3× tighter CI at 10× cost.

```python
for _ in range(N):  # default N=100
    random_dir = torch.randn_like(cav_direction)
    random_dir -= (random_dir @ cav_direction) * cav_direction  # orthogonalise
    random_dir /= random_dir.norm()
    steered = steer(embedding, alpha * random_dir)
    record_metrics(steered)
```

Aggregate as mean ± 1 SD across the 100 random draws.

---

## 8. Priority order

| Priority | Task | Effort | Blocker |
|---|---|---|---|
| 1 | Add random baseline + M1/M2 to SleepFM steering | 2 days | None |
| 2 | Adapt steering scripts for REVE (channel-avg XAE) | 3 days | None — TCAV done |
| 3 | REVE steering figures + metrics | 2 days | #2 |
| 4 | LaBraM backend + pipeline | 1–2 weeks | None — weights available |
| 5 | LaBraM steering + final bar chart | 2 days | #4 |

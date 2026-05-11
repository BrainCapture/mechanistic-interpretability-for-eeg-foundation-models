# Granular Annotation Training Plan

## Context

V4 data (`data/D4-v4-preprocessed-10s/`) has 10 structured label classes (integers 0–9 in `labels`
field, defined in `preprocessing_report_20260422_140927.csv`). The 317 fine-grained text
descriptions in `f.attrs['descriptions']` are sub-descriptions that collapse into these 10 classes.

**The `labels` field (not `metadata/classification`) is the source of truth for TCAV concepts.**

**Training decision is still open.** Options:
- Keep the existing binary encoder (normal/abnormal) and add V4 as a richer TCAV concept source
- Retrain with binary supervision on V4 (normal vs any-abnormal)
- Retrain with multi-class supervision on V4 (one of the 10 classes)

Do not assume either training approach until decided. The TCAV analysis can be done either way.

---

## Dataset: D4-v4-preprocessed-10s

| Property | Value |
|---|---|
| Files | 94 HDF5 files (`data_1.hdf5`…`data_87.hdf5`, sparse numbering) |
| Windows | 35,920 total (10-second windows) |
| Shape per file | `(N, 27, 2560)` float32 — 27 channels × 2560 samples |
| Sample rate | **256 Hz** (2560 samples = 10 s × 256 Hz) |
| Labels | `labels`: integer 0–9, mapping defined in `preprocessing_report_20260422_140927.csv` |
| Fine-grained descriptions | `f.attrs['descriptions']`: 317 text strings; subcategories that map into the 10 classes |
| Metadata | `metadata/` group: ~40 fields (age, age_group, gender, classification, indication_group, medication_group, pdr_*, recording_state, etc.) |

### The 10 label classes (from preprocessing_report CSV)

| Label | Class name | Windows | % |
|---|---|---|---|
| 0 | normal | 30,060 | 83.7% |
| 1 | interictal activity: diffuse slowing | 464 | 1.3% |
| 2 | interictal activity: focal slowing | 164 | 0.5% |
| 3 | interictal activity: focal sharp waves | 432 | 1.2% |
| 4 | interictal activity: focal spike and wave | 1,986 | 5.5% |
| 5 | interictal activity: generalized spike and wave activity | 1,203 | 3.4% |
| 6 | interictal activity: generalized spike, polyspike and wave | 936 | 2.6% |
| 7 | interictal activity: generalized sharp waves | 188 | 0.5% |
| 8 | interictal activity: burst suppression pattern | 29 | 0.1% |
| 9 | epileptic seizure | 458 | 1.3% |

At 10 tokens per 10s window (256 Hz → 128 Hz resample → 10 × 1s patches):
- Sufficient for TCAV: labels 1, 3, 4, 5, 6, 9 (>400 windows = >4000 tokens)
- Marginal: label 2 (164 windows = 1640 tokens), label 7 (188 = 1880 tokens)
- Sparse: label 8 (29 windows = 290 tokens — may fail TCAV; consider combining with 1)

### Differences from V3

| | V3 (`D4-v3-preprocessed-v2/`) | V4 (`D4-v4-preprocessed-10s/`) |
|---|---|---|
| Sample rate | 128 Hz | **256 Hz** — must resample to 128 Hz for existing SleepFM encoder |
| Window size | 1-second patches | **10-second windows** → split into 10 × 1s patches at load time |
| Label classes | 2 (normal/abnormal) | **10 structured classes** from CSV |
| Fine-grained labels | ~20 | **317** descriptions (sub-categories; use integer class, not these) |
| Metadata fields | age_group, indication_group, medication_group, classification | Same + pdr_*, recording_state, rested_status, heart_rate_*, 40 fields total |
| Total tokens (1s) | ~434K | ~359K (35,920 × 10) |

---

## What needs to change in the code

### 1. Data loading (new or adapted loader for V4)

V4 windows are 10 seconds at 256 Hz. Existing pipeline expects 1-second tokens at 128 Hz.
At load / feature-extraction time:
```python
from scipy.signal import resample_poly

def load_v4_window(eeg_256hz):
    """(27, 2560) float32 → list of 10 × (27, 128) float32 patches"""
    eeg_128hz = resample_poly(eeg_256hz, up=1, down=2, axis=-1)  # (27, 1280)
    patches = [eeg_128hz[:, i*128:(i+1)*128] for i in range(10)]
    return patches  # 10 × (27, 128)
```

**Normalization before encoding** (same as V3):
```python
EEG_CLIP, EEG_STD = 5e-5, 1e-5
patch_norm = np.clip(patch, -EEG_CLIP, EEG_CLIP) / EEG_STD
```

### 2. TCAV concepts (the primary goal)

The 10 label classes become TCAV concepts. Add to `_META_CONCEPTS` in `tools/run_tcav.py`:

```python
# V4 label-based concepts (label index in the 'labels' field)
_V4_LABEL_CONCEPTS = [
    # (concept_name, label_index, min_tokens)
    ("diffuse_slowing",     1, 500),
    ("focal_slowing",       2, 500),   # marginal — may skip
    ("focal_sharp",         3, 500),
    ("focal_spike_wave",    4, 500),
    ("gen_spike_wave",      5, 500),   # generalized spike-and-wave activity
    ("gen_poly_spike_wave", 6, 500),   # generalized polyspike-and-wave
    ("gen_sharp_waves",     7, 500),   # marginal
    ("burst_suppression",   8, 200),   # sparse — may need to combine with diffuse_slowing
    ("seizure",             9, 500),
]
```

For each concept `i`, pos_mask = `token_labels == i`, neg_mask = `token_labels == 0` (normal).

This requires `run_tcav.py` (and `build_app_cache.py`) to capture the per-token `labels` field
from V4 data, not just the metadata fields.

### 3. New experiment pointing at V4

Create `results/experiments/sleepfm_finetuned_v4_layer2/metadata.json` reusing the existing
SAE + XAE + codebook checkpoints but with V4 as the data source:

```json
{
  "name": "sleepfm_finetuned_v4_layer2",
  "display_name": "SLEEPFM finetuned · layer 2 (V4 data)",
  "encoder": "sleepfm_finetuned",
  "embed_dim": 128, "fs": 128, "patch_size": 128, "target_layer": 2,
  "expansion": 1.0, "k": 8, "n_features": 128, "n_clusters": 200,
  "sae_checkpoint": "results/features/sleepfm_finetuned/sae_sleepfm_exp1.0_k8_layer2.pt",
  "xae_checkpoint": "results/xae/sleepfm_finetuned/xae_checkpoint.pt",
  "codebook_path": "results/xae/sleepfm_finetuned/codebook/codebook.pt",
  "feature_explanations": "results/xae/sleepfm_finetuned/explanations/feature_explanations.json",
  "weights_path": "checkpoints/finetuned/sleepfm1.ckpt",
  "data_path": "data/D4-v4-preprocessed-10s"   ← new field; build_app_cache must respect this
}
```

### 4. `build_app_cache.py` changes

- Support `data_path` override from metadata.json (so V4 experiment uses V4 data)
- Handle 256 Hz → 128 Hz resampling when loading V4 windows
- Store the `labels` integer per token in the cache (currently only metadata fields are stored)

### 5. `run_tcav.py` changes

- Add `_V4_LABEL_CONCEPTS` concept list (driven by `labels` field, not metadata field)
- Skip concepts with fewer than `min_tokens` positive examples
- Update `plot_tcav_layer_emergence.py` `CLINICAL_CONCEPTS` with the new concept names

---

## Execution plan (Stage 1 — COMPLETE ✓)

```bash
# ✓ Built V4 app caches (all 3 layers, 2026-04-27)
uv run tools/build_app_cache.py --experiment sleepfm_finetuned_v4_layer{0,1,2}

# ✓ Ran TCAV with label-class concepts (all 3 layers, 2026-04-27)
for L in 0 1 2; do
  uv run tools/run_tcav.py --experiment sleepfm_finetuned_v4_layer${L}
done

# ✓ Generated V4 TCAV layer emergence plot
uv run tools/plot_tcav_layer_emergence.py --encoder sleepfm_finetuned_v4

# ✓ Ran encoder vs SAE binary probe + confusion matrix on V4 labels
uv run tools/probe_v4_labels.py
```

**Outcome:** Stage 1 null for slowing subtypes → Stage 2 retraining confirmed.

---

## Stage 1 — Results (completed 2026-04-27)

Full findings: `docs/stage1_v4_tcav_findings.typ`

### What Stage 1 showed

**Binary probe AUROC (encoder vs SAE, layer 2):**
All classes achieve high encoder AUROC (0.86–0.98 vs normal). SAE preserves
this within 0.001–0.094 — the information is retained through the bottleneck.

**TCAV at layer 2 — significant (p<0.05):**
- Epileptiform: focal_sharp_waves, focal_spike_wave, gen_spike_wave, seizure
- Metadata: abnormal, child, asm, seizure

**TCAV at layer 2 — NOT significant:**
- **diffuse_slowing (p=0.26)** ← primary gap
- **focal_slowing (p=0.06)** ← primary gap
- gen_polyspike_wave (p=0.20), gen_sharp_waves (p=0.16), theta (p=0.42)

**Confusion matrix:** focal_spike_wave is the dominant attractor for all
epileptiform classes. The encoder does not separate morphological subtypes from
each other — only from normal.

### Training decision (resolved)

Stage 1 shows the binary-trained encoder already encodes epileptiform concepts
(spikes, seizure) but is **blind to slowing subtypes** (diffuse/focal slowing).
The key gap is not epileptiform separation — it is the inability to place
diffuse vs focal slowing at distinct positions in representation space.

**Decision: retrain with multi-class supervision on V4 (all 10 classes).**

Rationale:
- Binary retraining would preserve the current gap (slowing still collapses to
  a single abnormality direction).
- Multi-class supervision forces the encoder to place each subtype at a distinct
  representation, giving TCAV and the SAE independent directions to work with.
- The existing SAE+XAE pipeline re-runs unchanged on the new encoder.

---

## Stage 2 — Encoder retraining (multi-class, V4)

Training decision: **multi-class cross-entropy on V4 10-class labels**.

### What to expect after Stage 2
- diffuse_slowing and focal_slowing should become significant TCAV concepts
- Confusion matrix diagonal should sharpen for slowing subtypes
- Epileptiform TCAV scores should remain high (they were already encoded)
- Age/metadata concepts should persist (learned in pre-training, not label-dependent)

### Encoder name
Use `sleepfm_granular` for all registrations and experiment directories.
Checkpoint path: `checkpoints/granular/sleepfm_granular.ckpt` (to be created).

### Training script
Does not exist yet — needs to be written. Base it on the existing SleepFM
finetuning code. Key differences from binary finetuning:
- Loss: `nn.CrossEntropyLoss()` over 10 classes (not binary BCE)
- Dataset: `D4-v4-preprocessed-10s` with `V4ResampleTransform`
- Label field: `labels` integer (0–9), not `metadata/classification`
- Consider class-weighted loss (class 0 is 83.7% of windows)
- Suggested: freeze pre-trained transformer, train only classification head
  first, then unfreeze for fine-tuning (standard transfer learning protocol)

### Once checkpoint is available, consult the encoder registration steps:

**Step 1 — Register encoder** (`src/sae4eeg/encoders.py`): add to `MODEL_CARDS` + name set.

**Step 2 — Register in all tools** (8 files):

| File | Dict/list |
|---|---|
| `tools/train_sae_layers.py` | `_V2_CHECKPOINTS`, `_ENCODER_DATA`, `_all_encoders` |
| `tools/train_xae.py` | `_V2_CHECKPOINTS`, `_ENCODER_CFG`, `_all_encoders` |
| `tools/build_codebook.py` | `_V2_CHECKPOINTS`, `_ENCODER_DATA`, `_ENCODER_FS`, `_ENCODER_EMBED`, `_ENCODER_PATCH`, `_all_encoders` |
| `tools/compare_xae.py` | `MODELS` |
| `tools/build_layer_umap_cache.py` | `ENCODER_CONFIGS` |
| `tools/build_app_cache.py` | `_ENCODER_DATA` |
| `tools/run_tcav.py` | `_ENCODER_DATA` |
| `tools/feature_maximization.py` | `_ENCODER_DATA` |

**Step 3 — Experiment metadata** (`results/experiments/{ENCODER}_layer{L}/metadata.json`).

**Step 4 — Full pipeline:**
```bash
uv run tools/train_sae_layers.py --encoder $ENCODER
uv run tools/train_xae.py --encoder $ENCODER
uv run tools/build_codebook.py --encoder $ENCODER
for L in 0 1 2; do
  uv run tools/build_app_cache.py --experiment ${ENCODER}_layer${L}
  uv run tools/run_tcav.py --experiment ${ENCODER}_layer${L}
done
uv run tools/build_layer_umap_cache.py --encoder $ENCODER
for L in 0 1 2; do
  uv run tools/feature_maximization.py --experiment ${ENCODER}_layer${L} \
    --n-features 128 --n-steps 400 --ws-proximity 0.05 --spectral-coef 0.3 --sinusoid-blend 0.4
done
```

---

## TCAV concept summary (intended final state)

| Concept | Source field | Pos | Neg |
|---|---|---|---|
| delta, theta, alpha, low-beta, high-beta, gamma | spectral (codebook bands) | — | — |
| abnormal | metadata/classification | Abnormal-* | Normal |
| child | metadata/age_group | 0-3, 4-9 | 40-49, 50-59, 60+ |
| asm | metadata/medication_group | ASM | (all non-ASM) |
| seizure | metadata/indication_group | Consciousness (loss of), Drop (attacks/spells) | Non-Symptom/Follow-up |
| **diffuse_slowing** | **labels == 1** | **label 1** | **label 0** |
| **focal_slowing** | **labels == 2** | **label 2** | **label 0** |
| **focal_sharp** | **labels == 3** | **label 3** | **label 0** |
| **focal_spike_wave** | **labels == 4** | **label 4** | **label 0** |
| **gen_spike_wave** | **labels == 5** | **label 5** | **label 0** |
| **gen_poly_spike_wave** | **labels == 6** | **label 6** | **label 0** |
| **gen_sharp_waves** | **labels == 7** | **label 7** | **label 0** |
| **burst_suppression** | **labels == 8** | **label 8** | **label 0** |
| **seizure_ictal** | **labels == 9** | **label 9** | **label 0** |

---

## Critical normalization / resampling

```python
from scipy.signal import resample_poly
import numpy as np

EEG_CLIP, EEG_STD = 5e-5, 1e-5

def prepare_v4_window(eeg_256hz: np.ndarray) -> list[np.ndarray]:
    """(27, 2560) → 10 normalized patches of shape (27, 128) ready for SleepFM encoder."""
    eeg_128hz = resample_poly(eeg_256hz, up=1, down=2, axis=-1)  # (27, 1280)
    patches = []
    for i in range(10):
        p = eeg_128hz[:, i*128:(i+1)*128]
        patches.append(np.clip(p, -EEG_CLIP, EEG_CLIP) / EEG_STD)
    return patches
```

Omitting either step causes near-zero encoder inputs and meaningless SAE scores.

# Deployment Guide: From Architecture to App

This guide describes how to deploy a new EEG encoder and dataset into the
SAE4EEG framework. By following these steps, you will go from a raw model
checkpoint to a fully interactive Streamlit app.

---

## 1. Implement the Encoder Backend

The framework is model-agnostic. You must wrap your model in an
`EncoderBackend` subclass in `src/sae4eeg/encoders.py`.

### Minimal Backend Template

```python
class MyNewModelBackend(EncoderBackend):
    sample_rate_in = 200  # The sample rate your model expects

    def __init__(self, weights_path: str, **kwargs):
        self.embed_dim = 512
        self.model = MyModelArchitecture()
        # ... load weights ...
        self.model.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, T) -> (B, S, E)"""
        with torch.no_grad():
            return self.model(x)

    def get_hookable_layers(self) -> List[nn.Module]:
        """Return list of transformer blocks for SAE hooking"""
        return list(self.model.transformer.blocks)

    def to(self, device):
        self.model.to(device)
        return self
```

---

## 2. Prepare the Dataset

The framework expects a directory of HDF5 files (`.h5` or `.hdf5`).

- **Shape:** `(N, C, T)` — N windows, C channels, T time samples.
- **Channels:** The standard clinical 10-20 montage (19 channels) is preferred.
- **Metadata:** Clinical labels (age, sex, abnormality, etc.) should be stored as HDF5 attributes or datasets within the file.
- **Registration:** Map your dataset path in `tools/build_all_caches.py` and other relevant tools under `_ENCODER_DATA`.

---

## 3. Register the Model

Update these locations so the tools and app recognize your model:

1.  **`src/sae4eeg/encoders.py`**:
    *   Add an entry to `MODEL_CARDS` (metadata for the app).
    *   Update `load_encoder()` factory to include your new name.
2.  **`tools/build_all_caches.py`**:
    *   Add your model to `_ENCODER_DATA` (path to HDF5s).
    *   Add to `_ENCODER_FS` (sample rate).
    *   Add to `_ENCODER_EMBED` (embedding dim).
3.  **`app/main.py`**:
    *   Update `_parse_encoder()` if your folder naming convention is new.

---

## 3. The Pipeline (Model → App)

Run these steps in order. Each step's output is required for the next.

| Step | Command | App Feature Enabled |
| :--- | :--- | :--- |
| **1. Train XAE** | `uv run tools/train_xae.py --encoder mymodel` | Physiological decoding (Spectra) |
| **2. Train SAEs** | `uv run tools/train_sae_layers.py --encoder mymodel --layers 0,1,2` | Feature Explorer (Activations) |
| **3. Build All** | `uv run tools/build_all_caches.py --encoder mymodel --all-layers` | Codebook, Landscape, Metadata |
| **4. Run TCAV** | `uv run tools/run_tcav.py --experiment mymodel_layer2` | Concept Attribution (TCAV) |
| **5. Layer UMAP** | `uv run tools/build_layer_umap_cache.py --encoder mymodel` | Layer Explorer (Animation) |
| **6. Prototypes** | `uv run tools/feature_maximization.py --experiment mymodel_layer2` | Prototype Waveforms |

---

## 4. App Discovery & Requirements

The app (`app/main.py`) "discovers" experiments by scanning the filesystem.
If a feature is missing in the app, check if the corresponding file exists:

| App Tab / Feature | Required File | Path Pattern |
| :--- | :--- | :--- |
| **Feature Explorer** | SAE Checkpoint | `results/features/{folder}/sae_*.pt` |
| **Spectral Profile** | XAE Checkpoint | `results/xae/{folder}/xae_checkpoint.pt` |
| **Codebook Browser** | App Cache | `results/experiments/{exp}/app_cache.pt` |
| **TCAV Scores** | TCAV Cache | `results/tcav/{exp}/tcav_cache.pt` |
| **Layer Explorer** | UMAP Cache | `results/layer_umap/{encoder}/umap_cache.pt` |
| **Prototype Waves** | Prototypes | `results/feat_viz/{exp}_{run}/prototypes.npz` |

### The `metadata.json` Glue

Every experiment in `results/experiments/{exp}/` **must** have a `metadata.json`.
**Good news:** `tools/build_all_caches.py` generates this for you automatically
from the SAE checkpoint.

---

## 5. Deployment Checklist

- [ ] `EncoderBackend` implemented and registered in `encoders.py`.
- [ ] Dataset exists in `data/` and is mapped in `_ENCODER_DATA`.
- [ ] `train_xae.py` completed (R² > 0.7 recommended).
- [ ] `train_sae_layers.py` completed (check sparsity/L0 in logs).
- [ ] `build_all_caches.py` finished without errors.
- [ ] (Optional) `run_tcav.py` for clinical concept mapping.
- [ ] Launch: `uv run streamlit run app/main.py`.

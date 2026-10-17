"""Rebuttal — preprocess TUAB v3.0.1 raw EDFs → HDF5 for the mecheeg pipeline.

Produces data compatible with src/mecheeg/dataset.py:H5PYDatasetLabeled and the
finetune tools, for the public-corpus age-pathology entanglement replication
(AC point 4 / rkJV Q3). Unlike the old sleepfm-v2 preprocess_tuab.py, this keeps
age/sex from the EDF headers so age can be a TCAV/steering concept.

Design decisions (grounded by probing real TUAB files + the D4 schema):
  * 19 standard channels in H5PYDatasetLabeled order; TUAB TCP-AR names
    'EEG FP1-REF' → map, with old-nomenclature T3/T4 → T7/T8 (T5/T6 kept).
  * resample to target Hz (128 SleepFM / 200 REVE·LaBraM); TUAB sfreq varies (250/256/400).
  * 60 Hz notch (US data) + 0.5–70 Hz bandpass; 60 s non-overlapping windows.
  * store data in Volts (mne native) — StandardizeLabel clips ±5e-5 and /1e-5 at load.
  * labels 0/1 from normal/abnormal; subjects = TUAB subject token → int (leakage-free split).
  * metadata group: age (int), gender ('male'/'female'), age_group (decade bins),
    classification ('Normal'/'Abnormal'), split ('train'/'eval').

Usage:
  uv run tools/rebuttal/preprocess_tuab_rebuttal.py --fs 128 --out data/tuab-sleepfm-128 \
      --max-windows-per-rec 12 --workers 8
  uv run tools/rebuttal/preprocess_tuab_rebuttal.py --smoke   # 4 files, prints shapes
"""
from __future__ import annotations

import argparse
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
TUAB_ROOT = ROOT / "data" / "tuab_v3.0.1" / "edf"

# H5PYDatasetLabeled channel order (the pipeline default).
PAPER_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                  "C4", "T8", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
# paper name -> TUAB 10-20 name (old nomenclature for temporals)
_TO_TUAB = {"T7": "T3", "T8": "T4"}


def _tuab_key(paper_name: str) -> str:
    return f"EEG {_TO_TUAB.get(paper_name, paper_name).upper()}-REF"


def age_to_group(age: int) -> str:
    if age < 0:
        return "unknown"
    for lo in (0, 4, 10, 20, 30, 40, 50, 60, 70, 80):
        hi = {0: 3, 4: 9, 10: 19, 20: 29, 30: 39, 40: 49, 50: 59, 60: 69, 70: 79, 80: 200}[lo]
        if lo <= age <= hi:
            return f"{lo}-{hi}" if lo != 80 else "80+"
    return "unknown"


def parse_demographics(edf_path: Path):
    """age (int, -1 if unknown), gender ('male'/'female'/'unknown') from EDF header."""
    with open(edf_path, "rb") as fh:
        patient = fh.read(88)[8:88].decode("latin-1")
    m_age = re.search(r"Age:(\d+)", patient)
    m_sex = re.search(r"\b([MF])\b", patient)
    age = int(m_age.group(1)) if m_age else -1
    gender = {"M": "male", "F": "female"}.get(m_sex.group(1)) if m_sex else "unknown"
    return age, gender


def subject_token(edf_path: Path) -> str:
    # TUAB filename: <subj>_s<sess>_t<tok>.edf ; subject = first token
    return edf_path.name.split("_")[0]


def process_one(edf_path_str: str, fs: int, window_s: int, max_windows: int):
    import mne
    from scipy.signal import resample_poly
    edf_path = Path(edf_path_str)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    except Exception as e:  # noqa: BLE001
        return {"error": f"{edf_path.name}: {e}"}

    # pick + order channels
    name_map = {ch.upper(): ch for ch in raw.ch_names}
    picks, missing = [], []
    for pname in PAPER_CHANNELS:
        key = _tuab_key(pname).upper()
        if key in name_map:
            picks.append(name_map[key])
        else:
            missing.append(pname)
    if missing:
        return {"error": f"{edf_path.name}: missing channels {missing}"}
    raw.pick(picks)
    raw.reorder_channels(picks)

    # filter (60 Hz notch for US data) then resample
    sf = raw.info["sfreq"]
    try:
        raw.notch_filter(60.0, verbose="error")
        raw.filter(0.5, 70.0, verbose="error")
    except Exception:  # noqa: BLE001
        pass
    x = raw.get_data()  # (19, T) in Volts
    if sf != fs:
        # rational resample sf -> fs
        from math import gcd
        g = gcd(int(round(sf)), fs)
        x = resample_poly(x, up=fs // g, down=int(round(sf)) // g, axis=-1).astype(np.float32)
    x = x.astype(np.float32)

    win = window_s * fs
    n_full = x.shape[1] // win
    if n_full == 0:
        return {"error": f"{edf_path.name}: too short ({x.shape[1]} < {win})"}
    n_full = min(n_full, max_windows) if max_windows else n_full
    windows = np.stack([x[:, i * win:(i + 1) * win] for i in range(n_full)])  # (n,19,win)

    label = 1 if "abnormal" in edf_path_str else 0
    split = "eval" if f"{Path('edf')/'eval'}" in edf_path_str or "/eval/" in edf_path_str else "train"
    age, gender = parse_demographics(edf_path)
    return {
        "data": windows, "label": label, "subject": subject_token(edf_path),
        "age": age, "gender": gender, "age_group": age_to_group(age),
        "classification": "Abnormal" if label else "Normal", "split": split,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--window-s", type=int, default=60)
    ap.add_argument("--out", default="data/tuab-sleepfm-128")
    ap.add_argument("--max-windows-per-rec", type=int, default=12)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="process only first N files (debug)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    edfs = sorted(str(p) for p in TUAB_ROOT.rglob("*.edf"))
    if args.smoke:
        edfs = edfs[:4]; args.workers = 1
    if args.limit:
        edfs = edfs[:args.limit]
    print(f"[tuab] {len(edfs)} EDFs → fs={args.fs}Hz, {args.window_s}s windows, "
          f"≤{args.max_windows_per_rec}/rec, {args.workers} workers", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_ids: dict[str, int] = {}
    buf_data, buf_lab, buf_subj, buf_file = [], [], [], []
    buf_meta = {k: [] for k in ("age", "gender", "age_group", "classification", "split")}
    shard_i = 0
    n_ok = n_err = n_win = 0
    file_counter = 0

    def flush():
        nonlocal shard_i
        if not buf_data:
            return
        data = np.concatenate(buf_data).astype(np.float32)
        path = out_dir / f"data_{shard_i}.hdf5"
        with h5py.File(path, "w") as h:
            h.create_dataset("data", data=data, compression="gzip", compression_opts=1)
            h.create_dataset("labels", data=np.array(buf_lab, dtype=np.int32))
            h.create_dataset("subjects", data=np.array(buf_subj, dtype=np.int32))
            h.create_dataset("file_idxs", data=np.array(buf_file, dtype=np.int32))
            g = h.create_group("metadata")
            g.create_dataset("age", data=np.array(buf_meta["age"], dtype=np.int32))
            for key in ("gender", "age_group", "classification", "split"):
                g.create_dataset(key, data=np.array(buf_meta[key], dtype=object),
                                 dtype=h5py.string_dtype())
        print(f"  wrote {path.name}: {data.shape}", flush=True)
        shard_i += 1
        buf_data.clear(); buf_lab.clear(); buf_subj.clear(); buf_file.clear()
        for v in buf_meta.values():
            v.clear()

    def handle(r):
        nonlocal n_ok, n_err, n_win, file_counter
        if r is None or "error" in r:
            n_err += 1
            if r and n_err <= 10:
                print(f"  [skip] {r['error']}", flush=True)
            return
        n = r["data"].shape[0]
        sid = subj_ids.setdefault(r["subject"], len(subj_ids))
        buf_data.append(r["data"])
        buf_lab.extend([r["label"]] * n)
        buf_subj.extend([sid] * n)
        buf_file.extend([file_counter] * n)
        buf_meta["age"].extend([r["age"]] * n)
        for key in ("gender", "age_group", "classification", "split"):
            buf_meta[key].extend([r[key]] * n)
        n_ok += 1; n_win += n; file_counter += 1
        if sum(a.shape[0] for a in buf_data) >= args.shard_size:
            flush()

    if args.workers == 1:
        for e in edfs:
            handle(process_one(e, args.fs, args.window_s, args.max_windows_per_rec))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, e, args.fs, args.window_s, args.max_windows_per_rec): e
                    for e in edfs}
            for fut in as_completed(futs):
                handle(fut.result())
    flush()
    print(f"[tuab] done: {n_ok} recordings, {n_win} windows, {len(subj_ids)} subjects, "
          f"{n_err} skipped → {out_dir}", flush=True)


if __name__ == "__main__":
    main()

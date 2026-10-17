"""Rebuttal — preprocess TUEV (TUH EEG Events) → event-labeled HDF5.

For rkJV Q4 / the morphology concern: build a dataset of specific epileptiform events
(spike-and-sharp-wave, GPED, PLED) vs background so we can test whether the framework
separates *specific* EEG phenomena — the granular signal the binary BrainCapture label
collapses (paper's epileptiform null).

Reuses the TUAB channel mapping (EEG XXX-REF, T3/T4→T7/T8). Per-channel .rec events are
deduped by (rounded onset, label) → one 10 s window per event. Labels 0–5:
  0=spsw 1=gped 2=pled 3=eyem 4=artf 5=bckg   (epileptiform = 0,1,2).

  uv run tools/rebuttal/preprocess_tuev_rebuttal.py --fs 128 --out data/tuev-sleepfm-128
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
TUEV = Path(os.environ.get("TUEV_ROOT", ROOT / "data" / "tuev"))  # TUH EEG Events corpus root

PAPER_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                  "C4", "T8", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
_TO_TUEV = {"T7": "T3", "T8": "T4"}
LABELS = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
WIN_S = 10


def _key(pname):
    return f"EEG {_TO_TUEV.get(pname, pname).upper()}-REF"


def parse_events(rec_path):
    """Deduped (onset_sec, label_int_0based) events."""
    seen, events = set(), []
    for line in open(rec_path):
        p = line.strip().split(",")
        if len(p) < 4:
            continue
        try:
            st = float(p[1]); lab = int(p[3])
        except ValueError:
            continue
        key = (round(st), lab)
        if key in seen:
            continue
        seen.add(key); events.append((st, lab - 1))
    return events


def process_one(stem, fs):
    import mne
    from scipy.signal import resample_poly
    from math import gcd
    edf = TUEV / "edf" / f"{stem}.edf"
    rec = TUEV / "recs" / f"{stem}.rec"
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    except Exception as e:  # noqa: BLE001
        return {"error": f"{stem}: {e}"}
    name_map = {ch.upper(): ch for ch in raw.ch_names}
    picks = []
    for pn in PAPER_CHANNELS:
        k = _key(pn).upper()
        if k not in name_map:
            return {"error": f"{stem}: missing {pn}"}
        picks.append(name_map[k])
    raw.pick(picks); raw.reorder_channels(picks)
    try:
        raw.notch_filter(60.0, verbose="error"); raw.filter(0.5, 70.0, verbose="error")
    except Exception:  # noqa: BLE001
        pass
    sf = raw.info["sfreq"]
    x = raw.get_data()  # (19, T) V
    if sf != fs:
        g = gcd(int(round(sf)), fs)
        x = resample_poly(x, up=fs // g, down=int(round(sf)) // g, axis=-1)
    x = x.astype(np.float32)

    win = WIN_S * fs
    wins, labs = [], []
    for onset, lab in parse_events(rec):
        a = int(round(onset * fs))
        if a + win > x.shape[1] or a < 0:
            continue
        wins.append(x[:, a:a + win]); labs.append(lab)
    if not wins:
        return {"error": f"{stem}: no valid windows"}
    return {"data": np.stack(wins), "labels": np.array(labs, dtype=np.int32),
            "subject": stem[:8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--out", default="data/tuev-sleepfm-128")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    edfs = {os.path.basename(f)[:-4] for f in glob.glob(str(TUEV / "edf" / "*.edf"))}
    recs = {os.path.basename(f)[:-4] for f in glob.glob(str(TUEV / "recs" / "*.rec"))}
    stems = sorted(edfs & recs)
    if args.smoke:
        stems = stems[:4]; args.workers = 1
    print(f"[tuev] {len(stems)} paired recordings → fs={args.fs}, {WIN_S}s windows", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_ids = {}
    buf_d, buf_l, buf_s, buf_f = [], [], [], []
    shard = fc = n_ok = n_err = n_win = 0

    def flush():
        nonlocal shard
        if not buf_d:
            return
        data = np.concatenate(buf_d).astype(np.float32)
        with h5py.File(out_dir / f"data_{shard}.hdf5", "w") as h:
            h.create_dataset("data", data=data)
            h.create_dataset("labels", data=np.array(buf_l, dtype=np.int32))
            h.create_dataset("subjects", data=np.array(buf_s, dtype=np.int32))
            h.create_dataset("file_idxs", data=np.array(buf_f, dtype=np.int32))
            g = h.create_group("metadata")
            g.create_dataset("event_class",
                             data=np.array([LABELS[x + 1] for x in buf_l], dtype=object),
                             dtype=h5py.string_dtype())
        print(f"  wrote data_{shard}.hdf5 {data.shape}", flush=True)
        shard += 1
        buf_d.clear(); buf_l.clear(); buf_s.clear(); buf_f.clear()

    def handle(r):
        nonlocal fc, n_ok, n_err, n_win
        if not r or "error" in r:
            n_err += 1
            if r and n_err <= 8:
                print(f"  [skip] {r['error']}", flush=True)
            return
        n = len(r["data"]); sid = subj_ids.setdefault(r["subject"], len(subj_ids))
        buf_d.append(r["data"]); buf_l.extend(r["labels"].tolist())
        buf_s.extend([sid] * n); buf_f.extend([fc] * n)
        fc += 1; n_ok += 1; n_win += n
        if sum(a.shape[0] for a in buf_d) >= args.shard_size:
            flush()

    if args.workers == 1:
        for s in stems:
            handle(process_one(s, args.fs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, s, args.fs): s for s in stems}
            for f in as_completed(futs):
                handle(f.result())
    flush()
    import collections
    print(f"[tuev] done: {n_ok} recs, {n_win} windows, {len(subj_ids)} subjects, {n_err} skipped")


if __name__ == "__main__":
    main()

"""Aggregate the seeded camera-ready reruns (results/camera_ready/seed{S}/) into mean ± sd.

Writes results/camera_ready/summary.json and summary.md. Every camera-ready number for the
rebuttal experiments should be quoted from these files, with the seed count.

  uv run tools/rebuttal/aggregate_camera_ready.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CR = ROOT / "results" / "camera_ready"
CONCEPTS = ["Pathology", "Age", "Sex", "Med-Psych", "Med-ASM"]


def seeds(suffix=""):
    return sorted(p for p in CR.glob(f"seed[0-9]{suffix}") if p.is_dir())


def load(dirs, name):
    return [json.loads((d / name).read_text()) for d in dirs if (d / name).exists()]


def ms(xs):
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
            "n": int(len(a)), "values": [round(float(x), 4) for x in a]}


def fmt(m, nd=3):
    return f"{m['mean']:.{nd}f} ± {m['sd']:.{nd}f}"


def main():
    S = seeds()
    out, md = {"seeds": [p.name for p in S]}, [f"# Camera-ready reruns — {len(S)} SAE seeds ({', '.join(p.name for p in S)})\n"]

    # Collateral matrix (SAE, clamp 25%) + LEACE + amnesic: AUROC after steering row→col
    for name, key in [("collateral_matrix.json", "SAE (clamp 25%)"), ("leace_matrix.json", "LEACE"),
                      ("amnesic_matrix.json", "amnesic (INLP)")]:
        runs = load(S, name)
        base = {c: ms([r["baseline"][c] for r in runs]) for c in CONCEPTS}
        mat = {s: {m: ms([r["matrix"][s][m] for r in runs]) for m in CONCEPTS} for s in CONCEPTS}
        out[key] = {"baseline": base, "matrix": mat}
        md += [f"\n## {key}: AUROC of measured concept (cols) after steering/erasing row concept\n",
               "| steered \\ measured | " + " | ".join(CONCEPTS) + " |", "|---" * (len(CONCEPTS) + 1) + "|",
               "| baseline | " + " | ".join(fmt(base[c]) for c in CONCEPTS) + " |"]
        md += ["| " + s + " | " + " | ".join(fmt(mat[s][m]) for m in CONCEPTS) + " |" for s in CONCEPTS]

    # Matched-suppression comparison
    runs = load(S, "selectivity_controlled.json")
    meths = list(runs[0]["methods"])
    sc = {m: {"target_auc": ms([r["methods"][m]["target_auc"] for r in runs]),
              "off_auc": ms([r["methods"][m]["off_auc"] for r in runs]),
              "reached": [bool(r["methods"][m]["reached"]) for r in runs]} for m in meths}
    out["matched_suppression"] = {"target_level": runs[0]["target_level"], "methods": sc}
    md += [f"\n## Matched suppression (Pathology → AUROC ≈ {runs[0]['target_level']}); off-target = Age AUROC\n",
           "| method | target AUROC | Age AUROC (preserved) | reached |", "|---|---|---|---|"]
    md += [f"| {m} | {fmt(v['target_auc'])} | {fmt(v['off_auc'])} | {sum(v['reached'])}/{len(v['reached'])} |"
           for m, v in sc.items()]

    # Deployed head vs probe at f = 0.25
    runs = load(S, "native_vs_probe.json")
    out["native_vs_probe"] = {}
    md += ["\n## Deployed classifier head vs probe (abnormality AUROC)\n",
           "| encoder | f | head | head (random clamp) | probe |", "|---|---|---|---|---|"]
    for ci, cell in enumerate(runs[0]["cells"]):
        fr = np.asarray(cell["fracs"])
        res = {}
        for f in (0.0, 0.25, 0.5):
            i = int(np.argmin(abs(fr - f)))
            row = {"probe": ms([r["cells"][ci]["probe"][i] for r in runs])}
            if cell.get("has_native"):
                row["head"] = ms([r["cells"][ci]["native"][i] for r in runs])
                row["head_random"] = ms([r["cells"][ci]["native_rand"][i] for r in runs])
            res[f"f={fr[i]:.4g}"] = row
            md.append(f"| {cell['name']} | {fr[i]:.3g} | {fmt(row['head']) if 'head' in row else '—'} | "
                      f"{fmt(row['head_random']) if 'head_random' in row else '—'} | {fmt(row['probe'])} |")
        out["native_vs_probe"][cell["name"]] = res

    # TUEV / TUSL separability (each seed itself = mean over 20 subject splits)
    for name, key in [("tuev_separability.json", "TUEV separability"), ("slowing_separability.json", "TUSL separability")]:
        runs = load(S, name)
        res = {ev: {"embedding": ms([r["results"][ev]["auroc_embedding"] for r in runs]),
                    "sae": ms([r["results"][ev]["auroc_sae"] for r in runs]),
                    "n_pos": runs[0]["results"][ev]["n_pos"]} for ev in runs[0]["results"]}
        out[key] = res
        md += [f"\n## {key} (AUROC vs background; each seed = mean of 20 subject splits)\n",
               "| event | n_pos | embedding | SAE |", "|---|---|---|---|"]
        md += [f"| {ev} | {v['n_pos']} | {fmt(v['embedding'])} | {fmt(v['sae'])} |" for ev, v in res.items()]

    # TUEV steering selectivity (self drop − mean off-target drop), TCAV vs random clamp
    for suffix, label in [("", "clamp 50%"), ("_clamp25", "clamp 25%")]:
        runs = load(seeds(suffix), "tuev_steering.json")
        if not runs:
            continue
        evs = list(runs[0]["selectivity"])
        res = {e: {k: ms([r["selectivity"][e][k] for r in runs]) for k in ("tcav", "rand_mean", "excess")}
               for e in evs}
        out[f"TUEV steering ({label})"] = res
        md += [f"\n## TUEV steering selectivity, {label} (self drop − mean off-target drop)\n",
               "| event | TCAV-ranked | random clamp | excess |", "|---|---|---|---|"]
        md += [f"| {e} | {fmt(v['tcav'])} | {fmt(v['rand_mean'])} | {fmt(v['excess'])} |" for e, v in res.items()]

    (CR / "summary.json").write_text(json.dumps(out, indent=2))
    (CR / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

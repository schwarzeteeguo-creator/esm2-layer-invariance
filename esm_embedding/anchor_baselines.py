"""Published-baseline anchoring for our supervised results (Reviewer 1, M6).

Generates a markdown table placing our numbers next to published
ProteinGym values so readers can judge pipeline competence. Published
values are hardcoded with sources; our values are read from the revision
result files when present.

Usage:
  python -m esm_embedding.anchor_baselines \
      --out revision_results/anchor_table.md
"""

import json, argparse
from pathlib import Path
import numpy as np

# Published zero-shot values, ProteinGym v1 substitution benchmark
# (mean Spearman rho over datasets; sources in parentheses).
PUBLISHED_ZERO_SHOT = [
    ("PoET-2 + VenusREM (ensemble)", 0.558, "arXiv:2508.04724 (2025 snapshot)"),
    ("GEMME", 0.457, "ProteinGym v1.0 (Notin et al. 2023)"),
    ("TranceptEVE L", 0.457, "ProteinGym v1.0"),
    ("SaProt 650M", 0.457, "cross-validated by ProtSSN/VenusREM papers"),
    ("ESM-2 650M", 0.414, "ProteinGym v1.0"),
    ("ESM-1v (ensemble)", 0.416, "ProteinGym v1.0"),
    ("ESM-2 3B", 0.410, "ProteinGym v1.0"),
    ("VESPA", 0.437, "ProteinGym v1.0"),
]

# Our own reproductions / new results (filled from files when available)
OUR_FILES = {
    "SaProt zero-shot (Phase-1 reproduction, 63 datasets)":
        "SaProt-main/ProteinGym_results.tsv",
    "ESM-2 lens final-layer (v2, fp32, gate-passed)":
        "revision_results/logit_lens_v2_esm2/logit_lens_v2_summary.json",
}


def read_tsv_mean(path):
    try:
        rhos = []
        with open(path) as f:
            header = f.readline().lower()
            idx = None
            for j, col in enumerate(header.strip().split("\t")):
                if "spearman" in col or "rho" in col:
                    idx = j
                    break
            if idx is None:
                return None
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > idx:
                    try:
                        rhos.append(float(parts[idx]))
                    except ValueError:
                        pass
        return float(np.mean(rhos)) if rhos else None, len(rhos)
    except FileNotFoundError:
        return None


def read_lens_summary(path):
    try:
        with open(path) as f:
            s = json.load(f)
        n = str(s["n_layers"] - 1)
        v = s["per_layer"].get(n, {}).get("mean")
        return (float(v), s["per_layer"][n]["n"]) if v is not None else None
    except (FileNotFoundError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revision_results/anchor_table.md")
    args = ap.parse_args()

    lines = ["# Anchor table: our results vs published ProteinGym values",
             "",
             "Zero-shot (no supervised readout), published:",
             "",
             "| Method | Spearman rho | Source |",
             "|---|---|---|"]
    for name, v, src in PUBLISHED_ZERO_SHOT:
        lines.append(f"| {name} | {v:.3f} | {src} |")

    lines += ["", "Ours:", "", "| Result | Spearman rho | N datasets |", "|---|---|---|"]
    r = read_tsv_mean(OUR_FILES["SaProt zero-shot (Phase-1 reproduction, 63 datasets)"])
    if r:
        lines.append(f"| SaProt zero-shot (our Phase-1 reproduction) | {r[0]:.3f} | {r[1]} |")
    r = read_lens_summary(
        OUR_FILES["ESM-2 lens final-layer (v2, fp32, gate-passed)"])
    if r:
        lines.append(f"| ESM-2 masked zero-shot = lens L32 (v2, fp32) | {r[0]:.3f} | {r[1]} |")

    # supervised probing anchors from v4 if present
    for f in sorted(Path("revision_results/probing_v4").glob(
            "v4__mutant__combined__*.json")) if Path(
                "revision_results/probing_v4").exists() else []:
        with open(f) as fh:
            d = json.load(fh)
        vals, names = [], []
        for name, e in d.items():
            layers = e["result"]
            v = [x["spearman"] for x in layers.values()
                 if np.isfinite(x.get("spearman", np.nan))]
            if v:
                vals.append(np.mean(v))
                names.append(name)
        if vals:
            split = f.stem.split("__")[-1]
            lines.append(f"| Supervised Ridge probing, MUTANT arm, {split} "
                         f"(layer-mean) | {np.mean(vals):.3f} | {len(vals)} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

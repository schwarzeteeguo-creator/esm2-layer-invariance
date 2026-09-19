#!/usr/bin/env python
"""Fix len-mismatch 3Di datasets by AA-level alignment + '#' gap padding.

make_3di.py rejects datasets whose AFDB structure is shorter than the
ProteinGym wild type (UBC9_HUMAN 158/159, Q59976_STRSQ 479/501) — the
AFDB model truncates terminal disordered regions. For each such dataset:

  1. read the residue sequence from the PDB CA atoms (file order);
  2. find the best-scoring offset of pdb_seq inside the ProteinGym WT
     (identity >= 0.9 required, else the dataset is skipped);
  3. write <dataset>.3di.txt = '#'*start + foldseek_3di
     + '#'*(L - start - len(3di));  '#' is SaProt's unknown-structure
     placeholder, so uncovered residues degrade to structure-masked
     instead of corrupting the alignment.

Also sweeps every EXISTING .3di.txt file at offset 0 and reports any
whose sequence identity vs WT is below 0.9 (would indicate a wrong
UniProt isoform — reported only, never rewritten).

Run inside ~/AIDD/projects/protein_revision/data with the protein env.
"""

import csv, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LMDB = ROOT / "substitutions"
PDB_DIR = ROOT / "afdb_pdbs"
OUT_DIR = ROOT / "3di"
TMP = ROOT / "3di_tmp"
FOLDSEEK = Path.home() / "foldseek/foldseek"
SAPROT_3DI_ALPHABET = set("pynwrqhgdlavtmfsekyic")
ACC_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])"
    r"(-\d+)?$")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def wt_of(lmdb_dir):
    import lmdb
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    with env.begin() as txn:
        wt = txn.get(b"wild_type")
    env.close()
    return wt.decode()


def pdb_ca_sequence(pdb_path):
    seq = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" \
                and line[16] in (" ", "A"):
            aa = THREE_TO_ONE.get(line[17:20].strip())
            if aa:
                seq.append(aa)
    return "".join(seq)


def foldseek_3di(pdb_path):
    tag = TMP / ("fix_" + pdb_path.stem)
    prefix = str(tag)
    for f in TMP.glob(tag.name + "*"):
        f.unlink()
    r = subprocess.run([str(FOLDSEEK), "createdb", str(pdb_path), prefix],
                       capture_output=True, text=True, timeout=300)
    ss = Path(prefix + "_ss")
    if r.returncode != 0 or not ss.exists():
        return None
    raw = ss.read_bytes().decode("ascii", errors="ignore").lower()
    return "".join(c for c in raw if c in SAPROT_3DI_ALPHABET)


def best_offset(wt, pdb_seq):
    """Slide pdb_seq over wt; return (start, identity) of the best match."""
    n, L = len(pdb_seq), len(wt)
    best = (0, -1.0)
    for start in range(0, L - n + 1):
        ident = sum(pdb_seq[i] == wt[start + i] for i in range(n)) / n
        if ident > best[1]:
            best = (start, ident)
    return best


def dataset_to_acc(name, seed, cache):
    up = None
    with open(ROOT / "DMS_substitutions.csv") as f:
        for row in csv.DictReader(f):
            if row["DMS_filename"].replace(".csv", "") == name:
                up = row["UniProt_ID"]
                break
    up = up or "_".join(name.split("_")[:2])
    if up in seed:
        return seed[up]
    if up in cache and cache[up]:
        return cache[up]
    cand = up.split("_")[0]
    return cand if ACC_RE.match(cand) else None


def main():
    seed = json.loads((ROOT / "acc_seed.json").read_text())
    cache = json.loads((ROOT / "acc_cache.json").read_text())
    OUT_DIR.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)

    datasets = sorted(p.name for p in LMDB.iterdir() if p.is_dir())

    # ---- pass 1: fix len-mismatch datasets (no .3di.txt yet) ----
    print("== gap-fix pass ==")
    fixed = 0
    for name in datasets:
        if (OUT_DIR / f"{name}.3di.txt").exists():
            continue
        acc = dataset_to_acc(name, seed, cache)
        pdb = PDB_DIR / f"{acc}.pdb" if acc else None
        if not acc or not pdb.exists():
            print(f"  SKIP {name}: no PDB (acc={acc})")
            continue
        wt = wt_of(LMDB / name)
        pdb_seq = pdb_ca_sequence(pdb)
        s3di = foldseek_3di(pdb)
        if not s3di or len(s3di) != len(pdb_seq):
            print(f"  SKIP {name}: foldseek 3di {len(s3di or '')} vs "
                  f"CA residues {len(pdb_seq)}")
            continue
        if len(pdb_seq) >= len(wt):
            print(f"  SKIP {name}: pdb {len(pdb_seq)} >= wt {len(wt)} "
                  "(unexpected — hand-check)")
            continue
        start, ident = best_offset(wt, pdb_seq)
        if ident < 0.9:
            print(f"  SKIP {name}: best identity {ident:.3f} < 0.90 "
                  f"at start={start} — alignment unreliable")
            continue
        aligned = "#" * start + s3di + "#" * (len(wt) - start - len(s3di))
        assert len(aligned) == len(wt)
        (OUT_DIR / f"{name}.3di.txt").write_text(aligned + "\n")
        print(f"  FIXED {name}: wt={len(wt)} pdb={len(pdb_seq)} "
              f"start={start} identity={ident:.3f} "
              f"pad={len(wt) - len(pdb_seq)}x'#'")
        fixed += 1

    # ---- pass 2: sanity sweep of every existing file at offset 0 ----
    print("\n== offset-0 sanity sweep (report only) ==")
    files = sorted(OUT_DIR.glob("*.3di.txt"))
    n_ok, warnings = 0, []
    for f in files:
        name = f.name[:-len(".3di.txt")]
        wt = wt_of(LMDB / name)
        toks = f.read_text().strip()
        if len(toks) != len(wt):
            warnings.append(f"{name}: len {len(toks)} != wt {len(wt)}")
            continue
        acc = dataset_to_acc(name, seed, cache)
        pdb = PDB_DIR / f"{acc}.pdb" if acc else None
        if not pdb or not pdb.exists():
            continue
        pdb_seq = pdb_ca_sequence(pdb)
        if len(pdb_seq) == len(wt):
            ident = sum(pdb_seq[i] == wt[i] for i in range(len(wt))) / len(wt)
            if ident < 0.9:
                warnings.append(f"{name}: offset-0 identity {ident:.3f} "
                                "< 0.90 — check isoform")
            else:
                n_ok += 1
    print(f"  verified at offset 0 with identity>=0.9: {n_ok}")
    for w in warnings:
        print(f"  WARN {w}")

    total = len(list(OUT_DIR.glob("*.3di.txt")))
    print(f"\n3di files now: {total} (was {total - fixed}; +{fixed} fixed)")


if __name__ == "__main__":
    main()

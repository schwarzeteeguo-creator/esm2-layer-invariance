"""Fold __shardK result files (from concurrent sharded runs) into the
canonical per-(arm,featset,split) / per-model JSON files.

Sharded runs write disjoint dataset subsets to suffixed files
(v4__arm__featset__split__shard0.json, pooling_v3__model__shard1.json, ...).
This script merges every shard file into its base name, dataset-keyed, so
downstream aggregation sees one canonical file per experiment.

Usage:  python -m esm_embedding.merge_shards <results_root>
"""

import json
import re
import sys
from pathlib import Path

SHARD_RE = re.compile(r"^(.*)__shard\d+$")


def main(results_root):
    root = Path(results_root)
    merged = 0
    for f in sorted(root.rglob("*__shard*.json")):
        m = SHARD_RE.match(f.stem)
        if not m:
            continue
        base = f.with_name(m.group(1) + ".json")
        data = {}
        if base.exists():
            try:
                data = json.loads(base.read_text())
            except json.JSONDecodeError:
                data = {}
        shard_data = json.loads(f.read_text())
        overlap = sorted(set(data) & set(shard_data))
        if overlap:
            print(f"  WARNING {f.name}: {len(overlap)} datasets already in "
                  f"{base.name} ({overlap[:3]}...); shard values win")
        data.update(shard_data)
        base.write_text(json.dumps(data, indent=1))
        print(f"  merged {f.name} -> {base.name} (total {len(data)})")
        merged += 1
    print(f"done: {merged} shard files merged under {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

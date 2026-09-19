"""Cross-validation split schemes for ProteinGym supervised probing.

Implements the three split schemes defined by the ProteinGym benchmark
(Notin et al., 2024) for supervised mutation-effect evaluation, plus a
strict GroupKFold on residue positions:

    random    : variants assigned to folds at random (seeded). All rows
                belonging to the same variant (multi-point substitutions
                are exploded into per-substitution rows) stay in the same
                fold, preventing variant-level leakage.
    modulo    : every fifth backbone position is assigned to the same fold
                (fold = (pos - 1) % n_folds).  -- ProteinGym official
    contiguous: the sequence is divided into n_folds contiguous segments
                of equal length (fold = floor((pos-1) * n_folds / L)).
                -- ProteinGym official
    position_groupkfold : sklearn GroupKFold grouped by residue position
                (the bespoke scheme used in the original submission).

All schemes return a list of (train_idx, test_idx) pairs over the row
indices of the feature matrix. Rows carry a `variant_id` (unique per
DMS variant) and a `position` (1-indexed residue of the substitution;
for multi-point variants the first substituted position determines the
modulo/contiguous fold, following ProteinGym's supervised convention).
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

SPLIT_SCHEMES = ["random", "modulo", "contiguous", "position_groupkfold"]


def make_splits(
    positions: np.ndarray,
    variant_ids: np.ndarray,
    seq_len: int,
    scheme: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) pairs for the given scheme."""
    n = len(positions)
    n_splits = min(n_folds, n)

    if scheme == "random":
        # Group by variant so exploded rows of one variant never straddle folds
        unique_variants = np.unique(variant_ids)
        rng = np.random.RandomState(seed)
        fold_arr = rng.randint(0, n_splits, size=len(unique_variants))
        fold_of_variant = dict(zip(unique_variants.tolist(),
                                   fold_arr.tolist()))
        row_fold = np.array([fold_of_variant[v] for v in variant_ids])

    elif scheme == "modulo":
        # ProteinGym official: every fifth backbone position in the same fold
        row_fold = (positions.astype(int) - 1) % n_splits

    elif scheme == "contiguous":
        # ProteinGym official: sequence divided into n_folds equal segments
        seg = np.floor((positions.astype(int) - 1) * n_splits / max(seq_len, 1))
        row_fold = np.clip(seg, 0, n_splits - 1).astype(int)

    elif scheme == "position_groupkfold":
        # sklearn GroupKFold requires n_splits <= number of groups. A few
        # ProteinGym datasets mutate only 2-4 positions (e.g. F7YBW8 with
        # 4), so cap folds at the group count — one held-out position per
        # fold, identical semantics to the n_folds=5 case.
        n_groups = len(np.unique(positions))
        kf = GroupKFold(n_splits=min(n_splits, n_groups))
        return list(kf.split(np.zeros(n), groups=positions))

    else:
        raise ValueError(f"Unknown split scheme: {scheme}. Choose from {SPLIT_SCHEMES}")

    folds = []
    for f in range(n_splits):
        test_idx = np.where(row_fold == f)[0]
        train_idx = np.where(row_fold != f)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds

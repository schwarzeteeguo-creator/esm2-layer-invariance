"""Structural confidence feature extraction from Boltz-1/AF3 predictions.

Extracts per-residue structural features from Boltzmann/AlphaFold confidence
metrics (pLDDT, PAE, ipTM). These features capture local and global structural
context that may complement sequence-based ESM embeddings for mutation effect
prediction.

Feature vector (12-dim):
    0: plddt_local       — pLDDT at the mutation position
    1-5: plddt_window    — pLDDT at [i-2, i-1, i, i+1, i+2]
    6: pae_mean_self     — mean PAE of position i vs all others
    7: pae_max           — max PAE — worst-case uncertainty
    8: plddt_vs_global   — plddt_local - global_mean_plddt
    9: plddt_local_std   — std of pLDDT in 5-residue window
    10: global_mean_plddt — overall protein structural confidence
    11: global_iptm       — interface predicted TM-score
"""

from typing import List, Optional
import numpy as np

FEATURE_DIM = 12


# ── Main extraction function ──────────────────────────────────────────

def extract_structural_features(
    plddt: np.ndarray,
    pae: np.ndarray,
    positions: List[int],
    iptm: float = 0.0,
    window_size: int = 2,
) -> np.ndarray:
    """Build per-position structural feature vectors.

    Args:
        plddt: Per-residue pLDDT, shape (L,), range ~0-100
        plddt: Predicted aligned error matrix, shape (L, L), in Angstroms
        positions: 1-indexed mutation positions to extract features for
        iptm: Interface predicted TM-score (0 for monomers)
        window_size: Half-window for local pLDDT features

    Returns:
        np.ndarray of shape (len(positions), 12) — structural feature matrix
    """
    L = len(plddt)
    n_pos = len(positions)
    features = np.zeros((n_pos, FEATURE_DIM), dtype=np.float32)

    # Convert 1-indexed to 0-indexed, clamp to valid range
    idx = np.array([max(0, min(p - 1, L - 1)) for p in positions], dtype=int)

    # Global statistics (repeated per position)
    global_mean_plddt = float(np.mean(plddt))

    # Per-position features
    for i, pos_idx in enumerate(idx):
        # [0] pLDDT at the mutation position
        plddt_local = float(plddt[pos_idx])
        features[i, 0] = plddt_local

        # [1-5] pLDDT window: [i-2, i-1, i, i+1, i+2]
        for w_offset in range(-window_size, window_size + 1):
            w_idx = pos_idx + w_offset
            if 0 <= w_idx < L:
                features[i, 1 + w_offset + window_size] = float(plddt[w_idx])
            else:
                # Edge padding: use the edge value
                edge_idx = max(0, min(w_idx, L - 1))
                features[i, 1 + w_offset + window_size] = float(plddt[edge_idx])

        # [6] Mean PAE — how coupled is position i to all other positions?
        pae_row = pae[pos_idx, :]
        features[i, 6] = float(np.mean(pae_row))

        # [7] Max PAE — worst-case structural uncertainty
        features[i, 7] = float(np.max(pae_row))

        # [8] Relative pLDDT: how confident is this position vs global average?
        features[i, 8] = plddt_local - global_mean_plddt

        # [9] Local pLDDT variance in 5-residue window
        window_vals = []
        for w_offset in range(-window_size, window_size + 1):
            w_idx = pos_idx + w_offset
            if 0 <= w_idx < L:
                window_vals.append(float(plddt[w_idx]))
            else:
                edge_idx = max(0, min(w_idx, L - 1))
                window_vals.append(float(plddt[edge_idx]))
        features[i, 9] = float(np.std(window_vals))

        # [10] Global mean pLDDT (same for all positions)
        features[i, 10] = global_mean_plddt

        # [11] Global ipTM (same for all positions)
        features[i, 11] = float(iptm)

    return features


# ── PAE-specific utilities ─────────────────────────────────────────────

def compute_pae_statistics(
    pae: np.ndarray,
    positions: List[int],
    radius: Optional[int] = None,
) -> dict:
    """Compute per-position PAE statistics.

    Args:
        pae: (L, L) matrix
        positions: 1-indexed positions
        radius: If set, only consider residues within this sequence distance

    Returns:
        dict with keys: mean, max, min per position (list of floats)
    """
    L = pae.shape[0]
    stats = {"mean": [], "max": [], "min": []}
    for p in positions:
        pos_idx = max(0, min(p - 1, L - 1))
        pae_row = pae[pos_idx, :]
        if radius is not None:
            start = max(0, pos_idx - radius)
            end = min(L, pos_idx + radius + 1)
            pae_row = pae_row[start:end]
        stats["mean"].append(float(np.mean(pae_row)))
        stats["max"].append(float(np.max(pae_row)))
        stats["min"].append(float(np.min(pae_row)))
    return stats


def pae_to_distance_proxy(pae: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Convert PAE matrix to approximate distance matrix.

    PAE scales roughly with predicted distance error. This converts
    PAE to a distance proxy useful for contact analysis.

    Args:
        pae: (L, L) PAE matrix
        beta: Scaling factor (default 0.5 — empirical)

    Returns:
        (L, L) approximate distance matrix
    """
    return pae * beta


# ── RSA approximation from pLDDT ────────────────────────────────────────

def approximate_rsa_from_plddt(plddt: np.ndarray) -> np.ndarray:
    """Approximate relative solvent accessibility from pLDDT.

    Low pLDDT often corresponds to exposed/disordered regions.
    High pLDDT typically indicates buried/structured core.

    This is a rough proxy — actual RSA requires DSSP or similar.
    Use only when explicit RSA is unavailable.

    Args:
        plddt: (L,) pLDDT array

    Returns:
        (L,) approximate RSA (0-1 scale)
    """
    # Invert and normalize: low pLDDT → high predicted RSA
    # pLDDT typically ranges 40-100
    plddt_clipped = np.clip(plddt, 40, 100)
    rsa_proxy = (100 - plddt_clipped) / 60.0  # maps [40,100] → [1,0]
    return np.clip(rsa_proxy, 0, 1)


# ── Load from BoltzResult ──────────────────────────────────────────────

def features_from_boltz_result(
    boltz_result,  # BoltzResult or compatible
    positions: List[int],
    window_size: int = 2,
) -> np.ndarray:
    """Convenience wrapper: extract structural features directly from BoltzResult.

    Args:
        boltz_result: BoltzResult with .plddt, .pae, .iptm attributes
        positions: 1-indexed mutation positions
        window_size: Half-window for local features

    Returns:
        (len(positions), 12) feature matrix
    """
    return extract_structural_features(
        plddt=boltz_result.plddt,
        pae=boltz_result.pae,
        positions=positions,
        iptm=boltz_result.iptm,
        window_size=window_size,
    )

"""Boltz-1 local inference wrapper for protein structure prediction.

Provides a unified interface for running Boltz-1 (open-source AlphaFold 3
implementation) and extracting confidence metrics (pLDDT, PAE, ipTM, pTM).

Usage:
    runner = BoltzRunner(model_dir="boltz1_weights", device="cuda")
    result = runner.predict_structure("MADE...", "my_protein")
    print(result.plddt)  # (L,) numpy array

Cache:
    Results are cached to disk as .npz files to avoid re-running inference.
    Default cache dir: boltz_cache/
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

# ── Data classes ─────────────────────────────────────────────────────

class BoltzResult:
    """Parsed Boltz-1 prediction result.

    Attributes:
        sequence: Input amino acid sequence
        plddt: Per-residue confidence scores, shape (L,), range ~0-100
        pae: Predicted aligned error matrix, shape (L, L), in Angstroms
        iptm: Interface predicted TM-score (only for complexes)
        ptm: Predicted TM-score (overall structure quality)
        ranking_score: Boltz confidence ranking score
        coords: Optional CA atom 3D coordinates, shape (L, 3) if available
    """
    def __init__(self, sequence: str, plddt: np.ndarray, pae: np.ndarray,
                 iptm: float = 0.0, ptm: float = 0.0,
                 ranking_score: float = 0.0, coords: Optional[np.ndarray] = None):
        self.sequence = sequence
        self.plddt = np.asarray(plddt, dtype=np.float32)
        self.pae = np.asarray(pae, dtype=np.float32)
        self.iptm = float(iptm)
        self.ptm = float(ptm)
        self.ranking_score = float(ranking_score)
        self.coords = np.asarray(coords, dtype=np.float32) if coords is not None else None

    def __repr__(self):
        return (f"BoltzResult(L={len(self.sequence)}, plddt_mean={self.plddt.mean():.1f}, "
                f"ptm={self.ptm:.3f}, iptm={self.iptm:.3f})")


# ── Boltz-1 Runner ────────────────────────────────────────────────────

class BoltzRunner:
    """Wrapper for Boltz-1 local inference.

    Args:
        model_dir: Path to Boltz-1 model weights directory
        device: 'cuda' or 'cpu'
        cache_dir: Directory to cache results (None to disable)
    """
    def __init__(self, model_dir: str, device: str = "cuda",
                 cache_dir: str = "boltz_cache"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load_model(self):
        """Lazy-load the Boltz-1 model."""
        if self._model is not None:
            return self._model
        try:
            # Boltz-1 import — may fail if not installed
            import boltz
            self._model = boltz.Boltz1(model_dir=str(self.model_dir), device=self.device)
            print(f"Boltz-1 model loaded from {self.model_dir}")
        except ImportError:
            raise ImportError(
                "Boltz-1 is not installed. Install with:\n"
                "  pip install boltz\n"
                "Or follow: https://github.com/schwarzeteeguo-creator/boltz-1"
            )
        return self._model

    def predict_structure(self, sequence: str, protein_id: str) -> BoltzResult:
        """Predict structure for a single protein sequence.

        Args:
            sequence: Amino acid sequence (uppercase, single-letter)
            protein_id: Unique identifier for caching and logging

        Returns:
            BoltzResult with pLDDT, PAE, ipTM, pTM
        """
        # Check cache first
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{protein_id}.npz"
            if cache_path.exists():
                return self._load_from_cache(cache_path)

        model = self._load_model()

        # Run inference
        # Boltz-1 expects a list of sequences or a dict
        try:
            raw = model.predict(sequence)
        except Exception as e:
            raise RuntimeError(f"Boltz-1 prediction failed for {protein_id}: {e}")

        # Parse results — adapt to actual Boltz-1 output format
        result = self._parse_output(raw, sequence)

        # Save to cache
        if cache_path:
            self._save_to_cache(result, cache_path)

        return result

    def batch_predict(self, sequences: Dict[str, str]) -> Dict[str, BoltzResult]:
        """Predict structures for multiple proteins.

        Args:
            sequences: Dict mapping protein_id -> sequence

        Returns:
            Dict mapping protein_id -> BoltzResult
        """
        results = {}
        for prot_id, seq in sequences.items():
            print(f"  Predicting {prot_id} (L={len(seq)})...")
            try:
                results[prot_id] = self.predict_structure(seq, prot_id)
            except Exception as e:
                print(f"  FAILED {prot_id}: {e}")
                results[prot_id] = None
        return results

    def _parse_output(self, raw, sequence: str) -> BoltzResult:
        """Parse Boltz-1 raw output into BoltzResult.

        Adapts to different Boltz-1 output formats.
        """
        # Boltz-1 typically returns a dict or object with these fields
        if isinstance(raw, dict):
            plddt = np.asarray(raw.get("plddt", raw.get("confidence", [])), dtype=np.float32)
            pae = np.asarray(raw.get("pae", raw.get("predicted_aligned_error", [])), dtype=np.float32)
            iptm = float(raw.get("iptm", 0.0))
            ptm = float(raw.get("ptm", raw.get("tm_score", 0.0)))
            ranking = float(raw.get("ranking_score", raw.get("confidence_score", 0.0)))
            coords = np.asarray(raw.get("coords", raw.get("positions", None)))
        elif hasattr(raw, "plddt"):
            plddt = np.asarray(raw.plddt)
            pae = np.asarray(raw.pae) if hasattr(raw, "pae") else np.zeros((len(sequence), len(sequence)))
            iptm = float(getattr(raw, "iptm", 0.0))
            ptm = float(getattr(raw, "ptm", 0.0))
            ranking = float(getattr(raw, "ranking_score", getattr(raw, "confidence_score", 0.0)))
            coords = getattr(raw, "coords", None)
        else:
            raise ValueError(f"Unknown Boltz-1 output format: {type(raw)}")

        return BoltzResult(
            sequence=sequence,
            plddt=plddt,
            pae=pae,
            iptm=iptm,
            ptm=ptm,
            ranking_score=ranking,
            coords=coords,
        )

    def _save_to_cache(self, result: BoltzResult, path: Path):
        """Save result to .npz cache."""
        save_dict = {
            "sequence": result.sequence,
            "plddt": result.plddt,
            "pae": result.pae,
            "iptm": result.iptm,
            "ptm": result.ptm,
            "ranking_score": result.ranking_score,
        }
        if result.coords is not None:
            save_dict["coords"] = result.coords
        np.savez_compressed(path, **save_dict)

    def _load_from_cache(self, path: Path) -> BoltzResult:
        """Load result from .npz cache."""
        data = np.load(path, allow_pickle=True)
        return BoltzResult(
            sequence=str(data["sequence"]),
            plddt=data["plddt"],
            pae=data["pae"],
            iptm=float(data.get("iptm", 0.0)),
            ptm=float(data.get("ptm", 0.0)),
            ranking_score=float(data.get("ranking_score", 0.0)),
            coords=data.get("coords", None),
        )


# ── Utility functions ──────────────────────────────────────────────────

def check_boltz_available() -> bool:
    """Check if Boltz-1 is installed and importable."""
    try:
        import boltz
        return True
    except ImportError:
        return False


def get_boltz_version() -> str:
    """Get installed Boltz-1 version."""
    try:
        import boltz
        return getattr(boltz, "__version__", "unknown")
    except ImportError:
        return "not installed"

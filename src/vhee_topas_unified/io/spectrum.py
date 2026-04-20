"""Beam-spectrum file loading."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple


def load_spectrum(spectrum_csv: str | Path) -> Tuple[List[float], List[float]]:
    """Load and normalize a discrete energy spectrum CSV.

    The input CSV is expected to contain `energy_mev` and `weight` columns.
    """

    spectrum_path = Path(spectrum_csv)
    energies: List[float] = []
    weights: List[float] = []
    with spectrum_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energies.append(float(row["energy_mev"]))
            weights.append(float(row["weight"]))

    if not energies:
        raise ValueError(f"No spectrum rows found in {spectrum_path}")

    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError(f"Spectrum weights in {spectrum_path} sum to a non-positive value.")

    normalized_weights = [value / total for value in weights]
    return energies, normalized_weights

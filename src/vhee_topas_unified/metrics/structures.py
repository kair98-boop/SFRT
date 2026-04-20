"""Structure-level planning metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .dvh import dose_at_volume_percent


def compute_structure_metrics(
    dose_grid: np.ndarray,
    mask: np.ndarray,
    *,
    prescription_gy: float | None,
    voxel_volume_cc: float,
    volume_thresholds_gy: Sequence[float] = (),
) -> Dict[str, float]:
    """Compute common target/OAR summary metrics for one structure."""

    dose = np.asarray(dose_grid, dtype=np.float64)
    structure_mask = np.asarray(mask, dtype=bool)
    if dose.shape != structure_mask.shape:
        raise ValueError("dose_grid and mask must have the same shape.")

    values = np.asarray(dose[structure_mask], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute metrics on an empty structure.")

    metrics: Dict[str, float] = {
        "volume_cc": float(values.size * float(voxel_volume_cc)),
        "mean_gy": float(np.mean(values)),
        "d2_gy": dose_at_volume_percent(values, 2.0),
        "d98_gy": dose_at_volume_percent(values, 98.0),
        "d95_gy": dose_at_volume_percent(values, 95.0),
        "d50_gy": dose_at_volume_percent(values, 50.0),
        "dmax_gy": float(np.max(values)),
    }
    if prescription_gy is not None:
        rx = float(prescription_gy)
        metrics["v95_pct"] = float(np.mean(values >= 0.95 * rx) * 100.0)
        metrics["v100_pct"] = float(np.mean(values >= 1.00 * rx) * 100.0)
        metrics["coverage_ratio"] = float(np.mean(values >= 1.00 * rx))
    for threshold in volume_thresholds_gy:
        label = f"v{int(round(float(threshold)))}_pct"
        metrics[label] = float(np.mean(values >= float(threshold)) * 100.0)
    return metrics


def metrics_table_rows(
    structure_metrics: Dict[str, Dict[str, float]],
    domain_label: str,
) -> List[Dict[str, object]]:
    """Convert nested metric dictionaries into flat table rows."""

    rows: List[Dict[str, object]] = []
    for name, metrics in structure_metrics.items():
        row: Dict[str, object] = {"domain": domain_label, "structure": name}
        row.update({key: float(value) for key, value in metrics.items()})
        rows.append(row)
    return rows

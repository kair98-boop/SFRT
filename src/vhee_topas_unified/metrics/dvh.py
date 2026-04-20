"""Dose-volume histogram helpers."""

from __future__ import annotations

import numpy as np


def dose_at_volume_percent(dose_values: np.ndarray, percent_volume: float) -> float:
    """Return the dose received by `percent_volume` of the structure volume."""

    values = np.asarray(dose_values, dtype=np.float64)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("dose_values must not be empty.")
    return float(np.percentile(values, 100.0 - float(percent_volume)))


def compute_dvh(values_gy: np.ndarray, dose_axis_gy: np.ndarray) -> np.ndarray:
    """Compute a cumulative DVH in percent volume."""

    values = np.asarray(values_gy, dtype=np.float64)
    axis = np.asarray(dose_axis_gy, dtype=np.float64)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("values_gy must not be empty.")
    if axis.ndim != 1:
        raise ValueError("dose_axis_gy must be one-dimensional.")
    return np.array([100.0 * np.mean(values >= dose) for dose in axis], dtype=np.float32)


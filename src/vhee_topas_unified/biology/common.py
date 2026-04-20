"""Shared helpers for biology calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def voxel_spacing_xyz_mm(voxel_size_mm: float | Iterable[float]) -> tuple[float, float, float]:
    """Normalize scalar or `(dx, dy, dz)` spacing input."""

    if np.isscalar(voxel_size_mm):
        spacing = float(voxel_size_mm)
        if spacing <= 0.0:
            raise ValueError("voxel_size_mm must be positive.")
        return spacing, spacing, spacing

    values = tuple(float(value) for value in voxel_size_mm)
    if len(values) != 3:
        raise ValueError("voxel_size_mm must be a scalar or an iterable of length 3.")
    if any(value <= 0.0 for value in values):
        raise ValueError("All voxel spacings must be positive.")
    return values


def as_species_vector(
    values: np.ndarray | Iterable[float] | float,
    *,
    name: str,
    num_species: int,
) -> np.ndarray:
    """Normalize scalar or 1D species parameters to a fixed-length vector."""

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(int(num_species), float(arr), dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != int(num_species):
        raise ValueError(f"{name} must be a scalar or a sequence of length {num_species}.")
    return arr.astype(np.float32, copy=False)


def centered_z_offset_from_surface_depth_mm(
    n_voxels: int,
    spacing_mm: float,
    depth_from_surface_mm: float,
) -> float:
    """Convert a physical depth-from-surface into a centered z-axis frame."""

    if int(n_voxels) < 1:
        raise ValueError("n_voxels must be at least 1.")
    if float(spacing_mm) <= 0.0:
        raise ValueError("spacing_mm must be positive.")
    if float(depth_from_surface_mm) < 0.0:
        raise ValueError("depth_from_surface_mm must be non-negative.")

    return float(depth_from_surface_mm) - (float(n_voxels) * float(spacing_mm) / 2.0)

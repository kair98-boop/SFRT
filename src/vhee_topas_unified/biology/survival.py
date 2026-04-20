"""Survival and effective-dose calculations extracted into the unified package."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .common import voxel_spacing_xyz_mm
from .constants import DEFAULT_ALPHA, DEFAULT_BETA


def lq_survival_from_dose(
    dose_grid: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """Compute standard LQ survival from physical dose."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    dose = np.asarray(dose_grid, dtype=np.float32)
    return np.exp(-float(alpha) * dose - float(beta) * dose**2).astype(np.float32, copy=False)


def calculate_systemic_immune_penalty(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    icd_threshold_gy: float = 10.0,
    immune_max_penalty: float = 1.0,
    immune_half_volume_cm3: float = 5.0,
) -> tuple[float, float]:
    """Return the global immune-penalty scalar and ICD volume in cm^3."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if icd_threshold_gy < 0.0:
        raise ValueError("icd_threshold_gy must be non-negative.")
    if immune_max_penalty < 0.0:
        raise ValueError("immune_max_penalty must be non-negative.")
    if immune_half_volume_cm3 <= 0.0:
        raise ValueError("immune_half_volume_cm3 must be positive.")

    dx_mm, dy_mm, dz_mm = voxel_spacing_xyz_mm(voxel_size_mm)
    voxel_volume_cm3 = (float(dx_mm) * float(dy_mm) * float(dz_mm)) / 1000.0
    icd_voxels = int(np.count_nonzero(dose_grid >= float(icd_threshold_gy)))
    icd_volume_cm3 = float(icd_voxels) * float(voxel_volume_cm3)
    immune_penalty = float(immune_max_penalty) * (
        icd_volume_cm3 / (icd_volume_cm3 + float(immune_half_volume_cm3))
    )
    return float(immune_penalty), float(icd_volume_cm3)


def calculate_phase7_survival(
    lq_survival_grid: np.ndarray,
    hazard_grid: np.ndarray,
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    scaling_factor: float,
    *,
    weight_immune: float = 0.20,
    icd_threshold_gy: float = 10.0,
    immune_max_penalty: float = 1.0,
    immune_half_volume_cm3: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Apply cumulative hazard plus systemic immune penalty."""

    if scaling_factor < 0.0:
        raise ValueError("scaling_factor must be non-negative.")
    if weight_immune < 0.0:
        raise ValueError("weight_immune must be non-negative.")

    lq_survival_grid = np.asarray(lq_survival_grid, dtype=np.float32)
    hazard_grid = np.asarray(hazard_grid, dtype=np.float32)
    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if lq_survival_grid.shape != hazard_grid.shape or lq_survival_grid.shape != dose_grid.shape:
        raise ValueError("lq_survival_grid, hazard_grid, and dose_grid must share the same shape.")

    immune_penalty, icd_volume_cm3 = calculate_systemic_immune_penalty(
        dose_grid,
        voxel_size_mm,
        icd_threshold_gy=float(icd_threshold_gy),
        immune_max_penalty=float(immune_max_penalty),
        immune_half_volume_cm3=float(immune_half_volume_cm3),
    )
    if verbose:
        print(
            f"  -> ICD Volume (>{float(icd_threshold_gy):.2f} Gy): {icd_volume_cm3:.2f} cm^3 | "
            f"P_immune scalar: {immune_penalty:.4f}"
        )

    nonlocal_penalty = hazard_grid + (float(weight_immune) * float(immune_penalty))
    return (lq_survival_grid * np.exp(-nonlocal_penalty * float(scaling_factor))).astype(np.float32, copy=False)


def calculate_effective_dose(
    final_survival_grid: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    min_survival: float = 1.0e-10,
) -> np.ndarray:
    """Invert the LQ model and return LQ-equivalent effective dose in Gy."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if min_survival <= 0.0 or min_survival > 1.0:
        raise ValueError("min_survival must be in the interval (0, 1].")

    survival = np.asarray(final_survival_grid, dtype=np.float32)
    safe_survival = np.clip(survival, float(min_survival), 1.0)
    discriminant = (float(alpha) ** 2) - (4.0 * float(beta) * np.log(safe_survival))
    return ((-float(alpha) + np.sqrt(discriminant)) / (2.0 * float(beta))).astype(np.float32, copy=False)


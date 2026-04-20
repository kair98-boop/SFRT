"""State modifiers and emission builders for the unified biology package."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .common import as_species_vector, voxel_spacing_xyz_mm


def build_state_modifier_tensors(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create spatial modifier tensors for cell type and oxygenation."""

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")
    if tumor_radius_mm <= 0.0:
        raise ValueError("tumor_radius_mm must be positive.")
    if hypoxic_radius_mm <= 0.0:
        raise ValueError("hypoxic_radius_mm must be positive.")
    if tumor_cytokine_multiplier < 0.0:
        raise ValueError("tumor_cytokine_multiplier must be non-negative.")
    if hypoxic_ros_scale < 0.0 or hypoxic_cytokine_multiplier < 0.0:
        raise ValueError("Hypoxic emission multipliers must be non-negative.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, dz_mm = voxel_spacing_xyz_mm(voxel_size_mm)

    tumor_offset_x_mm, tumor_offset_y_mm, tumor_offset_z_mm = (
        float(value) for value in tumor_center_offset_mm
    )
    if hypoxic_center_offset_mm is None:
        hypoxic_offset_x_mm, hypoxic_offset_y_mm, hypoxic_offset_z_mm = (
            tumor_offset_x_mm,
            tumor_offset_y_mm,
            tumor_offset_z_mm,
        )
    else:
        hypoxic_offset_x_mm, hypoxic_offset_y_mm, hypoxic_offset_z_mm = (
            float(value) for value in hypoxic_center_offset_mm
        )

    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    z_mm = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * float(dz_mm)

    tumor_distance_sq_mm = (
        (x_mm[:, None, None] - tumor_offset_x_mm) ** 2
        + (y_mm[None, :, None] - tumor_offset_y_mm) ** 2
        + (z_mm[None, None, :] - tumor_offset_z_mm) ** 2
    )
    tumor_mask = tumor_distance_sq_mm <= float(tumor_radius_mm) ** 2

    hypoxic_distance_sq_mm = (
        (x_mm[:, None, None] - hypoxic_offset_x_mm) ** 2
        + (y_mm[None, :, None] - hypoxic_offset_y_mm) ** 2
        + (z_mm[None, None, :] - hypoxic_offset_z_mm) ** 2
    )
    hypoxic_mask = hypoxic_distance_sq_mm <= float(hypoxic_radius_mm) ** 2

    type_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)
    oxygen_modifier = np.ones((num_species, nx, ny, nz), dtype=dtype)

    if num_species > 1:
        type_modifier[1, tumor_mask] = float(tumor_cytokine_multiplier)

    oxygen_modifier[0, hypoxic_mask] = float(hypoxic_ros_scale)
    if num_species > 1:
        oxygen_modifier[1, hypoxic_mask] = float(hypoxic_cytokine_multiplier)

    return type_modifier, oxygen_modifier, tumor_mask, hypoxic_mask


def calculate_state_dependent_emission(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
) -> np.ndarray:
    """Precompute the state-dependent emission tensor E_k(x)."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    emax_vector = as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=int(num_species),
    )
    base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
    type_modifier, oxygen_modifier, _, _ = build_state_modifier_tensors(
        dose_grid.shape,
        voxel_size_mm,
        num_species=int(num_species),
        tumor_radius_mm=float(tumor_radius_mm),
        tumor_center_offset_mm=tuple(float(value) for value in tumor_center_offset_mm),
        tumor_cytokine_multiplier=float(tumor_cytokine_multiplier),
        hypoxic_radius_mm=float(hypoxic_radius_mm),
        hypoxic_center_offset_mm=(
            None
            if hypoxic_center_offset_mm is None
            else tuple(float(value) for value in hypoxic_center_offset_mm)
        ),
        hypoxic_ros_scale=float(hypoxic_ros_scale),
        hypoxic_cytokine_multiplier=float(hypoxic_cytokine_multiplier),
        dtype=np.float32,
    )
    return (
        emax_vector[:, None, None, None]
        * base_emission[None, :, :, :]
        * type_modifier
        * oxygen_modifier
    ).astype(np.float32, copy=False)

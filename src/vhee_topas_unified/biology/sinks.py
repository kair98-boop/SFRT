"""Uptake-field builders for the unified biology package."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .common import as_species_vector, voxel_spacing_xyz_mm


def build_cylindrical_uptake_tensor(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    *,
    num_species: int = 2,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a species-specific cylindrical uptake sink running along z."""

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")
    if vessel_radius_mm <= 0.0:
        raise ValueError("vessel_radius_mm must be positive.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, _ = voxel_spacing_xyz_mm(voxel_size_mm)
    offset_x_mm, offset_y_mm = (float(value) for value in vessel_center_offset_mm)
    uptake_rates = as_species_vector(
        uptake_rates_in_vessel,
        name="uptake_rates_in_vessel",
        num_species=num_species,
    )

    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    x_rel_mm = x_mm[:, None] - offset_x_mm
    y_rel_mm = y_mm[None, :] - offset_y_mm
    vessel_mask_xy = (x_rel_mm**2 + y_rel_mm**2) <= float(vessel_radius_mm) ** 2
    vessel_mask = np.broadcast_to(vessel_mask_xy[:, :, None], (nx, ny, nz))

    uptake_tensor = np.zeros((num_species, nx, ny, nz), dtype=dtype)
    for species_idx, rate in enumerate(uptake_rates):
        uptake_tensor[species_idx, vessel_mask] = float(rate)

    return uptake_tensor, vessel_mask


def build_vessel_network_uptake_tensor(
    grid_shape: Sequence[int],
    voxel_size_mm: float | Iterable[float],
    vessel_specs: Sequence[dict],
    *,
    num_species: int = 2,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a multi-vessel uptake tensor from piecewise-linear centerlines."""

    if len(tuple(grid_shape)) != 3:
        raise ValueError("grid_shape must contain exactly three dimensions.")
    if num_species < 1:
        raise ValueError("num_species must be at least 1.")

    nx, ny, nz = (int(value) for value in grid_shape)
    dx_mm, dy_mm, dz_mm = voxel_spacing_xyz_mm(voxel_size_mm)
    x_mm = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * float(dx_mm)
    y_mm = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * float(dy_mm)
    z_mm = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * float(dz_mm)

    x_grid_mm = x_mm[:, None]
    y_grid_mm = y_mm[None, :]

    uptake_tensor = np.zeros((num_species, nx, ny, nz), dtype=dtype)
    vessel_union_mask = np.zeros((nx, ny, nz), dtype=bool)

    for spec in vessel_specs:
        nodes = np.asarray(spec["nodes_mm"], dtype=np.float32)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 2:
            raise ValueError("Each vessel spec must provide at least two 3D nodes in nodes_mm.")
        radius_mm = float(spec["radius_mm"])
        if radius_mm <= 0.0:
            raise ValueError("Each vessel radius_mm must be positive.")
        uptake_rates = as_species_vector(
            spec.get("uptake_rates_in_vessel", (0.05, 0.60)),
            name="uptake_rates_in_vessel",
            num_species=num_species,
        )

        order = np.argsort(nodes[:, 2])
        nodes = nodes[order]
        z_nodes = nodes[:, 2]
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        z_min = float(np.min(z_nodes))
        z_max = float(np.max(z_nodes))

        slice_indices = np.flatnonzero((z_mm >= z_min) & (z_mm <= z_max))
        if slice_indices.size == 0:
            continue

        x_interp = np.interp(z_mm[slice_indices], z_nodes, x_nodes)
        y_interp = np.interp(z_mm[slice_indices], z_nodes, y_nodes)
        radius_sq = float(radius_mm) ** 2

        for local_idx, z_idx in enumerate(slice_indices):
            mask_xy = (
                (x_grid_mm - float(x_interp[local_idx])) ** 2
                + (y_grid_mm - float(y_interp[local_idx])) ** 2
            ) <= radius_sq
            if not np.any(mask_xy):
                continue
            vessel_union_mask[:, :, int(z_idx)] |= mask_xy
            for species_idx, rate in enumerate(uptake_rates):
                uptake_tensor[species_idx, :, :, int(z_idx)][mask_xy] = np.maximum(
                    uptake_tensor[species_idx, :, :, int(z_idx)][mask_xy],
                    float(rate),
                )

    return uptake_tensor, vessel_union_mask

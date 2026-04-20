"""PDE-facing biology API.

This module now owns the main multispecies explicit solver and the standard
temporal hazard workflow. A small legacy bridge remains available for pieces
that have not been migrated yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import importlib.util
import sys
import time

import numpy as np

from ..legacy import resolve_legacy_root
from .common import as_species_vector, voxel_spacing_xyz_mm
from .emission import calculate_state_dependent_emission
from .sinks import build_cylindrical_uptake_tensor


def cfl_stability_limit_3d(
    voxel_size_mm: float | Iterable[float],
    diffusion_coeff_max: float,
) -> float:
    """Return the explicit-Euler CFL time-step limit for 3D diffusion."""

    if diffusion_coeff_max <= 0.0:
        raise ValueError("diffusion_coeff_max must be positive.")
    dx_mm, dy_mm, dz_mm = voxel_spacing_xyz_mm(voxel_size_mm)
    inverse_spacing_sum = (1.0 / dx_mm**2) + (1.0 / dy_mm**2) + (1.0 / dz_mm**2)
    return 1.0 / (2.0 * float(diffusion_coeff_max) * inverse_spacing_sum)


def anisotropic_laplacian_3d(
    field: np.ndarray,
    voxel_size_mm: float | Iterable[float],
) -> np.ndarray:
    """Compute a 3D Laplacian with finite differences and edge padding."""

    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("field must be a 3D numpy array.")

    dx_mm, dy_mm, dz_mm = voxel_spacing_xyz_mm(voxel_size_mm)
    padded = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")

    d2x = (padded[2:, 1:-1, 1:-1] - 2.0 * field + padded[:-2, 1:-1, 1:-1]) / (dx_mm**2)
    d2y = (padded[1:-1, 2:, 1:-1] - 2.0 * field + padded[1:-1, :-2, 1:-1]) / (dy_mm**2)
    d2z = (padded[1:-1, 1:-1, 2:] - 2.0 * field + padded[1:-1, 1:-1, :-2]) / (dz_mm**2)
    return (d2x + d2y + d2z).astype(np.float32, copy=False)


def _infer_num_species(
    uptake_tensor: np.ndarray | None,
    *species_parameters: Sequence[float] | np.ndarray | float,
) -> int:
    if uptake_tensor is not None:
        if uptake_tensor.ndim != 4:
            raise ValueError("uptake_tensor must have shape (species, x, y, z).")
        return int(uptake_tensor.shape[0])

    inferred = 1
    for values in species_parameters:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            continue
        if arr.ndim != 1:
            raise ValueError("Species parameters must be scalars or 1D sequences.")
        inferred = max(inferred, int(arr.shape[0]))
    return max(2, inferred)


def solve_multispecies_pde_3d(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
    diffusion_coeffs: Sequence[float] | np.ndarray | float = (0.8, 0.4),
    decay_coeffs: Sequence[float] | np.ndarray | float = (0.2, 0.02),
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    emission_tensor: np.ndarray | None = None,
    state_dependent_emission: bool = False,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    uptake_tensor: np.ndarray | None = None,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    progress_interval: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Solve the multi-species explicit reaction-diffusion PDE."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    num_species = _infer_num_species(
        uptake_tensor,
        diffusion_coeffs,
        decay_coeffs,
        emission_emax,
        uptake_rates_in_vessel,
    )

    diffusion_vector = as_species_vector(
        diffusion_coeffs,
        name="diffusion_coeffs",
        num_species=num_species,
    )
    decay_vector = as_species_vector(
        decay_coeffs,
        name="decay_coeffs",
        num_species=num_species,
    )
    emax_vector = as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=num_species,
    )

    if np.any(diffusion_vector <= 0.0):
        raise ValueError("All diffusion coefficients must be positive.")
    if np.any(decay_vector < 0.0):
        raise ValueError("All decay coefficients must be non-negative.")
    if np.any(emax_vector < 0.0):
        raise ValueError("All emission_emax values must be non-negative.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(np.max(diffusion_vector)))
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the maximum diffusion coefficient."
        )

    if uptake_tensor is None:
        uptake_field, _ = build_cylindrical_uptake_tensor(
            dose_grid.shape,
            voxel_size_mm,
            num_species=num_species,
            vessel_radius_mm=vessel_radius_mm,
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates_in_vessel,
            dtype=np.float32,
        )
    else:
        uptake_field = np.asarray(uptake_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if uptake_field.shape != expected_shape:
            raise ValueError(
                f"uptake_tensor must have shape {expected_shape}, got {uptake_field.shape}."
            )
        if np.any(uptake_field < 0.0):
            raise ValueError("uptake_tensor must be non-negative.")

    if verbose:
        print("\n--- Initializing Multi-Species Heterogeneous PDE Solver ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Tensor shape: {(num_species, *dose_grid.shape)}")
        print(f"Voxel size (mm): {voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")

    start_time = time.time()
    if emission_tensor is not None:
        source_terms = np.asarray(emission_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if source_terms.shape != expected_shape:
            raise ValueError(
                f"emission_tensor must have shape {expected_shape}, got {source_terms.shape}."
            )
        if np.any(source_terms < 0.0):
            raise ValueError("emission_tensor must be non-negative.")
        emission_mode = "precomputed_tensor"
    elif state_dependent_emission:
        source_terms = calculate_state_dependent_emission(
            dose_grid,
            voxel_size_mm,
            num_species=num_species,
            emission_emax=emax_vector,
            emission_gamma_per_gy=float(emission_gamma_per_gy),
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
        )
        emission_mode = "state_dependent"
    else:
        base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
        source_terms = emax_vector[:, None, None, None] * base_emission[None, :, :, :]
        emission_mode = "uniform_saturated"

    concentration = np.zeros((num_species, *dose_grid.shape), dtype=np.float32)

    if verbose:
        print(f"Emission mode: {emission_mode}")
        print(f"Simulating {steps} time steps for {num_species} species...")

    for step in range(int(steps)):
        for species_idx in range(num_species):
            laplacian = anisotropic_laplacian_3d(concentration[species_idx], voxel_size_mm)
            dcdt = (
                diffusion_vector[species_idx] * laplacian
                - decay_vector[species_idx] * concentration[species_idx]
                - uptake_field[species_idx] * concentration[species_idx]
                + source_terms[species_idx]
            )
            concentration[species_idx] += dcdt * float(dt)
            np.maximum(concentration[species_idx], 0.0, out=concentration[species_idx])

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"Solver finished in {time.time() - start_time:.2f} seconds.")

    return concentration


def solve_multispecies_pde_3d_with_hazard(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    steps: int = 400,
    dt: float = 0.15,
    diffusion_coeffs: Sequence[float] | np.ndarray | float = (0.8, 0.4),
    decay_coeffs: Sequence[float] | np.ndarray | float = (0.2, 0.02),
    emission_emax: Sequence[float] | np.ndarray | float = (1.5, 0.8),
    emission_gamma_per_gy: float = 0.35,
    emission_tensor: np.ndarray | None = None,
    state_dependent_emission: bool = False,
    tumor_radius_mm: float = 15.0,
    tumor_center_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tumor_cytokine_multiplier: float = 2.0,
    hypoxic_radius_mm: float = 5.0,
    hypoxic_center_offset_mm: tuple[float, float, float] | None = None,
    hypoxic_ros_scale: float = 0.10,
    hypoxic_cytokine_multiplier: float = 3.0,
    uptake_tensor: np.ndarray | None = None,
    vessel_radius_mm: float = 3.0,
    vessel_center_offset_mm: tuple[float, float] = (0.0, 0.0),
    uptake_rates_in_vessel: Sequence[float] | np.ndarray | float = (0.05, 0.60),
    hazard_weights: Sequence[float] | np.ndarray = (0.40, 0.40),
    progress_interval: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the multi-species PDE and integrate a cumulative hazard grid."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if emission_gamma_per_gy < 0.0:
        raise ValueError("emission_gamma_per_gy must be non-negative.")

    num_species = _infer_num_species(
        uptake_tensor,
        diffusion_coeffs,
        decay_coeffs,
        emission_emax,
        uptake_rates_in_vessel,
    )
    if num_species < 2:
        raise ValueError("Phase 7 hazard integration requires at least two species channels.")

    diffusion_vector = as_species_vector(
        diffusion_coeffs,
        name="diffusion_coeffs",
        num_species=num_species,
    )
    decay_vector = as_species_vector(
        decay_coeffs,
        name="decay_coeffs",
        num_species=num_species,
    )
    emax_vector = as_species_vector(
        emission_emax,
        name="emission_emax",
        num_species=num_species,
    )

    if np.any(diffusion_vector <= 0.0):
        raise ValueError("All diffusion coefficients must be positive.")
    if np.any(decay_vector < 0.0):
        raise ValueError("All decay coefficients must be non-negative.")
    if np.any(emax_vector < 0.0):
        raise ValueError("All emission_emax values must be non-negative.")

    dt_limit = cfl_stability_limit_3d(voxel_size_mm, float(np.max(diffusion_vector)))
    if dt > dt_limit:
        raise ValueError(
            f"Chosen dt={dt:.6f} exceeds the CFL stability limit {dt_limit:.6f}. "
            "Reduce dt or the maximum diffusion coefficient."
        )

    if uptake_tensor is None:
        uptake_field, _ = build_cylindrical_uptake_tensor(
            dose_grid.shape,
            voxel_size_mm,
            num_species=num_species,
            vessel_radius_mm=vessel_radius_mm,
            vessel_center_offset_mm=vessel_center_offset_mm,
            uptake_rates_in_vessel=uptake_rates_in_vessel,
            dtype=np.float32,
        )
    else:
        uptake_field = np.asarray(uptake_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if uptake_field.shape != expected_shape:
            raise ValueError(
                f"uptake_tensor must have shape {expected_shape}, got {uptake_field.shape}."
            )
        if np.any(uptake_field < 0.0):
            raise ValueError("uptake_tensor must be non-negative.")

    if emission_tensor is not None:
        source_terms = np.asarray(emission_tensor, dtype=np.float32)
        expected_shape = (num_species, *dose_grid.shape)
        if source_terms.shape != expected_shape:
            raise ValueError(
                f"emission_tensor must have shape {expected_shape}, got {source_terms.shape}."
            )
        if np.any(source_terms < 0.0):
            raise ValueError("emission_tensor must be non-negative.")
        emission_mode = "precomputed_tensor"
    elif state_dependent_emission:
        source_terms = calculate_state_dependent_emission(
            dose_grid,
            voxel_size_mm,
            num_species=num_species,
            emission_emax=emax_vector,
            emission_gamma_per_gy=float(emission_gamma_per_gy),
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
        )
        emission_mode = "state_dependent"
    else:
        base_emission = (1.0 - np.exp(-float(emission_gamma_per_gy) * dose_grid)).astype(np.float32)
        source_terms = emax_vector[:, None, None, None] * base_emission[None, :, :, :]
        emission_mode = "uniform_saturated"

    local_weights = as_species_vector(
        hazard_weights,
        name="hazard_weights",
        num_species=2,
    )
    if np.any(local_weights < 0.0):
        raise ValueError("hazard_weights must be non-negative.")

    concentration = np.zeros((num_species, *dose_grid.shape), dtype=np.float32)
    hazard_grid = np.zeros(dose_grid.shape, dtype=np.float32)

    if verbose:
        print("\n--- Initializing Multi-Species Temporal Hazard PDE Solver ---")
        print(f"Grid shape: {dose_grid.shape}")
        print(f"Tensor shape: {(num_species, *dose_grid.shape)}")
        print(f"Voxel size (mm): {voxel_spacing_xyz_mm(voxel_size_mm)}")
        print(f"CFL dt limit: {dt_limit:.6f}")
        print(f"Using dt={dt:.6f} for {steps} steps.")
        print(f"Emission mode: {emission_mode}")
        print(
            f"Simulating {steps} time steps and integrating cumulative hazard with "
            f"weights [{float(local_weights[0]):.2f}, {float(local_weights[1]):.2f}]..."
        )

    start_time = time.time()
    for step in range(int(steps)):
        for species_idx in range(num_species):
            laplacian = anisotropic_laplacian_3d(concentration[species_idx], voxel_size_mm)
            dcdt = (
                diffusion_vector[species_idx] * laplacian
                - decay_vector[species_idx] * concentration[species_idx]
                - uptake_field[species_idx] * concentration[species_idx]
                + source_terms[species_idx]
            )
            concentration[species_idx] += dcdt * float(dt)
            np.maximum(concentration[species_idx], 0.0, out=concentration[species_idx])

        instantaneous_stress = (
            float(local_weights[0]) * concentration[0]
            + float(local_weights[1]) * concentration[1]
        )
        hazard_grid += instantaneous_stress * float(dt)

        if verbose and progress_interval > 0 and (step + 1) % int(progress_interval) == 0:
            print(f"  ... Step {step + 1}/{steps} completed.")

    if verbose:
        print(f"Temporal hazard solver finished in {time.time() - start_time:.2f} seconds.")

    return concentration, hazard_grid


def run_pde_temporal_integration(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    D_cyto: float,
    lambda_cyto: float,
    gamma: float,
    u_k: np.ndarray | None = None,
    M_oxygen: np.ndarray | None = None,
    M_type: np.ndarray | None = None,
    D_ros: float = 0.8,
    lambda_ros: float = 0.2,
    Emax_ros: float = 1.5,
    Emax_cyto: float = 0.8,
    w_ros: float = 0.40,
    w_cyto: float = 0.40,
    steps: int = 400,
    dt: float = 0.12,
    progress_interval: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Convenience wrapper returning only the cumulative hazard grid."""

    dose_grid = np.asarray(dose_grid, dtype=np.float32)
    if dose_grid.ndim != 3:
        raise ValueError("dose_grid must be a 3D numpy array.")

    if M_oxygen is None:
        oxygen_modifier = np.ones((2, *dose_grid.shape), dtype=np.float32)
    else:
        oxygen_modifier = np.asarray(M_oxygen, dtype=np.float32)
    if M_type is None:
        type_modifier = np.ones((2, *dose_grid.shape), dtype=np.float32)
    else:
        type_modifier = np.asarray(M_type, dtype=np.float32)
    expected_modifier_shape = (2, *dose_grid.shape)
    if oxygen_modifier.shape != expected_modifier_shape or type_modifier.shape != expected_modifier_shape:
        raise ValueError(
            f"M_oxygen and M_type must each have shape {expected_modifier_shape}."
        )

    base_emission = (1.0 - np.exp(-float(gamma) * dose_grid)).astype(np.float32)
    emission_tensor = (
        np.asarray([float(Emax_ros), float(Emax_cyto)], dtype=np.float32)[:, None, None, None]
        * base_emission[None, :, :, :]
        * type_modifier
        * oxygen_modifier
    ).astype(np.float32, copy=False)

    _, hazard_grid = solve_multispecies_pde_3d_with_hazard(
        dose_grid=dose_grid,
        voxel_size_mm=voxel_size_mm,
        steps=int(steps),
        dt=float(dt),
        diffusion_coeffs=(float(D_ros), float(D_cyto)),
        decay_coeffs=(float(lambda_ros), float(lambda_cyto)),
        emission_emax=(float(Emax_ros), float(Emax_cyto)),
        emission_gamma_per_gy=float(gamma),
        emission_tensor=emission_tensor,
        uptake_tensor=(None if u_k is None else np.asarray(u_k, dtype=np.float32)),
        hazard_weights=(float(w_ros), float(w_cyto)),
        progress_interval=int(progress_interval),
        verbose=bool(verbose),
    )
    return hazard_grid


def _load_legacy_multispecies_module(legacy_root: str | Path | None = None):
    """Dynamically load the legacy multispecies solver module."""

    root = resolve_legacy_root(legacy_root)
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / "bystander_multispecies_pde_solver.py"
    spec = importlib.util.spec_from_file_location(
        "vhee_topas_legacy_bystander_multispecies_pde_solver",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for legacy solver: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_run_pde_temporal_integration(
    dose_grid: np.ndarray,
    voxel_size_mm: float | Iterable[float],
    *,
    D_cyto: float,
    lambda_cyto: float,
    gamma: float,
    u_k: np.ndarray | None = None,
    M_oxygen: np.ndarray | None = None,
    M_type: np.ndarray | None = None,
    D_ros: float = 0.8,
    lambda_ros: float = 0.2,
    Emax_ros: float = 1.5,
    Emax_cyto: float = 0.8,
    w_ros: float = 0.40,
    w_cyto: float = 0.40,
    steps: int = 400,
    dt: float = 0.12,
    progress_interval: int = 50,
    verbose: bool = True,
    legacy_root: str | Path | None = None,
) -> np.ndarray:
    """Compatibility bridge to the legacy temporal PDE engine."""

    module = _load_legacy_multispecies_module(legacy_root)
    return module.run_pde_temporal_integration(
        dose_grid=dose_grid,
        voxel_size_mm=voxel_size_mm,
        D_cyto=float(D_cyto),
        lambda_cyto=float(lambda_cyto),
        gamma=float(gamma),
        u_k=u_k,
        M_oxygen=M_oxygen,
        M_type=M_type,
        D_ros=float(D_ros),
        lambda_ros=float(lambda_ros),
        Emax_ros=float(Emax_ros),
        Emax_cyto=float(Emax_cyto),
        w_ros=float(w_ros),
        w_cyto=float(w_cyto),
        steps=int(steps),
        dt=float(dt),
        progress_interval=int(progress_interval),
        verbose=bool(verbose),
    )

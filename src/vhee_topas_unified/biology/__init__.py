"""Biology-facing API for the unified SFRT package.

This package is the start of the real model migration out of the legacy
research scripts and into a reusable library layout.
"""

from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    D_ROS,
    EMAX_CYTO,
    EMAX_ROS,
    LAMBDA_ROS,
    LOCKED_D_CYTO,
    LOCKED_GAMMA,
    LOCKED_LAMBDA_CYTO,
    LOCKED_SCALING_FACTOR,
    W_CYTO,
    W_IMMUNE,
    W_ROS,
)
from .common import centered_z_offset_from_surface_depth_mm
from .emission import calculate_state_dependent_emission, build_state_modifier_tensors
from .pde import (
    anisotropic_laplacian_3d,
    cfl_stability_limit_3d,
    legacy_run_pde_temporal_integration,
    run_pde_temporal_integration,
    solve_multispecies_pde_3d,
    solve_multispecies_pde_3d_with_hazard,
)
from .sinks import build_cylindrical_uptake_tensor, build_vessel_network_uptake_tensor
from .survival import (
    calculate_effective_dose,
    calculate_phase7_survival,
    calculate_systemic_immune_penalty,
    lq_survival_from_dose,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "D_ROS",
    "EMAX_CYTO",
    "EMAX_ROS",
    "LAMBDA_ROS",
    "LOCKED_D_CYTO",
    "LOCKED_GAMMA",
    "LOCKED_LAMBDA_CYTO",
    "LOCKED_SCALING_FACTOR",
    "W_CYTO",
    "W_IMMUNE",
    "W_ROS",
    "build_cylindrical_uptake_tensor",
    "build_state_modifier_tensors",
    "build_vessel_network_uptake_tensor",
    "cfl_stability_limit_3d",
    "calculate_effective_dose",
    "calculate_phase7_survival",
    "calculate_state_dependent_emission",
    "calculate_systemic_immune_penalty",
    "centered_z_offset_from_surface_depth_mm",
    "anisotropic_laplacian_3d",
    "legacy_run_pde_temporal_integration",
    "lq_survival_from_dose",
    "run_pde_temporal_integration",
    "solve_multispecies_pde_3d",
    "solve_multispecies_pde_3d_with_hazard",
]

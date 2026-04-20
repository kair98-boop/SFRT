"""Simple voxelized head-and-neck audit phantom."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import capped_cylinder_along_y_mask, centered_axis_mm, cylinder_along_y_mask, ellipsoid_mask


AIR_TAG = np.int16(0)
SOFT_TAG = np.int16(10)
BONE_TAG = np.int16(20)


def build_simple_headneck_phantom(
    *,
    size_x_cm: float = 18.0,
    size_y_cm: float = 24.0,
    size_z_cm: float = 16.0,
    voxel_mm: float = 2.0,
) -> Dict[str, object]:
    """Build the earlier simple audit-style head-and-neck phantom."""

    dx = dy = dz = float(voxel_mm)
    nx = int(round(float(size_x_cm) * 10.0 / dx))
    ny = int(round(float(size_y_cm) * 10.0 / dy))
    nz = int(round(float(size_z_cm) * 10.0 / dz))
    x_mm = centered_axis_mm(nx, dx)
    y_mm = centered_axis_mm(ny, dy)
    z_mm = centered_axis_mm(nz, dz)

    head = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 38.0, 0.0), radii_mm=(78.0, 76.0, 70.0))
    neck = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -34.0, 0.0), radii_mm=(62.0, 80.0, 48.0))
    shoulder_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-58.0, -88.0, 0.0), radii_mm=(42.0, 28.0, 52.0))
    shoulder_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(58.0, -88.0, 0.0), radii_mm=(42.0, 28.0, 52.0))
    body_mask = head | neck | shoulder_l | shoulder_r

    skull_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 40.0, 0.0), radii_mm=(74.0, 70.0, 64.0))
    skull_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 40.0, 0.0), radii_mm=(66.0, 62.0, 57.0))
    skull_mask = skull_outer & ~skull_inner

    mandible_arch = capped_cylinder_along_y_mask(
        x_mm,
        y_mm,
        z_mm,
        center_x_mm=0.0,
        center_z_mm=-8.0,
        radius_x_mm=44.0,
        radius_z_mm=24.0,
        y_min_mm=-8.0,
        y_max_mm=14.0,
    )
    mandible_gap = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 0.0, 10.0), radii_mm=(36.0, 18.0, 32.0))
    ramus_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-42.0, 8.0, 6.0), radii_mm=(8.0, 26.0, 10.0))
    ramus_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(42.0, 8.0, 6.0), radii_mm=(8.0, 26.0, 10.0))
    mandible_mask = (mandible_arch & ~mandible_gap) | ramus_l | ramus_r

    vertebrae_mask = np.zeros((nx, ny, nz), dtype=bool)
    for center_y in np.arange(-62.0, 42.0, 18.0):
        body = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 24.0), radii_mm=(12.0, 8.0, 10.0))
        spinous = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 34.0), radii_mm=(4.0, 8.0, 6.0))
        vertebrae_mask |= body | spinous

    airway_mask = capped_cylinder_along_y_mask(
        x_mm,
        y_mm,
        z_mm,
        center_x_mm=0.0,
        center_z_mm=-8.0,
        radius_x_mm=9.0,
        radius_z_mm=11.0,
        y_min_mm=-12.0,
        y_max_mm=34.0,
    )
    trachea_mask = cylinder_along_y_mask(
        x_mm,
        y_mm,
        z_mm,
        center_x_mm=0.0,
        center_z_mm=-7.0,
        radius_mm=7.0,
        y_min_mm=-76.0,
        y_max_mm=-12.0,
    )
    oral_cavity = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 18.0, -18.0), radii_mm=(16.0, 12.0, 10.0))
    airway_mask |= trachea_mask | oral_cavity

    parotid_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-50.0, -4.0, -4.0), radii_mm=(16.0, 24.0, 12.0)) & body_mask
    parotid_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(50.0, -4.0, -4.0), radii_mm=(16.0, 24.0, 12.0)) & body_mask
    spinal_cord = cylinder_along_y_mask(
        x_mm,
        y_mm,
        z_mm,
        center_x_mm=0.0,
        center_z_mm=22.0,
        radius_mm=4.5,
        y_min_mm=-70.0,
        y_max_mm=56.0,
    ) & body_mask
    brainstem = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 58.0, 20.0), radii_mm=(10.0, 18.0, 12.0)) & body_mask

    gtv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(28.0, 30.0, 24.0))
    gtv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(24.0, 38.0, 21.0))
    gtv_mask = (gtv_primary | gtv_nodal) & body_mask & ~airway_mask & ~spinal_cord & ~brainstem

    ptv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(32.0, 34.0, 28.0))
    ptv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(28.0, 42.0, 25.0))
    ptv_mask = (ptv_primary | ptv_nodal) & body_mask & ~airway_mask

    hypoxic_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(22.0, 8.0, -4.0), radii_mm=(16.0, 18.0, 14.0))
    hypoxic_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -34.0, 2.0), radii_mm=(12.0, 20.0, 11.0))
    hypoxic_mask = (hypoxic_primary | hypoxic_nodal) & gtv_mask

    body_mask &= ~airway_mask

    tag_grid = np.full((nx, ny, nz), AIR_TAG, dtype=np.int16)
    tag_grid[body_mask] = SOFT_TAG
    tag_grid[skull_mask | mandible_mask | vertebrae_mask] = BONE_TAG
    tag_grid[airway_mask & body_mask] = AIR_TAG

    structures = {
        "BODY": body_mask,
        "GTV": gtv_mask,
        "TUMOUR": gtv_mask,
        "PTV": ptv_mask,
        "HYPOXIA": hypoxic_mask,
        "PAROTID_L": parotid_l,
        "PAROTID_R": parotid_r,
        "SPINAL_CORD": spinal_cord,
        "BRAINSTEM": brainstem,
        "MANDIBLE": mandible_mask & body_mask,
        "AIRWAY": airway_mask,
        "BONE": (skull_mask | mandible_mask | vertebrae_mask) & body_mask,
    }

    voxel_volume_cc = (dx * dy * dz) / 1000.0
    meta = {
        "grid_shape": [int(nx), int(ny), int(nz)],
        "voxel_size_mm": [dx, dy, dz],
        "size_cm": [float(size_x_cm), float(size_y_cm), float(size_z_cm)],
        "voxel_volume_cc": float(voxel_volume_cc),
        "structure_volumes_cc": {
            name: float(np.count_nonzero(mask) * voxel_volume_cc)
            for name, mask in structures.items()
            if name not in {"AIRWAY", "BONE", "HYPOXIA"}
        },
        "anatomical_note": (
            "Synthetic IAEA-style head-and-neck audit surrogate with external contour, "
            "airway, skull/mandible, cervical vertebrae, bilateral parotids, spinal cord, "
            "brainstem, and a bulky right-sided oropharyngeal-nodal target."
        ),
    }
    return {
        "tag_grid": tag_grid,
        "structures": structures,
        "axes_mm": {"x": x_mm, "y": y_mm, "z": z_mm},
        "meta": meta,
    }

"""Detailed synthetic head-and-neck phantom and planning-target wrapper."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import (
    capped_cylinder_along_y_mask,
    centered_axis_mm,
    combine_polylines,
    cylinder_along_y_mask,
    ellipsoid_mask,
)
from .materials import build_material_tag_grid


DENSITY_G_CM3 = {
    "AIR": 0.0012,
    "SOFT_TISSUE": 1.04,
    "BRAIN": 1.04,
    "BLOOD_BRAIN_BARRIER": 1.06,
    "BRAINSTEM": 1.04,
    "SPINAL_CORD": 1.04,
    "PAROTID": 1.03,
    "SUBMANDIBULAR": 1.04,
    "THYROID": 1.05,
    "PARATHYROIDS": 1.05,
    "BLOOD": 1.06,
    "TUMOUR": 1.05,
    "SKULL_CORTICAL_BONE": 1.85,
    "MANDIBLE_MAXILLA_BONE": 1.80,
    "VERTEBRAL_BONE": 1.45,
}


def build_detailed_headneck_phantom(
    *,
    size_x_cm: float = 20.0,
    size_y_cm: float = 26.0,
    size_z_cm: float = 18.0,
    voxel_mm: float = 1.5,
) -> Dict[str, object]:
    """Build the detailed heterogeneous anatomical head-and-neck phantom."""

    dx = dy = dz = float(voxel_mm)
    nx = int(round(float(size_x_cm) * 10.0 / dx))
    ny = int(round(float(size_y_cm) * 10.0 / dy))
    nz = int(round(float(size_z_cm) * 10.0 / dz))
    x_mm = centered_axis_mm(nx, dx)
    y_mm = centered_axis_mm(ny, dy)
    z_mm = centered_axis_mm(nz, dz)

    head = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 42.0, 0.0), radii_mm=(80.0, 78.0, 70.0))
    neck = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -30.0, 0.0), radii_mm=(66.0, 88.0, 50.0))
    shoulder_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-64.0, -96.0, 0.0), radii_mm=(46.0, 30.0, 56.0))
    shoulder_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(64.0, -96.0, 0.0), radii_mm=(46.0, 30.0, 56.0))
    torso_stub = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -118.0, 0.0), radii_mm=(78.0, 24.0, 62.0))
    body_mask = head | neck | shoulder_l | shoulder_r | torso_stub

    skull_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 44.0, 0.0), radii_mm=(75.0, 71.0, 64.0))
    skull_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 44.0, 0.0), radii_mm=(67.0, 63.0, 57.0))
    skull_mask = skull_outer & ~skull_inner

    mandible_arch = capped_cylinder_along_y_mask(
        x_mm,
        y_mm,
        z_mm,
        center_x_mm=0.0,
        center_z_mm=-9.0,
        radius_x_mm=46.0,
        radius_z_mm=25.0,
        y_min_mm=-8.0,
        y_max_mm=15.0,
    )
    mandible_gap = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 1.0, 10.0), radii_mm=(38.0, 18.0, 34.0))
    ramus_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-43.0, 10.0, 5.0), radii_mm=(8.0, 28.0, 10.0))
    ramus_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(43.0, 10.0, 5.0), radii_mm=(8.0, 28.0, 10.0))
    maxilla = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 17.0, -3.0), radii_mm=(34.0, 14.0, 18.0))
    mandible_mask = ((mandible_arch & ~mandible_gap) | ramus_l | ramus_r) & body_mask

    vertebrae_mask = np.zeros((nx, ny, nz), dtype=bool)
    for center_y in np.arange(-76.0, 42.0, 16.0):
        vertebral_body = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 24.0), radii_mm=(13.0, 8.0, 10.0))
        spinous = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, float(center_y), 35.0), radii_mm=(4.0, 9.0, 6.0))
        transverse_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-10.0, float(center_y), 25.0), radii_mm=(4.0, 6.0, 5.0))
        transverse_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(10.0, float(center_y), 25.0), radii_mm=(4.0, 6.0, 5.0))
        vertebrae_mask |= vertebral_body | spinous | transverse_l | transverse_r

    airway_mask = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-9.0, radius_x_mm=9.0, radius_z_mm=11.0, y_min_mm=-6.0, y_max_mm=36.0
    )
    trachea_mask = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=-8.0, radius_mm=7.0, y_min_mm=-86.0, y_max_mm=-6.0
    )
    oral_cavity = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 18.0, -18.0), radii_mm=(17.0, 12.0, 11.0))
    larynx = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -12.0, -5.0), radii_mm=(13.0, 16.0, 10.0))
    esophagus = capped_cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=6.0, radius_x_mm=5.0, radius_z_mm=4.0, y_min_mm=-86.0, y_max_mm=-10.0
    )
    airway_complex = airway_mask | trachea_mask | oral_cavity

    parotid_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-50.0, -4.0, -4.0), radii_mm=(16.0, 25.0, 13.0)) & body_mask
    parotid_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(50.0, -4.0, -4.0), radii_mm=(16.0, 25.0, 13.0)) & body_mask
    submandibular_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-31.0, -16.0, -5.0), radii_mm=(12.0, 10.0, 8.0)) & body_mask
    submandibular_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(31.0, -16.0, -5.0), radii_mm=(12.0, 10.0, 8.0)) & body_mask

    spinal_cord = cylinder_along_y_mask(
        x_mm, y_mm, z_mm, center_x_mm=0.0, center_z_mm=22.5, radius_mm=4.5, y_min_mm=-82.0, y_max_mm=62.0
    ) & body_mask
    brain_mask = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(61.0, 58.0, 52.0))
    brain_mask &= skull_inner & body_mask
    brainstem = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 58.0, 19.0), radii_mm=(10.0, 18.0, 12.0)) & body_mask
    bbb_outer = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(63.5, 60.5, 54.5))
    bbb_inner = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, 46.0, 0.0), radii_mm=(61.8, 58.8, 52.8))
    blood_brain_barrier = (bbb_outer & ~bbb_inner) & skull_inner & body_mask

    thyroid_lobe_l = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -47.0, -2.0), radii_mm=(10.0, 16.0, 7.0))
    thyroid_lobe_r = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -47.0, -2.0), radii_mm=(10.0, 16.0, 7.0))
    thyroid_isthmus = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(0.0, -48.0, -4.0), radii_mm=(9.0, 5.0, 4.0))
    thyroid_mask = (thyroid_lobe_l | thyroid_lobe_r | thyroid_isthmus) & body_mask & ~trachea_mask

    parathyroid_l_sup = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -40.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_l_inf = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(-18.0, -55.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_r_sup = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -40.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_r_inf = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(18.0, -55.0, 4.0), radii_mm=(3.2, 4.0, 2.8))
    parathyroid_mask = (parathyroid_l_sup | parathyroid_l_inf | parathyroid_r_sup | parathyroid_r_inf) & body_mask

    tumor_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(25.0, 27.0, 22.0))
    tumor_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(21.0, 34.0, 18.0))
    tumor_mask = (tumor_primary | tumor_nodal) & body_mask & ~airway_complex & ~spinal_cord & ~brainstem

    artery_specs = [
        {"name": "common_carotid_l", "nodes_mm": [(-22, -88, 2), (-22, -60, 2), (-21, -36, 3), (-18, -12, 4), (-16, 5, 4)], "radius_mm": 3.3},
        {"name": "common_carotid_r", "nodes_mm": [(22, -88, 2), (22, -60, 2), (21, -36, 3), (18, -12, 4), (16, 5, 4)], "radius_mm": 3.3},
        {"name": "internal_carotid_l", "nodes_mm": [(-16, 5, 4), (-18, 22, 6), (-20, 40, 7), (-18, 60, 8)], "radius_mm": 2.6},
        {"name": "internal_carotid_r", "nodes_mm": [(16, 5, 4), (18, 22, 6), (20, 40, 7), (18, 60, 8)], "radius_mm": 2.6},
        {"name": "external_carotid_l", "nodes_mm": [(-16, 5, 4), (-14, 16, 1), (-11, 28, -2), (-8, 42, -6)], "radius_mm": 2.1},
        {"name": "external_carotid_r", "nodes_mm": [(16, 5, 4), (14, 16, 1), (11, 28, -2), (8, 42, -6)], "radius_mm": 2.1},
        {"name": "vertebral_artery_l", "nodes_mm": [(-12, -86, 18), (-11, -52, 19), (-10, -20, 20), (-8, 12, 21), (-6, 42, 22)], "radius_mm": 1.5},
        {"name": "vertebral_artery_r", "nodes_mm": [(12, -86, 18), (11, -52, 19), (10, -20, 20), (8, 12, 21), (6, 42, 22)], "radius_mm": 1.5},
        {"name": "superior_thyroid_artery_l", "nodes_mm": [(-14, -24, 1), (-16, -32, -1), (-19, -42, -2)], "radius_mm": 1.2},
        {"name": "superior_thyroid_artery_r", "nodes_mm": [(14, -24, 1), (16, -32, -1), (19, -42, -2)], "radius_mm": 1.2},
        {"name": "lingual_facial_branch_l", "nodes_mm": [(-11, 16, 1), (-18, 20, -5), (-28, 12, -12)], "radius_mm": 1.1},
        {"name": "lingual_facial_branch_r", "nodes_mm": [(11, 16, 1), (18, 20, -5), (28, 12, -12)], "radius_mm": 1.1},
    ]

    vein_specs = [
        {"name": "internal_jugular_l", "nodes_mm": [(-31, -90, 3), (-30, -58, 3), (-28, -25, 4), (-26, 8, 5), (-25, 35, 6), (-23, 62, 7)], "radius_mm": 4.4},
        {"name": "internal_jugular_r", "nodes_mm": [(31, -90, 3), (30, -58, 3), (28, -25, 4), (26, 8, 5), (25, 35, 6), (23, 62, 7)], "radius_mm": 4.4},
        {"name": "external_jugular_l", "nodes_mm": [(-41, -72, -2), (-42, -40, -1), (-43, -8, 0), (-44, 18, 1)], "radius_mm": 2.0},
        {"name": "external_jugular_r", "nodes_mm": [(41, -72, -2), (42, -40, -1), (43, -8, 0), (44, 18, 1)], "radius_mm": 2.0},
        {"name": "thyroid_venous_plexus", "nodes_mm": [(-20, -53, -1), (-8, -50, -2), (0, -50, -3), (8, -50, -2), (20, -53, -1)], "radius_mm": 1.8},
        {"name": "middle_thyroid_vein_l", "nodes_mm": [(-20, -47, -1), (-26, -44, 1), (-30, -40, 3)], "radius_mm": 1.3},
        {"name": "middle_thyroid_vein_r", "nodes_mm": [(20, -47, -1), (26, -44, 1), (30, -40, 3)], "radius_mm": 1.3},
        {"name": "anterior_jugular_arch", "nodes_mm": [(-12, -70, -2), (-8, -64, -1), (0, -62, -1), (8, -64, -1), (12, -70, -2)], "radius_mm": 1.4},
    ]

    arteries_mask, artery_individual = combine_polylines(x_mm, y_mm, z_mm, artery_specs)
    veins_mask, vein_individual = combine_polylines(x_mm, y_mm, z_mm, vein_specs)
    arteries_mask &= body_mask & ~airway_complex
    veins_mask &= body_mask & ~airway_complex

    body_mask &= ~airway_complex

    coarse_tag_grid = np.full((nx, ny, nz), 0, dtype=np.int16)
    coarse_tag_grid[body_mask] = 10
    coarse_tag_grid[skull_mask | mandible_mask | maxilla | vertebrae_mask] = 20
    coarse_tag_grid[arteries_mask | veins_mask] = 30
    coarse_tag_grid[thyroid_mask | parotid_l | parotid_r | submandibular_l | submandibular_r | parathyroid_mask] = 40
    coarse_tag_grid[brain_mask | blood_brain_barrier | brainstem | spinal_cord] = 50
    coarse_tag_grid[tumor_mask] = 60

    density_grid = np.full((nx, ny, nz), DENSITY_G_CM3["AIR"], dtype=np.float32)
    density_grid[body_mask] = DENSITY_G_CM3["SOFT_TISSUE"]
    density_grid[parotid_l | parotid_r] = DENSITY_G_CM3["PAROTID"]
    density_grid[submandibular_l | submandibular_r] = DENSITY_G_CM3["SUBMANDIBULAR"]
    density_grid[thyroid_mask] = DENSITY_G_CM3["THYROID"]
    density_grid[parathyroid_mask] = DENSITY_G_CM3["PARATHYROIDS"]
    density_grid[brain_mask] = DENSITY_G_CM3["BRAIN"]
    density_grid[blood_brain_barrier] = DENSITY_G_CM3["BLOOD_BRAIN_BARRIER"]
    density_grid[brainstem] = DENSITY_G_CM3["BRAINSTEM"]
    density_grid[spinal_cord] = DENSITY_G_CM3["SPINAL_CORD"]
    density_grid[arteries_mask | veins_mask] = DENSITY_G_CM3["BLOOD"]
    density_grid[tumor_mask] = DENSITY_G_CM3["TUMOUR"]
    density_grid[maxilla | mandible_mask] = DENSITY_G_CM3["MANDIBLE_MAXILLA_BONE"]
    density_grid[skull_mask] = DENSITY_G_CM3["SKULL_CORTICAL_BONE"]
    density_grid[vertebrae_mask] = DENSITY_G_CM3["VERTEBRAL_BONE"]
    density_grid[airway_complex | trachea_mask] = DENSITY_G_CM3["AIR"]

    structures = {
        "BODY": body_mask,
        "SKULL": skull_mask,
        "MANDIBLE": mandible_mask,
        "MAXILLA": maxilla & body_mask,
        "VERTEBRAE": vertebrae_mask & body_mask,
        "AIRWAY": airway_complex,
        "TRACHEA": trachea_mask,
        "LARYNX": larynx & body_mask,
        "ESOPHAGUS": esophagus & body_mask,
        "PAROTID_L": parotid_l,
        "PAROTID_R": parotid_r,
        "SUBMANDIBULAR_L": submandibular_l,
        "SUBMANDIBULAR_R": submandibular_r,
        "SPINAL_CORD": spinal_cord,
        "BRAIN": brain_mask,
        "BLOOD_BRAIN_BARRIER": blood_brain_barrier,
        "BRAINSTEM": brainstem,
        "THYROID": thyroid_mask,
        "PARATHYROIDS": parathyroid_mask,
        "ARTERIES": arteries_mask,
        "VEINS": veins_mask,
        "TUMOUR": tumor_mask,
    }
    structures.update({name.upper(): mask & body_mask for name, mask in artery_individual.items()})
    structures.update({name.upper(): mask & body_mask for name, mask in vein_individual.items()})

    voxel_volume_cc = (dx * dy * dz) / 1000.0
    meta = {
        "grid_shape": [int(nx), int(ny), int(nz)],
        "voxel_size_mm": [dx, dy, dz],
        "size_cm": [float(size_x_cm), float(size_y_cm), float(size_z_cm)],
        "voxel_volume_cc": float(voxel_volume_cc),
        "assigned_densities_g_cm3": DENSITY_G_CM3,
        "coarse_tag_meanings": {
            "0": "air",
            "10": "generic_soft_tissue",
            "20": "bone",
            "30": "vasculature",
            "40": "glandular_endocrine_tissue",
            "50": "central_nervous_system",
            "60": "tumour",
        },
        "structure_volumes_cc": {
            name: float(np.count_nonzero(mask) * voxel_volume_cc)
            for name, mask in structures.items()
            if name in {
                "BODY",
                "THYROID",
                "PARATHYROIDS",
                "PAROTID_L",
                "PAROTID_R",
                "SUBMANDIBULAR_L",
                "SUBMANDIBULAR_R",
                "ARTERIES",
                "VEINS",
                "BRAIN",
                "BLOOD_BRAIN_BARRIER",
                "SPINAL_CORD",
                "BRAINSTEM",
                "TUMOUR",
            }
        },
        "anatomical_note": (
            "Detailed synthetic head-and-neck phantom with intracranial brain and blood-brain "
            "barrier shell, thyroid-parathyroid complex, major carotid/jugular vessels, "
            "vertebral arteries, thyroid vasculature, salivary glands, airway-larynx-trachea "
            "complex, cervical spine, and a bulky right-sided oropharyngeal-nodal tumour surrogate."
        ),
    }
    return {
        "tag_grid": coarse_tag_grid,
        "density_grid_g_cm3": density_grid,
        "structures": structures,
        "axes_mm": {"x": x_mm, "y": y_mm, "z": z_mm},
        "meta": meta,
    }


def build_detailed_plan_phantom(
    *,
    size_x_cm: float = 20.0,
    size_y_cm: float = 26.0,
    size_z_cm: float = 18.0,
    voxel_mm: float = 1.5,
) -> Dict[str, object]:
    """Build the detailed anatomy plus planning structures/material tags."""

    phantom = build_detailed_headneck_phantom(
        size_x_cm=float(size_x_cm),
        size_y_cm=float(size_y_cm),
        size_z_cm=float(size_z_cm),
        voxel_mm=float(voxel_mm),
    )
    structures = dict(phantom["structures"])
    axes_mm = phantom["axes_mm"]
    x_mm = axes_mm["x"]
    y_mm = axes_mm["y"]
    z_mm = axes_mm["z"]

    gtv_mask = structures["TUMOUR"]
    ptv_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(29.0, 31.0, 26.0))
    ptv_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(25.0, 38.0, 22.0))
    ptv_mask = (ptv_primary | ptv_nodal) & structures["BODY"] & ~structures["AIRWAY"]
    hypoxic_primary = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(20.0, 4.0, -4.0), radii_mm=(14.0, 16.0, 13.0))
    hypoxic_nodal = ellipsoid_mask(x_mm, y_mm, z_mm, center_mm=(34.0, -35.0, 1.0), radii_mm=(10.0, 18.0, 10.0))
    hypoxic_mask = (hypoxic_primary | hypoxic_nodal) & gtv_mask

    structures["GTV"] = gtv_mask
    structures["PTV"] = ptv_mask
    structures["HYPOXIA"] = hypoxic_mask
    structures["BONE"] = structures["SKULL"] | structures["MANDIBLE"] | structures["MAXILLA"] | structures["VERTEBRAE"]

    tag_grid = build_material_tag_grid(structures)

    meta = dict(phantom["meta"])
    voxel_volume_cc = float(meta["voxel_volume_cc"])
    meta["structure_volumes_cc"] = dict(meta["structure_volumes_cc"])
    meta["structure_volumes_cc"]["GTV"] = float(np.count_nonzero(gtv_mask) * voxel_volume_cc)
    meta["structure_volumes_cc"]["PTV"] = float(np.count_nonzero(ptv_mask) * voxel_volume_cc)
    meta["structure_volumes_cc"]["MANDIBLE"] = float(np.count_nonzero(structures["MANDIBLE"]) * voxel_volume_cc)
    meta["anatomical_note"] = (
        "Detailed heterogeneous head-and-neck phantom with explicit brain, BBB, thyroid-parathyroid complex, "
        "arterial/venous network, salivary glands, cervical spine, and bulky right-sided oropharyngeal-nodal target."
    )

    return {
        "tag_grid": tag_grid,
        "density_grid_g_cm3": phantom["density_grid_g_cm3"],
        "structures": structures,
        "axes_mm": axes_mm,
        "meta": meta,
    }

"""TOPAS materialization for the detailed head-and-neck phantom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    """One material entry for the ImageCube phantom."""

    tag: int
    name: str
    base_material: str
    density_g_cm3: float
    color: str
    description: str


MATERIAL_SPECS: List[MaterialSpec] = [
    MaterialSpec(0, "HN_AIR", "G4_AIR", 0.0012, "black", "Outside phantom and airway"),
    MaterialSpec(10, "HN_SOFT_TISSUE", "G4_TISSUE_SOFT_ICRP", 1.04, "lightpink", "Generic soft tissue"),
    MaterialSpec(11, "HN_BRAIN", "G4_TISSUE_SOFT_ICRP", 1.04, "skyblue", "Intracranial brain"),
    MaterialSpec(12, "HN_BBB", "G4_TISSUE_SOFT_ICRP", 1.06, "cyan", "Blood-brain barrier shell"),
    MaterialSpec(13, "HN_BRAINSTEM", "G4_TISSUE_SOFT_ICRP", 1.04, "dodgerblue", "Brainstem"),
    MaterialSpec(14, "HN_SPINAL_CORD", "G4_TISSUE_SOFT_ICRP", 1.04, "lightslategray", "Spinal cord"),
    MaterialSpec(20, "HN_SKULL_BONE", "G4_BONE_CORTICAL_ICRP", 1.85, "white", "Skull cortical bone"),
    MaterialSpec(21, "HN_MANDIBLE_MAXILLA_BONE", "G4_BONE_COMPACT_ICRU", 1.80, "lightgray", "Mandible and maxilla bone"),
    MaterialSpec(22, "HN_VERTEBRAL_BONE", "G4_BONE_CORTICAL_ICRP", 1.45, "gray", "Cervical vertebral bone"),
    MaterialSpec(30, "HN_PAROTID", "G4_TISSUE_SOFT_ICRP", 1.03, "green", "Parotid gland"),
    MaterialSpec(31, "HN_SUBMANDIBULAR", "G4_TISSUE_SOFT_ICRP", 1.04, "limegreen", "Submandibular gland"),
    MaterialSpec(32, "HN_THYROID", "G4_TISSUE_SOFT_ICRP", 1.05, "orange", "Thyroid"),
    MaterialSpec(33, "HN_PARATHYROID", "G4_TISSUE_SOFT_ICRP", 1.05, "yellow", "Parathyroids"),
    MaterialSpec(40, "HN_ARTERIAL_BLOOD", "G4_TISSUE_SOFT_ICRP", 1.06, "red", "Arterial blood pool"),
    MaterialSpec(41, "HN_VENOUS_BLOOD", "G4_TISSUE_SOFT_ICRP", 1.06, "blue", "Venous blood pool"),
    MaterialSpec(50, "HN_TUMOUR", "G4_TISSUE_SOFT_ICRP", 1.05, "magenta", "Tumour surrogate"),
]

TAG_TO_SPEC = {spec.tag: spec for spec in MATERIAL_SPECS}


def write_image_cube(tag_grid_xyz: np.ndarray, out_file) -> None:
    """Write an ImageCube binary in TOPAS ordering."""

    np.asarray(tag_grid_xyz, dtype=np.int16).transpose(2, 1, 0).tofile(out_file)


def build_material_tag_grid(structures: Dict[str, np.ndarray]) -> np.ndarray:
    """Map detailed structure masks to explicit TOPAS material tags."""

    shape = structures["BODY"].shape
    grid = np.zeros(shape, dtype=np.int16)

    grid[structures["BODY"]] = 10
    grid[structures["PAROTID_L"] | structures["PAROTID_R"]] = 30
    grid[structures["SUBMANDIBULAR_L"] | structures["SUBMANDIBULAR_R"]] = 31
    grid[structures["THYROID"]] = 32
    grid[structures["PARATHYROIDS"]] = 33
    grid[structures["BRAIN"]] = 11
    grid[structures["BLOOD_BRAIN_BARRIER"]] = 12
    grid[structures["BRAINSTEM"]] = 13
    grid[structures["SPINAL_CORD"]] = 14
    grid[structures["ARTERIES"]] = 40
    grid[structures["VEINS"]] = 41
    grid[structures["TUMOUR"]] = 50
    grid[structures["MAXILLA"] | structures["MANDIBLE"]] = 21
    grid[structures["SKULL"]] = 20
    grid[structures["VERTEBRAE"]] = 22
    grid[structures["AIRWAY"] | structures["TRACHEA"]] = 0

    return grid


def build_density_from_tags(tag_grid: np.ndarray) -> np.ndarray:
    """Convert material tags back to density."""

    density = np.zeros(tag_grid.shape, dtype=np.float32)
    for tag, spec in TAG_TO_SPEC.items():
        density[tag_grid == int(tag)] = float(spec.density_g_cm3)
    return density


def render_materials_include(specs: Sequence[MaterialSpec] | None = None) -> str:
    """Render TOPAS custom material definitions."""

    active_specs = MATERIAL_SPECS if specs is None else specs
    lines: List[str] = ["# Custom heterogeneous materials for the detailed head-and-neck phantom", ""]
    for spec in active_specs:
        lines.extend(
            [
                f's:Ma/{spec.name}/BaseMaterial = "{spec.base_material}"',
                f"d:Ma/{spec.name}/Density = {spec.density_g_cm3:.6f} g/cm3",
                f's:Ma/{spec.name}/DefaultColor = "{spec.color}"',
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"

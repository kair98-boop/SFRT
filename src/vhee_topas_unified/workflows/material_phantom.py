"""Native workflow for materializing the detailed phantom for TOPAS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..phantom import (
    build_density_from_tags,
    build_detailed_plan_phantom,
    render_materials_include,
    write_image_cube,
)
from .common import default_run_root, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="material-phantom",
        description="Build the native TOPAS material phantom bundle.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_run_root("native_detailed_headneck_material_phantom"),
        help="Output root for the materialized phantom bundle.",
    )
    parser.add_argument("--size-x-cm", type=float, default=20.0)
    parser.add_argument("--size-y-cm", type=float, default=26.0)
    parser.add_argument("--size-z-cm", type=float, default=18.0)
    parser.add_argument("--voxel-mm", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    phantom = build_detailed_plan_phantom(
        size_x_cm=float(args.size_x_cm),
        size_y_cm=float(args.size_y_cm),
        size_z_cm=float(args.size_z_cm),
        voxel_mm=float(args.voxel_mm),
    )

    run_root = Path(args.run_root)
    phantom_dir = run_root / "phantom"
    case_dir = run_root / "case"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    tag_grid = phantom["tag_grid"]
    density_grid = build_density_from_tags(tag_grid)

    write_image_cube(tag_grid, case_dir / "patient_material_tags.bin")
    (case_dir / "materials.txt").write_text(render_materials_include(), encoding="utf-8")
    np.savez_compressed(phantom_dir / "patient_material_tags.npz", tag_grid=tag_grid)
    np.savez_compressed(phantom_dir / "patient_material_density_from_tags.npz", density_grid_g_cm3=density_grid)
    write_json(phantom_dir / "material_phantom_summary.json", phantom["meta"])

    print(f"Wrote material phantom bundle to {run_root}")
    print(f"Binary tag grid: {case_dir / 'patient_material_tags.bin'}")
    print(f"Materials include: {case_dir / 'materials.txt'}")
    return 0

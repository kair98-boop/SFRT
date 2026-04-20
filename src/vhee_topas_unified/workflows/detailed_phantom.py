"""Native workflow for generating the detailed head-and-neck phantom."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..phantom import build_detailed_headneck_phantom
from .common import default_run_root, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detailed-phantom",
        description="Generate the native detailed head-and-neck phantom bundle.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_run_root("native_detailed_headneck_phantom"),
        help="Output root for the generated phantom bundle.",
    )
    parser.add_argument("--size-x-cm", type=float, default=20.0)
    parser.add_argument("--size-y-cm", type=float, default=26.0)
    parser.add_argument("--size-z-cm", type=float, default=18.0)
    parser.add_argument("--voxel-mm", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    phantom = build_detailed_headneck_phantom(
        size_x_cm=float(args.size_x_cm),
        size_y_cm=float(args.size_y_cm),
        size_z_cm=float(args.size_z_cm),
        voxel_mm=float(args.voxel_mm),
    )

    run_root = Path(args.run_root)
    phantom_dir = run_root / "phantom"
    phantom_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        phantom_dir / "detailed_headneck_phantom.npz",
        tag_grid=phantom["tag_grid"],
        density_grid_g_cm3=phantom["density_grid_g_cm3"],
    )
    write_json(phantom_dir / "detailed_headneck_summary.json", phantom["meta"])

    print(f"Wrote detailed phantom bundle to {run_root}")
    print(f"Summary: {phantom_dir / 'detailed_headneck_summary.json'}")
    print(f"Grid archive: {phantom_dir / 'detailed_headneck_phantom.npz'}")
    return 0

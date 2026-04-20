"""Native workflows for source-plan preview generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

from ..io import load_spectrum
from ..phantom import build_detailed_plan_phantom, build_simple_headneck_phantom
from ..planning import (
    build_plan_settings_from_phantom_meta,
    build_plan_sources,
    pick_lattice_spots,
    render_source_block,
)
from .common import autodetect_legacy_spectrum_csv, default_run_root, write_json


def _write_sources_csv(path: Path, sources) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_name",
                "trans_x_mm",
                "trans_y_mm",
                "trans_z_mm",
                "rot_x_deg",
                "rot_y_deg",
                "rot_z_deg",
                "cutoff_x_mm",
                "cutoff_y_mm",
                "histories",
            ],
        )
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "source_name": source.name,
                    "trans_x_mm": source.center_mm[0],
                    "trans_y_mm": source.center_mm[1],
                    "trans_z_mm": source.center_mm[2],
                    "rot_x_deg": source.rotation_deg[0],
                    "rot_y_deg": source.rotation_deg[1],
                    "rot_z_deg": source.rotation_deg[2],
                    "cutoff_x_mm": source.cutoff_mm[0],
                    "cutoff_y_mm": source.cutoff_mm[1],
                    "histories": source.histories,
                }
            )


def _build_parser(
    *,
    prog: str,
    description: str,
    default_run_folder: str,
    default_size_x_cm: float,
    default_size_y_cm: float,
    default_size_z_cm: float,
    default_voxel_mm: float,
    default_spot_radius_mm: float,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_run_root(default_run_folder),
        help="Output root for the native plan preview bundle.",
    )
    parser.add_argument("--size-x-cm", type=float, default=default_size_x_cm)
    parser.add_argument("--size-y-cm", type=float, default=default_size_y_cm)
    parser.add_argument("--size-z-cm", type=float, default=default_size_z_cm)
    parser.add_argument("--voxel-mm", type=float, default=default_voxel_mm)
    parser.add_argument("--spot-radius-mm", type=float, default=default_spot_radius_mm)
    parser.add_argument("--base-margin-mm", type=float, default=6.0)
    parser.add_argument("--base-history-fraction", type=float, default=0.42 if prog == "simple-plan-preview" else 0.95)
    parser.add_argument("--histories", type=int, default=1_000_000)
    parser.add_argument("--num-spots", type=int, default=4)
    parser.add_argument(
        "--lattice-spacing-mm",
        nargs=3,
        type=float,
        default=[18.0, 20.0, 18.0],
        help="Nominal lattice spacing in x, y, z for centroid-anchored spot picking.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=autodetect_legacy_spectrum_csv(),
        help="Optional spectrum CSV used to render a TOPAS source block preview.",
    )
    return parser


def _run_plan_preview(
    argv: list[str] | None,
    *,
    phantom_builder: Callable[..., dict],
    prog: str,
    description: str,
    default_run_folder: str,
    default_size_x_cm: float,
    default_size_y_cm: float,
    default_size_z_cm: float,
    default_voxel_mm: float,
    default_spot_radius_mm: float,
) -> int:
    parser = _build_parser(
        prog=prog,
        description=description,
        default_run_folder=default_run_folder,
        default_size_x_cm=default_size_x_cm,
        default_size_y_cm=default_size_y_cm,
        default_size_z_cm=default_size_z_cm,
        default_voxel_mm=default_voxel_mm,
        default_spot_radius_mm=default_spot_radius_mm,
    )
    args = parser.parse_args(argv)

    phantom = phantom_builder(
        size_x_cm=float(args.size_x_cm),
        size_y_cm=float(args.size_y_cm),
        size_z_cm=float(args.size_z_cm),
        voxel_mm=float(args.voxel_mm),
    )
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    gtv_key = "GTV" if "GTV" in structures else "TUMOUR"

    spot_centers_mm = pick_lattice_spots(
        structures[gtv_key],
        axes_mm,
        spacing_mm=tuple(float(v) for v in args.lattice_spacing_mm),
        spot_radius_mm=float(args.spot_radius_mm),
        limit=int(args.num_spots),
    )[: int(args.num_spots)]

    settings = build_plan_settings_from_phantom_meta(
        phantom["meta"],
        spot_radius_mm=float(args.spot_radius_mm),
        base_margin_mm=float(args.base_margin_mm),
        base_history_fraction=float(args.base_history_fraction),
        histories=int(args.histories),
    )
    plan = build_plan_sources(settings, axes_mm, structures["PTV"], spot_centers_mm)

    run_root = Path(args.run_root)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_sources_csv(analysis_dir / "plan_sources.csv", plan["sources"])
    write_json(
        analysis_dir / "plan_summary.json",
        {
            "workflow": prog,
            "phantom_meta": phantom["meta"],
            "num_spots": len(spot_centers_mm),
            "spot_centers_mm": [list(spot) for spot in spot_centers_mm],
            "ptv_centroid_mm": plan["ptv_centroid_mm"],
            "ap_radius_mm": plan["ap_radius_mm"],
            "lateral_radius_mm": plan["lateral_radius_mm"],
            "num_sources": len(plan["sources"]),
        },
    )

    if args.spectrum_csv is not None and Path(args.spectrum_csv).exists():
        energies, weights = load_spectrum(args.spectrum_csv)
        source_block = render_source_block(plan["sources"], energies, weights)
        (analysis_dir / "beam_source_block.txt").write_text(source_block, encoding="utf-8")

    print(f"Wrote native plan preview to {run_root}")
    print(f"Plan summary: {analysis_dir / 'plan_summary.json'}")
    print(f"Source CSV: {analysis_dir / 'plan_sources.csv'}")
    return 0


def main_simple(argv: list[str] | None = None) -> int:
    return _run_plan_preview(
        argv,
        phantom_builder=build_simple_headneck_phantom,
        prog="simple-plan-preview",
        description="Build the native simple-phantom SFRT source-plan preview.",
        default_run_folder="native_simple_plan_preview",
        default_size_x_cm=18.0,
        default_size_y_cm=24.0,
        default_size_z_cm=16.0,
        default_voxel_mm=2.0,
        default_spot_radius_mm=4.0,
    )


def main_detailed(argv: list[str] | None = None) -> int:
    return _run_plan_preview(
        argv,
        phantom_builder=build_detailed_plan_phantom,
        prog="detailed-plan-preview",
        description="Build the native detailed-phantom SFRT source-plan preview.",
        default_run_folder="native_detailed_plan_preview",
        default_size_x_cm=20.0,
        default_size_y_cm=26.0,
        default_size_z_cm=18.0,
        default_voxel_mm=1.5,
        default_spot_radius_mm=8.0,
    )

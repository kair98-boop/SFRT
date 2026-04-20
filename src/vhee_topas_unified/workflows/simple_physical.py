"""Native end-to-end simple physical SFRT workflow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np

from ..io import (
    PHYSICS_PROFILES,
    format_physics_modules,
    has_nonempty_output,
    load_spectrum,
    load_topas_grid,
    render_case_text,
    run_topas_case,
    write_text_with_retries,
)
from ..metrics import compute_dvh, compute_structure_metrics, metrics_table_rows
from ..phantom import AIR_TAG, BONE_TAG, SOFT_TAG, build_simple_headneck_phantom, write_image_cube
from ..planning import (
    build_plan_settings_from_phantom_meta,
    build_plan_sources,
    pick_lattice_spots,
    render_source_block,
)
from .common import default_run_root, package_asset_path, write_json


MATERIAL_TAGS: Dict[int, str] = {
    int(AIR_TAG): "G4_AIR",
    int(SOFT_TAG): "G4_WATER",
    int(BONE_TAG): "G4_BONE_COMPACT_ICRU",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-physical",
        description="Run the native simple voxelized head-and-neck SFRT physical workflow.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=package_asset_path("topas", "headneck_voxel_lattice_template.txt"),
        help="TOPAS ImageCube template.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=package_asset_path("data", "linac_6mv_representative_spectrum.csv"),
        help="Representative 6 MV spectrum as energy_mev,weight CSV.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_run_root("native_simple_physical"),
        help="Output root for the phantom, TOPAS case, and analysis artifacts.",
    )
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas", help="TOPAS executable.")
    parser.add_argument(
        "--g4-data-dir",
        type=str,
        default="/Applications/GEANT4",
        help="Directory containing Geant4 data folders.",
    )
    parser.add_argument(
        "--physics-profile",
        choices=sorted(PHYSICS_PROFILES),
        default="em_opt4_only",
        help="Named TOPAS modular physics profile.",
    )
    parser.add_argument("--histories", type=int, default=1_000_000, help="Total TOPAS histories.")
    parser.add_argument("--threads", type=int, default=8, help="TOPAS threads.")
    parser.add_argument("--seed", type=int, default=33, help="TOPAS RNG seed.")
    parser.add_argument("--size-x-cm", type=float, default=18.0, help="Phantom left-right size.")
    parser.add_argument("--size-y-cm", type=float, default=24.0, help="Phantom superior-inferior size.")
    parser.add_argument("--size-z-cm", type=float, default=16.0, help="Phantom anterior-posterior size.")
    parser.add_argument("--voxel-mm", type=float, default=2.0, help="Isotropic voxel size.")
    parser.add_argument("--spot-radius-mm", type=float, default=4.0, help="Lattice beamlet radius.")
    parser.add_argument("--base-margin-mm", type=float, default=6.0, help="Margin added to broad-field radius.")
    parser.add_argument(
        "--base-history-fraction",
        type=float,
        default=0.42,
        help="Fraction of histories assigned to the broad AP base field.",
    )
    parser.add_argument(
        "--lattice-spacing-mm",
        nargs=3,
        type=float,
        default=[18.0, 20.0, 18.0],
        help="Nominal lattice spacing in x, y, z within the bulky target.",
    )
    parser.add_argument(
        "--prescription-gy",
        type=float,
        default=6.0,
        help="Physical PTV prescription used to normalize the plan to D95.",
    )
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing dose CSV if present.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip phantom/TOPAS generation and analyze dose only.")
    parser.add_argument("--dvh-max-gy", type=float, default=35.0, help="Upper bound for DVH axis.")
    parser.add_argument("--dvh-bins", type=int, default=351, help="Number of DVH dose-axis samples.")
    return parser


def _patient_input_dir(path: Path) -> str:
    value = str(path.parent.resolve())
    if not value.endswith("/"):
        value += "/"
    return value


def _render_case_file(
    args: argparse.Namespace,
    *,
    patient_bin: Path,
    grid_shape,
    voxel_size_mm,
    spectrum_energies,
    spectrum_weights,
    sources,
) -> str:
    world_hlx_cm = max(40.0, 0.5 * float(args.size_x_cm) + 20.0)
    world_hly_cm = max(40.0, 0.5 * float(args.size_y_cm) + 20.0)
    world_hlz_cm = max(40.0, 0.5 * float(args.size_z_cm) + 20.0)
    show_interval = max(1000, int(args.histories) // 10)

    replacements = {
        "__G4_DATA_DIR__": str(Path(args.g4_data_dir).expanduser()),
        "__PHYSICS_MODULES__": format_physics_modules(str(args.physics_profile)),
        "__CUT_GAMMA_MM__": f"{float(args.cut_gamma_mm):.6f}",
        "__CUT_ELECTRON_MM__": f"{float(args.cut_electron_mm):.6f}",
        "__CUT_POSITRON_MM__": f"{float(args.cut_positron_mm):.6f}",
        "__WORLD_HLX_CM__": f"{world_hlx_cm:.6f}",
        "__WORLD_HLY_CM__": f"{world_hly_cm:.6f}",
        "__WORLD_HLZ_CM__": f"{world_hlz_cm:.6f}",
        "__PATIENT_INPUT_DIR__": _patient_input_dir(patient_bin),
        "__PATIENT_INPUT_FILE__": patient_bin.name,
        "__XBINS__": str(int(grid_shape[0])),
        "__YBINS__": str(int(grid_shape[1])),
        "__ZBINS__": str(int(grid_shape[2])),
        "__VOXEL_SIZE_X_MM__": f"{float(voxel_size_mm[0]):.6f}",
        "__VOXEL_SIZE_Y_MM__": f"{float(voxel_size_mm[1]):.6f}",
        "__VOXEL_SIZE_Z_MM__": f"{float(voxel_size_mm[2]):.6f}",
        "__MATERIAL_TAG_COUNT__": str(len(MATERIAL_TAGS)),
        "__MATERIAL_TAG_VALUES__": " ".join(str(int(v)) for v in MATERIAL_TAGS.keys()),
        "__MATERIAL_NAME_VALUES__": " ".join(f'"{name}"' for name in MATERIAL_TAGS.values()),
        "__OUTPUT_STEM__": "dosedata",
        "__SOURCE_BLOCK__": render_source_block(sources, spectrum_energies, spectrum_weights),
        "__N_THREADS__": str(int(args.threads)),
        "__SEED__": str(int(args.seed)),
        "__SHOW_HISTORY_INTERVAL__": str(int(show_interval)),
    }
    return render_case_text(args.template, replacements)


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


def _save_csv_rows(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root.mkdir(parents=True, exist_ok=True)

    case_dir = args.run_root / "case"
    phantom_dir = args.run_root / "phantom"
    analysis_dir = args.run_root / "analysis"
    case_dir.mkdir(parents=True, exist_ok=True)
    phantom_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    phantom = build_simple_headneck_phantom(
        size_x_cm=float(args.size_x_cm),
        size_y_cm=float(args.size_y_cm),
        size_z_cm=float(args.size_z_cm),
        voxel_mm=float(args.voxel_mm),
    )
    tag_grid = phantom["tag_grid"]
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]

    spot_centers_mm = pick_lattice_spots(
        structures["GTV"],
        axes_mm,
        spacing_mm=tuple(float(v) for v in args.lattice_spacing_mm),
        spot_radius_mm=float(args.spot_radius_mm),
        limit=12,
    )
    settings = build_plan_settings_from_phantom_meta(
        phantom_meta,
        spot_radius_mm=float(args.spot_radius_mm),
        base_margin_mm=float(args.base_margin_mm),
        base_history_fraction=float(args.base_history_fraction),
        histories=int(args.histories),
    )
    plan_meta = build_plan_sources(settings, axes_mm, structures["PTV"], spot_centers_mm)
    sources = plan_meta["sources"]

    patient_bin = case_dir / "synthetic_headneck_tags.bin"
    parameter_file = case_dir / "beamline.txt"
    dose_csv = case_dir / "dosedata.csv"
    log_file = case_dir / "topas.log"

    if not args.analyze_only:
        write_image_cube(tag_grid, patient_bin)
        write_json(phantom_dir / "phantom_meta.json", phantom_meta)
        write_json(
            phantom_dir / "lattice_spots.json",
            {
                "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in spot_centers_mm],
                "plan_meta": {
                    "ptv_centroid_mm": plan_meta["ptv_centroid_mm"],
                    "ap_radius_mm": plan_meta["ap_radius_mm"],
                    "lateral_radius_mm": plan_meta["lateral_radius_mm"],
                    "num_sources": len(sources),
                },
            },
        )
        _write_sources_csv(analysis_dir / "plan_sources.csv", sources)

        spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)
        rendered = _render_case_file(
            args,
            patient_bin=patient_bin,
            grid_shape=phantom_meta["grid_shape"],
            voxel_size_mm=phantom_meta["voxel_size_mm"],
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            sources=sources,
        )
        write_text_with_retries(parameter_file, rendered)
        (analysis_dir / "beam_source_block.txt").write_text(
            render_source_block(sources, spectrum_energies, spectrum_weights),
            encoding="utf-8",
        )

        if not (args.skip_existing and has_nonempty_output(dose_csv)):
            run_topas_case(
                topas_bin=args.topas_bin,
                case_dir=case_dir,
                parameter_file=parameter_file,
                g4_data_dir=args.g4_data_dir,
                expected_outputs=["dosedata.csv"],
                log_file=log_file,
                failure_context="TOPAS run failed for the native simple physical workflow.",
            )

    dose_raw, _ = load_topas_grid(dose_csv)
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])
    ptv_raw_metrics = compute_structure_metrics(
        dose_raw,
        structures["PTV"],
        prescription_gy=float(args.prescription_gy),
        voxel_volume_cc=voxel_volume_cc,
    )
    raw_d95 = float(ptv_raw_metrics["d95_gy"])
    if raw_d95 <= 0.0:
        raise RuntimeError("PTV raw D95 is non-positive; cannot normalize the plan.")
    physical_scale_factor = float(args.prescription_gy) / raw_d95
    physical_dose = dose_raw.astype(np.float32) * np.float32(physical_scale_factor)

    metric_config = {
        "PTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "GTV": {"prescription": float(args.prescription_gy), "vxs": [6.0, 10.0]},
        "SPINAL_CORD": {"prescription": None, "vxs": [5.0, 8.0]},
        "BRAINSTEM": {"prescription": None, "vxs": [5.0, 8.0]},
        "PAROTID_R": {"prescription": None, "vxs": [5.0, 10.0]},
        "PAROTID_L": {"prescription": None, "vxs": [5.0, 10.0]},
        "MANDIBLE": {"prescription": None, "vxs": [10.0, 20.0]},
    }
    physical_metrics = {
        name: compute_structure_metrics(
            physical_dose,
            structures[name],
            prescription_gy=config["prescription"],
            voxel_volume_cc=voxel_volume_cc,
            volume_thresholds_gy=config["vxs"],
        )
        for name, config in metric_config.items()
    }
    _save_csv_rows(analysis_dir / "physical_plan_metrics.csv", metrics_table_rows(physical_metrics, "physical"))

    dose_axis_gy = np.linspace(0.0, float(args.dvh_max_gy), int(args.dvh_bins), dtype=np.float32)
    dvh_rows = []
    for structure in metric_config:
        curve = compute_dvh(physical_dose[structures[structure]], dose_axis_gy)
        for dose_gy, volume_pct in zip(dose_axis_gy.tolist(), curve.tolist()):
            dvh_rows.append(
                {
                    "structure": structure,
                    "dose_gy": float(dose_gy),
                    "volume_pct": float(volume_pct),
                }
            )
    _save_csv_rows(analysis_dir / "physical_dvhs.csv", dvh_rows)

    summary = {
        "workflow": "simple-physical",
        "template": str(args.template),
        "spectrum_csv": str(args.spectrum_csv),
        "prescription_gy": float(args.prescription_gy),
        "physical_scale_factor": float(physical_scale_factor),
        "phantom": phantom_meta,
        "plan": {
            "histories": int(args.histories),
            "num_lattice_spots": len(spot_centers_mm),
            "ptv_centroid_mm": plan_meta["ptv_centroid_mm"],
            "ap_radius_mm": plan_meta["ap_radius_mm"],
            "lateral_radius_mm": plan_meta["lateral_radius_mm"],
            "num_sources": len(sources),
        },
        "physical_metrics": physical_metrics,
        "outputs": {
            "dose_csv": str(dose_csv),
            "beamline": str(parameter_file),
            "source_csv": str(analysis_dir / "plan_sources.csv"),
            "metrics_csv": str(analysis_dir / "physical_plan_metrics.csv"),
            "dvh_csv": str(analysis_dir / "physical_dvhs.csv"),
        },
    }
    write_json(analysis_dir / "simple_physical_summary.json", summary)

    print(f"Wrote native simple physical workflow outputs to {args.run_root}")
    print(f"Summary: {analysis_dir / 'simple_physical_summary.json'}")
    print(f"Metrics: {analysis_dir / 'physical_plan_metrics.csv'}")
    return 0

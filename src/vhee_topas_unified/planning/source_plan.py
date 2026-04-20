"""Source-plan construction helpers for voxelized SFRT workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    """One TOPAS beam source specification."""

    name: str
    center_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    cutoff_mm: Tuple[float, float]
    histories: int


@dataclass(frozen=True)
class LatticePlanSettings:
    """Geometry and weighting controls for a direct lattice plan."""

    size_x_cm: float
    size_y_cm: float
    size_z_cm: float
    spot_radius_mm: float
    base_margin_mm: float
    base_history_fraction: float
    histories: int


def build_plan_settings_from_phantom_meta(
    phantom_meta: Mapping[str, object],
    *,
    spot_radius_mm: float,
    base_margin_mm: float,
    base_history_fraction: float,
    histories: int,
) -> LatticePlanSettings:
    """Build a reusable settings object from a phantom metadata dictionary."""

    size_cm = phantom_meta["size_cm"]
    return LatticePlanSettings(
        size_x_cm=float(size_cm[0]),
        size_y_cm=float(size_cm[1]),
        size_z_cm=float(size_cm[2]),
        spot_radius_mm=float(spot_radius_mm),
        base_margin_mm=float(base_margin_mm),
        base_history_fraction=float(base_history_fraction),
        histories=int(histories),
    )


def histories_from_weights(total_histories: int, weights: Sequence[float]) -> List[int]:
    """Distribute a total history count across weighted source entries."""

    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0.0) or float(weights_arr.sum()) <= 0.0:
        raise ValueError("weights must be non-negative and sum to a positive value.")

    weights_arr = weights_arr / float(weights_arr.sum())
    raw = weights_arr * float(total_histories)
    histories = np.floor(raw).astype(int)
    remainder = int(total_histories) - int(histories.sum())
    if remainder > 0:
        order = np.argsort(-(raw - histories))
        histories[order[:remainder]] += 1
    return histories.tolist()


def projected_radius_mm(
    mask: np.ndarray,
    axes_a_mm: np.ndarray,
    axes_b_mm: np.ndarray,
    centroid_a_mm: float,
    centroid_b_mm: float,
    axis_order: Tuple[int, int],
) -> float:
    """Estimate the projected target radius in one view."""

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty.")
    aa = axes_a_mm[coords[:, axis_order[0]]]
    bb = axes_b_mm[coords[:, axis_order[1]]]
    radii = np.sqrt((aa - float(centroid_a_mm)) ** 2 + (bb - float(centroid_b_mm)) ** 2)
    return float(np.percentile(radii, 99.0))


def build_plan_sources(
    settings: LatticePlanSettings,
    axes_mm: Mapping[str, np.ndarray],
    ptv_mask: np.ndarray,
    spot_centers_mm: Sequence[Tuple[float, float, float]],
) -> Dict[str, object]:
    """Build the AP base field plus AP/lateral lattice spot sources."""

    x_mm = np.asarray(axes_mm["x"])
    y_mm = np.asarray(axes_mm["y"])
    z_mm = np.asarray(axes_mm["z"])
    ptv_idx = np.argwhere(ptv_mask)
    centroid_idx = np.round(ptv_idx.mean(axis=0)).astype(int)
    ptv_centroid_mm = (
        float(x_mm[centroid_idx[0]]),
        float(y_mm[centroid_idx[1]]),
        float(z_mm[centroid_idx[2]]),
    )

    ap_radius = projected_radius_mm(
        ptv_mask,
        x_mm,
        y_mm,
        ptv_centroid_mm[0],
        ptv_centroid_mm[1],
        (0, 1),
    ) + float(settings.base_margin_mm)
    lat_radius = projected_radius_mm(
        ptv_mask,
        z_mm,
        y_mm,
        ptv_centroid_mm[2],
        ptv_centroid_mm[1],
        (2, 1),
    ) + float(settings.base_margin_mm)

    source_plane_margin_mm = 5.0
    x_half_mm = 0.5 * float(settings.size_x_cm) * 10.0
    z_half_mm = 0.5 * float(settings.size_z_cm) * 10.0
    ap_source_z = -z_half_mm - source_plane_margin_mm
    left_source_x = -x_half_mm - source_plane_margin_mm
    right_source_x = x_half_mm + source_plane_margin_mm

    source_entries: List[
        Tuple[str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float], float]
    ] = [
        (
            "AP_BASE",
            (ptv_centroid_mm[0], ptv_centroid_mm[1], ap_source_z),
            (0.0, 0.0, 0.0),
            (ap_radius, ap_radius),
            float(settings.base_history_fraction),
        ),
    ]

    spot_fraction = max(0.0, 1.0 - float(settings.base_history_fraction))
    if spot_centers_mm:
        per_spot_weight = spot_fraction / float(len(spot_centers_mm) * 3)
        for spot_idx, (sx, sy, sz) in enumerate(spot_centers_mm, start=1):
            label = f"SPOT{spot_idx:02d}"
            source_entries.extend(
                [
                    (
                        f"AP_{label}",
                        (float(sx), float(sy), ap_source_z),
                        (0.0, 0.0, 0.0),
                        (float(settings.spot_radius_mm), float(settings.spot_radius_mm)),
                        per_spot_weight,
                    ),
                    (
                        f"LATL_{label}",
                        (left_source_x, float(sy), float(sz)),
                        (0.0, 90.0, 0.0),
                        (float(settings.spot_radius_mm), float(settings.spot_radius_mm)),
                        per_spot_weight,
                    ),
                    (
                        f"LATR_{label}",
                        (right_source_x, float(sy), float(sz)),
                        (0.0, -90.0, 0.0),
                        (float(settings.spot_radius_mm), float(settings.spot_radius_mm)),
                        per_spot_weight,
                    ),
                ]
            )

    history_counts = histories_from_weights(int(settings.histories), [row[4] for row in source_entries])
    specs = [
        SourceSpec(
            name=row[0],
            center_mm=(float(row[1][0]), float(row[1][1]), float(row[1][2])),
            rotation_deg=(float(row[2][0]), float(row[2][1]), float(row[2][2])),
            cutoff_mm=(float(row[3][0]), float(row[3][1])),
            histories=int(histories),
        )
        for row, histories in zip(source_entries, history_counts)
    ]
    return {
        "ptv_centroid_mm": [float(v) for v in ptv_centroid_mm],
        "ap_radius_mm": float(ap_radius),
        "lateral_radius_mm": float(lat_radius),
        "sources": specs,
    }


def render_source_block(
    sources: Sequence[SourceSpec],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
) -> str:
    """Render a TOPAS source block for multiple beam entries."""

    lines: List[str] = []
    spectrum_count = len(spectrum_energies)
    spectrum_values = " ".join(f"{float(v):.6f}" for v in spectrum_energies)
    spectrum_weight_values = " ".join(f"{float(v):.8f}" for v in spectrum_weights)

    for spec in sources:
        group_name = f"BeamOrigin_{spec.name}"
        source_name = f"Source_{spec.name}"
        lines.extend(
            [
                f's:Ge/{group_name}/Type = "Group"',
                f's:Ge/{group_name}/Parent = "World"',
                f"d:Ge/{group_name}/TransX = {spec.center_mm[0]:.6f} mm",
                f"d:Ge/{group_name}/TransY = {spec.center_mm[1]:.6f} mm",
                f"d:Ge/{group_name}/TransZ = {spec.center_mm[2]:.6f} mm",
                f"d:Ge/{group_name}/RotX = {spec.rotation_deg[0]:.6f} deg",
                f"d:Ge/{group_name}/RotY = {spec.rotation_deg[1]:.6f} deg",
                f"d:Ge/{group_name}/RotZ = {spec.rotation_deg[2]:.6f} deg",
                f's:So/{source_name}/Type = "Beam"',
                f's:So/{source_name}/Component = "{group_name}"',
                f's:So/{source_name}/BeamParticle = "gamma"',
                f's:So/{source_name}/BeamEnergySpectrumType = "Discrete"',
                f"dv:So/{source_name}/BeamEnergySpectrumValues = {spectrum_count} {spectrum_values} MeV",
                f"uv:So/{source_name}/BeamEnergySpectrumWeights = {spectrum_count} {spectrum_weight_values}",
                f's:So/{source_name}/BeamPositionDistribution = "Flat"',
                f's:So/{source_name}/BeamPositionCutoffShape = "Ellipse"',
                f"d:So/{source_name}/BeamPositionCutoffX = {spec.cutoff_mm[0]:.6f} mm",
                f"d:So/{source_name}/BeamPositionCutoffY = {spec.cutoff_mm[1]:.6f} mm",
                f's:So/{source_name}/BeamAngularDistribution = "None"',
                f"i:So/{source_name}/NumberOfHistoriesInRun = {spec.histories}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def load_legacy_sources(csv_path: str | Path, total_histories: int) -> List[SourceSpec]:
    """Load and renormalize an existing CSV source table from a legacy run."""

    path = Path(csv_path)
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No source rows found in {path}")

    legacy_total = sum(int(float(row["histories"])) for row in rows)
    if legacy_total <= 0:
        raise RuntimeError(f"Legacy source file has non-positive total histories: {path}")

    scaled = (
        np.array([int(float(row["histories"])) for row in rows], dtype=np.float64)
        * (float(total_histories) / float(legacy_total))
    )
    histories = np.floor(scaled).astype(int)
    remainder = int(total_histories) - int(histories.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - histories))
        histories[order[:remainder]] += 1

    return [
        SourceSpec(
            name=row["source_name"],
            center_mm=(float(row["trans_x_mm"]), float(row["trans_y_mm"]), float(row["trans_z_mm"])),
            rotation_deg=(float(row["rot_x_deg"]), float(row["rot_y_deg"]), float(row["rot_z_deg"])),
            cutoff_mm=(float(row["cutoff_x_mm"]), float(row["cutoff_y_mm"])),
            histories=int(hist),
        )
        for row, hist in zip(rows, histories.tolist())
    ]

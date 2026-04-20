"""Lattice-geometry construction and biology-guided spot selection."""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def sphere_fits(mask: np.ndarray, center_idx: Tuple[int, int, int], radius_vox: int) -> bool:
    """Return True when a voxelized sphere fits entirely inside a mask."""

    cx, cy, cz = center_idx
    x0 = max(cx - radius_vox, 0)
    x1 = min(cx + radius_vox + 1, mask.shape[0])
    y0 = max(cy - radius_vox, 0)
    y1 = min(cy + radius_vox + 1, mask.shape[1])
    z0 = max(cz - radius_vox, 0)
    z1 = min(cz + radius_vox + 1, mask.shape[2])
    local = mask[x0:x1, y0:y1, z0:z1]
    gx = np.arange(x0, x1) - cx
    gy = np.arange(y0, y1) - cy
    gz = np.arange(z0, z1) - cz
    sphere = gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2 <= radius_vox**2
    return bool(np.all(local[sphere]))


def pick_lattice_spots(
    gtv_mask: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
    spacing_mm: Sequence[float],
    *,
    spot_radius_mm: float,
    limit: int = 12,
) -> List[Tuple[float, float, float]]:
    """Pick initial lattice vertices from a regular grid anchored at the GTV centroid."""

    x_mm = np.asarray(axes_mm["x"])
    y_mm = np.asarray(axes_mm["y"])
    z_mm = np.asarray(axes_mm["z"])
    gtv_idx = np.argwhere(gtv_mask)
    centroid_idx = np.round(gtv_idx.mean(axis=0)).astype(int)
    centroid_mm = (
        float(x_mm[centroid_idx[0]]),
        float(y_mm[centroid_idx[1]]),
        float(z_mm[centroid_idx[2]]),
    )

    xs = np.arange(centroid_mm[0] - 1.5 * spacing_mm[0], centroid_mm[0] + 1.51 * spacing_mm[0], spacing_mm[0])
    ys = np.arange(centroid_mm[1] - 1.0 * spacing_mm[1], centroid_mm[1] + 1.01 * spacing_mm[1], spacing_mm[1])
    zs = np.arange(centroid_mm[2] - 1.0 * spacing_mm[2], centroid_mm[2] + 1.01 * spacing_mm[2], spacing_mm[2])

    radius_vox = max(1, int(round(float(spot_radius_mm) / float(x_mm[1] - x_mm[0]))))
    candidates: List[Tuple[float, float, float, float]] = []
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                ix = int(np.argmin(np.abs(x_mm - x0)))
                iy = int(np.argmin(np.abs(y_mm - y0)))
                iz = int(np.argmin(np.abs(z_mm - z0)))
                if not gtv_mask[ix, iy, iz]:
                    continue
                if not sphere_fits(gtv_mask, (ix, iy, iz), radius_vox):
                    continue
                dist2 = (
                    (float(x_mm[ix]) - centroid_mm[0]) ** 2
                    + (float(y_mm[iy]) - centroid_mm[1]) ** 2
                    + (float(z_mm[iz]) - centroid_mm[2]) ** 2
                )
                candidates.append((float(x_mm[ix]), float(y_mm[iy]), float(z_mm[iz]), dist2))

    if not candidates:
        raise RuntimeError("Could not place any lattice spots inside the bulky GTV.")

    ordered = sorted(candidates, key=lambda row: row[3])
    unique_spots: List[Tuple[float, float, float]] = []
    for x0, y0, z0, _ in ordered:
        spot = (x0, y0, z0)
        if spot not in unique_spots:
            unique_spots.append(spot)
        if len(unique_spots) >= int(limit):
            break
    return unique_spots


def build_candidate_centers(
    gtv_mask: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
    *,
    spot_radius_mm: float,
    candidate_step_mm: float,
) -> List[Tuple[int, int, int]]:
    """Enumerate anatomy-valid candidate lattice centers on a regular voxel stride."""

    x_mm = np.asarray(axes_mm["x"])
    step_vox = max(1, int(round(float(candidate_step_mm) / float(x_mm[1] - x_mm[0]))))
    radius_vox = max(1, int(round(float(spot_radius_mm) / float(x_mm[1] - x_mm[0]))))

    candidates: List[Tuple[int, int, int]] = []
    for ix in range(0, gtv_mask.shape[0], step_vox):
        for iy in range(0, gtv_mask.shape[1], step_vox):
            for iz in range(0, gtv_mask.shape[2], step_vox):
                if not gtv_mask[ix, iy, iz]:
                    continue
                if not sphere_fits(gtv_mask, (ix, iy, iz), radius_vox):
                    continue
                candidates.append((ix, iy, iz))

    if not candidates:
        raise RuntimeError("No valid lattice candidate centers were found inside the GTV.")
    return candidates


def point_from_index(candidate_idx: Tuple[int, int, int], axes_mm: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert a voxel index to physical millimetre coordinates."""

    return np.array(
        [
            float(axes_mm["x"][candidate_idx[0]]),
            float(axes_mm["y"][candidate_idx[1]]),
            float(axes_mm["z"][candidate_idx[2]]),
        ],
        dtype=np.float32,
    )


def build_structure_points_mm(
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    names: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Extract point clouds for a list of structure masks."""

    points: Dict[str, np.ndarray] = {}
    for name in names:
        coords = np.argwhere(structures[name])
        points[name] = np.column_stack(
            [
                np.asarray(axes_mm["x"])[coords[:, 0]],
                np.asarray(axes_mm["y"])[coords[:, 1]],
                np.asarray(axes_mm["z"])[coords[:, 2]],
            ]
        ).astype(np.float32)
    return points


def min_distance_mm(point_mm: np.ndarray, structure_points_mm: np.ndarray) -> float:
    """Distance from a point to the closest voxel in a structure point cloud."""

    return float(np.min(np.linalg.norm(structure_points_mm - point_mm[None, :], axis=1)))


def build_safe_candidate_centers(
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Mapping[str, np.ndarray],
    structure_points_mm: Mapping[str, np.ndarray],
    *,
    hard_min_dist_cord_mm: float,
    hard_min_dist_brainstem_mm: float,
) -> List[Tuple[int, int, int]]:
    """Apply hard serial-organ avoidance to candidate centers, with staged relaxation."""

    cord_limit = float(hard_min_dist_cord_mm)
    brainstem_limit = float(hard_min_dist_brainstem_mm)
    min_cord = 25.0
    min_brainstem = 30.0

    while True:
        safe_candidates: List[Tuple[int, int, int]] = []
        for cand in candidate_indices:
            point_mm = point_from_index(cand, axes_mm)
            if (
                min_distance_mm(point_mm, structure_points_mm["SPINAL_CORD"]) >= cord_limit
                and min_distance_mm(point_mm, structure_points_mm["BRAINSTEM"]) >= brainstem_limit
            ):
                safe_candidates.append(cand)
        if len(safe_candidates) >= 4:
            return safe_candidates
        if cord_limit <= min_cord and brainstem_limit <= min_brainstem:
            raise RuntimeError("No safe lattice candidate centers remain after anatomy-aware filtering.")
        cord_limit = max(min_cord, cord_limit - 5.0)
        brainstem_limit = max(min_brainstem, brainstem_limit - 5.0)


def compute_vessel_distance_reward(
    candidate_idx: Tuple[int, int, int],
    vessel_coords_mm: np.ndarray,
    axes_mm: Mapping[str, np.ndarray],
) -> float:
    """Reward placements closer to supportive vascular sink regions."""

    point = point_from_index(candidate_idx, axes_mm)
    distances = np.linalg.norm(vessel_coords_mm - point[None, :], axis=1)
    min_distance = float(np.min(distances))
    return 1.0 / (1.0 + min_distance / 10.0)


def score_oar_exceedances(
    effective_metrics: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, float], Dict[str, Tuple[str, float, float, float]]]:
    """Compute OAR penalty weights based on effective-dose exceedance."""

    rules = {
        "SPINAL_CORD": ("d2_gy", 35.0, 1.20),
        "BRAINSTEM": ("d2_gy", 8.0, 0.90),
        "PAROTID_R": ("mean_gy", 20.0, 1.00),
        "PAROTID_L": ("mean_gy", 5.0, 0.25),
        "THYROID": ("mean_gy", 15.0, 0.75),
        "PARATHYROIDS": ("mean_gy", 15.0, 0.55),
        "BRAIN": ("mean_gy", 5.0, 0.70),
        "BLOOD_BRAIN_BARRIER": ("mean_gy", 5.0, 0.45),
        "MANDIBLE": ("mean_gy", 15.0, 0.50),
    }
    weights: Dict[str, float] = {}
    details: Dict[str, Tuple[str, float, float, float]] = {}
    for structure, (metric, threshold, base_weight) in rules.items():
        value = float(effective_metrics[structure][metric])
        exceed = max(0.0, (value - threshold) / threshold)
        weight = float(base_weight * (1.0 + 2.5 * exceed))
        weights[structure] = weight
        details[structure] = (metric, value, threshold, exceed)
    return weights, details


def compute_plan_objective(effective_metrics: Mapping[str, Mapping[str, float]]) -> Tuple[float, Dict[str, float]]:
    """Biology-aware objective balancing target reward against OAR exceedance."""

    penalties = {
        "SPINAL_CORD": ("d2_gy", 35.0, 1.20),
        "BRAINSTEM": ("d2_gy", 8.0, 0.90),
        "PAROTID_R": ("mean_gy", 20.0, 1.00),
        "PAROTID_L": ("mean_gy", 5.0, 0.25),
        "THYROID": ("mean_gy", 15.0, 0.75),
        "PARATHYROIDS": ("mean_gy", 15.0, 0.55),
        "BRAIN": ("mean_gy", 5.0, 0.70),
        "BLOOD_BRAIN_BARRIER": ("mean_gy", 5.0, 0.45),
        "MANDIBLE": ("mean_gy", 15.0, 0.50),
    }
    reward = (
        2.5 * float(effective_metrics["GTV"]["d95_gy"])
        + 2.0 * float(effective_metrics["PTV"]["d95_gy"])
        + 0.5 * float(effective_metrics["GTV"]["d50_gy"])
    )
    penalty_terms: Dict[str, float] = {}
    total_penalty = 0.0
    for structure, (metric, threshold, weight) in penalties.items():
        value = float(effective_metrics[structure][metric])
        exceed = max(0.0, value - threshold)
        penalty = float(weight * exceed)
        penalty_terms[structure] = penalty
        total_penalty += penalty
    return float(reward - total_penalty), penalty_terms


def choose_next_spots(
    *,
    prev_effective_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    axes_mm: Mapping[str, np.ndarray],
    uptake_tensor: np.ndarray,
    candidate_indices: Sequence[Tuple[int, int, int]],
    num_spots: int,
    min_spacing_mm: float,
    target_effective_gy: float,
    prev_selected_mm: Sequence[Tuple[float, float, float]],
    oar_weights: Mapping[str, float],
    structure_points_mm: Mapping[str, np.ndarray],
    vessel_coords_mm: np.ndarray,
    history_counts: Mapping[Tuple[float, float, float], int],
) -> Tuple[List[Tuple[float, float, float]], Dict[str, object]]:
    """Select the next lattice placement from anatomy- and biology-aware scores."""

    tumour_need = np.clip(float(target_effective_gy) - prev_effective_dose, a_min=0.0, a_max=None)
    hypoxia_mask = structures["HYPOXIA"]
    cyto_uptake = uptake_tensor[1]
    prev_selected = [np.array(p, dtype=np.float32) for p in prev_selected_mm]

    distance_weight_map = {
        "SPINAL_CORD": 0.90,
        "BRAINSTEM": 0.70,
        "PAROTID_R": 1.25,
        "PAROTID_L": 0.35,
        "THYROID": 0.95,
        "PARATHYROIDS": 0.55,
        "BRAIN": 0.80,
        "BLOOD_BRAIN_BARRIER": 0.55,
        "MANDIBLE": 0.45,
    }

    scored: List[Tuple[float, Tuple[int, int, int], Dict[str, float | str]]] = []
    for cand in candidate_indices:
        ix, iy, iz = cand
        point_mm = point_from_index(cand, axes_mm)
        need_score = float(min(tumour_need[ix, iy, iz], 8.0))
        hypoxia_bonus = 3.0 if bool(hypoxia_mask[ix, iy, iz]) else 0.0
        sink_bonus = 6.0 * compute_vessel_distance_reward(cand, vessel_coords_mm, axes_mm)
        direct_sink_bonus = 8.0 * float(cyto_uptake[ix, iy, iz])

        distance_score = 0.0
        dominant_structure = ""
        dominant_penalty = -1.0
        for structure, weight in oar_weights.items():
            dist = min_distance_mm(point_mm, structure_points_mm[structure])
            weighted_distance = float(
                weight * distance_weight_map.get(structure, 0.25) * min(dist, 45.0) / 10.0
            )
            distance_score += weighted_distance
            if weight > dominant_penalty:
                dominant_penalty = float(weight)
                dominant_structure = str(structure)

        history_penalty = 0.0
        history_key = tuple(round(float(v), 1) for v in point_mm.tolist())
        history_penalty += 2.0 * float(history_counts.get(history_key, 0))
        for prev in prev_selected:
            if float(np.linalg.norm(point_mm - prev)) < 3.0:
                history_penalty += 1.5

        score = 1.5 * need_score + hypoxia_bonus + sink_bonus + direct_sink_bonus + distance_score - history_penalty
        scored.append(
            (
                float(score),
                cand,
                {
                    "need_score": float(need_score),
                    "hypoxia_bonus": float(hypoxia_bonus),
                    "sink_bonus": float(sink_bonus + direct_sink_bonus),
                    "distance_score": float(distance_score),
                    "history_penalty": float(history_penalty),
                    "dominant_structure": dominant_structure,
                },
            )
        )

    scored.sort(key=lambda row: row[0], reverse=True)
    best_combo: Tuple[int, ...] | None = None
    best_combo_score = -float("inf")
    num_required = int(num_spots)
    combo_space = min(len(scored), 60)
    candidate_subset = scored[:combo_space]
    relax_spacings = [
        float(min_spacing_mm),
        max(14.0, float(min_spacing_mm) * 0.85),
        12.0,
        10.0,
        8.0,
    ]

    for spacing_limit in relax_spacings:
        for combo in itertools.combinations(range(len(candidate_subset)), num_required):
            points = [
                tuple(float(v) for v in point_from_index(candidate_subset[idx][1], axes_mm).tolist())
                for idx in combo
            ]
            valid = True
            pairwise_sum = 0.0
            for a, b in itertools.combinations(range(len(points)), 2):
                dist = math.dist(points[a], points[b])
                if dist < spacing_limit:
                    valid = False
                    break
                pairwise_sum += dist
            if not valid:
                continue
            combo_score = float(sum(candidate_subset[idx][0] for idx in combo) + 0.02 * pairwise_sum)
            if combo_score > best_combo_score:
                best_combo = combo
                best_combo_score = combo_score
        if best_combo is not None:
            break

    if best_combo is None:
        for spacing_limit in relax_spacings[::-1]:
            selected_points: List[Tuple[float, float, float]] = []
            selected_debug: List[Dict[str, object]] = []
            for score, cand, debug in scored:
                point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
                if selected_points and min(math.dist(point, other) for other in selected_points) < float(spacing_limit):
                    continue
                selected_points.append(point)
                selected_debug.append({"center_mm": list(point), "score": float(score), **debug})
                if len(selected_points) >= num_required:
                    return selected_points, {
                        "candidate_rankings": selected_debug[: min(10, len(selected_debug))],
                        "fallback": True,
                    }

        unique_points: List[Tuple[float, float, float]] = []
        unique_debug: List[Dict[str, object]] = []
        for score, cand, debug in scored:
            point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
            if point in unique_points:
                continue
            unique_points.append(point)
            unique_debug.append({"center_mm": list(point), "score": float(score), **debug})
            if len(unique_points) >= num_required:
                return unique_points, {
                    "candidate_rankings": unique_debug[: min(10, len(unique_debug))],
                    "fallback": True,
                    "spacing_violated": True,
                }
        raise RuntimeError("Unable to place the requested number of lattice vertices in the feedback loop.")

    selected: List[Tuple[float, float, float]] = []
    selected_debug: List[Dict[str, object]] = []
    for idx in best_combo:
        score, cand, debug = candidate_subset[idx]
        point_mm = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
        spread_bonus = 0.0 if not selected else 0.02 * min(math.dist(point_mm, other) for other in selected)
        selected.append(point_mm)
        selected_debug.append(
            {
                "center_mm": list(point_mm),
                "score": float(score + spread_bonus),
                "spread_bonus": float(spread_bonus),
                **debug,
            }
        )
    return selected, {"candidate_rankings": selected_debug, "combo_score": float(best_combo_score)}

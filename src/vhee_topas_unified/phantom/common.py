"""Common geometry helpers for synthetic phantoms."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def centered_axis_mm(count: int, spacing_mm: float) -> np.ndarray:
    """Return a centered 1D axis in mm."""

    return (np.arange(int(count), dtype=np.float32) - (int(count) - 1) / 2.0) * float(spacing_mm)


def ellipsoid_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_mm: Tuple[float, float, float],
    radii_mm: Tuple[float, float, float],
) -> np.ndarray:
    """Return a 3D ellipsoid mask."""

    cx, cy, cz = (float(v) for v in center_mm)
    rx, ry, rz = (float(v) for v in radii_mm)
    return (
        ((x_mm[:, None, None] - cx) / rx) ** 2
        + ((y_mm[None, :, None] - cy) / ry) ** 2
        + ((z_mm[None, None, :] - cz) / rz) ** 2
        <= 1.0
    )


def capped_cylinder_along_y_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_x_mm: float,
    center_z_mm: float,
    radius_x_mm: float,
    radius_z_mm: float,
    y_min_mm: float,
    y_max_mm: float,
) -> np.ndarray:
    """Return an elliptical cylinder clipped along the y-axis."""

    radial = (
        ((x_mm[:, None, None] - float(center_x_mm)) / float(radius_x_mm)) ** 2
        + ((z_mm[None, None, :] - float(center_z_mm)) / float(radius_z_mm)) ** 2
    )
    return (radial <= 1.0) & (
        (y_mm[None, :, None] >= float(y_min_mm)) & (y_mm[None, :, None] <= float(y_max_mm))
    )


def cylinder_along_y_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    center_x_mm: float,
    center_z_mm: float,
    radius_mm: float,
    y_min_mm: float,
    y_max_mm: float,
) -> np.ndarray:
    """Return a circular cylinder clipped along the y-axis."""

    radial = (x_mm[:, None, None] - float(center_x_mm)) ** 2 + (z_mm[None, None, :] - float(center_z_mm)) ** 2
    return (radial <= float(radius_mm) ** 2) & (
        (y_mm[None, :, None] >= float(y_min_mm)) & (y_mm[None, :, None] <= float(y_max_mm))
    )


def add_tube_segment(
    mask: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    start_mm: Sequence[float],
    end_mm: Sequence[float],
    radius_mm: float,
) -> None:
    """Rasterize a single tube segment into an existing mask."""

    start = np.asarray(start_mm, dtype=np.float32)
    end = np.asarray(end_mm, dtype=np.float32)
    segment = end - start
    seg_len2 = float(np.dot(segment, segment))
    if seg_len2 <= 0.0:
        return

    margin = float(radius_mm) + max(
        float(x_mm[1] - x_mm[0]),
        float(y_mm[1] - y_mm[0]),
        float(z_mm[1] - z_mm[0]),
    )
    mins = np.minimum(start, end) - margin
    maxs = np.maximum(start, end) + margin
    ix = np.flatnonzero((x_mm >= mins[0]) & (x_mm <= maxs[0]))
    iy = np.flatnonzero((y_mm >= mins[1]) & (y_mm <= maxs[1]))
    iz = np.flatnonzero((z_mm >= mins[2]) & (z_mm <= maxs[2]))
    if ix.size == 0 or iy.size == 0 or iz.size == 0:
        return

    xx = x_mm[ix][:, None, None]
    yy = y_mm[iy][None, :, None]
    zz = z_mm[iz][None, None, :]
    px = xx - start[0]
    py = yy - start[1]
    pz = zz - start[2]
    t = (px * segment[0] + py * segment[1] + pz * segment[2]) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest_x = start[0] + t * segment[0]
    closest_y = start[1] + t * segment[1]
    closest_z = start[2] + t * segment[2]
    dist2 = (xx - closest_x) ** 2 + (yy - closest_y) ** 2 + (zz - closest_z) ** 2
    mask[np.ix_(ix, iy, iz)] |= dist2 <= float(radius_mm) ** 2


def polyline_tube_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    nodes_mm: Sequence[Sequence[float]],
    radius_mm: float,
) -> np.ndarray:
    """Return a union mask for a polyline tube."""

    mask = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    nodes = [tuple(float(v) for v in node) for node in nodes_mm]
    for start, end in zip(nodes[:-1], nodes[1:]):
        add_tube_segment(mask, x_mm, y_mm, z_mm, start, end, float(radius_mm))
    return mask


def combine_polylines(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    specs: Sequence[dict],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Combine multiple named polyline tubes into one union mask and per-name masks."""

    union = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    individual: Dict[str, np.ndarray] = {}
    for spec in specs:
        mask = polyline_tube_mask(
            x_mm,
            y_mm,
            z_mm,
            nodes_mm=spec["nodes_mm"],
            radius_mm=float(spec["radius_mm"]),
        )
        individual[str(spec["name"])] = mask
        union |= mask
    return union, individual


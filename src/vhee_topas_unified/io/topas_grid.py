"""TOPAS CSV grid readers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


AXIS_HEADER_RE = re.compile(
    r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins\s+of\s+([0-9eE+\-.]+)\s+cm\s*$",
    re.IGNORECASE,
)
REPORT_HEADER_RE = re.compile(
    r"^#\s*(.+?)\s*\(\s*([^)]+?)\s*\)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def parse_topas_header(csv_path: str | Path, retries: int = 1, retry_delay_sec: float = 0.0) -> Dict[str, float]:
    """Parse the TOPAS comment header for grid shape, spacing, and report names."""

    path = Path(csv_path)
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay_sec))
    for attempt in range(1, attempts + 1):
        try:
            header: Dict[str, float] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.startswith("#"):
                        break
                    axis_match = AXIS_HEADER_RE.match(line.strip())
                    if axis_match:
                        axis = axis_match.group(1).upper()
                        header[f"n{axis.lower()}"] = int(axis_match.group(2))
                        header[f"d{axis.lower()}_cm"] = float(axis_match.group(3))
                        continue
                    report_match = REPORT_HEADER_RE.match(line.strip())
                    if report_match:
                        quantity = report_match.group(1).strip()
                        unit = report_match.group(2).strip()
                        reports = [value.strip() for value in report_match.group(3).split() if value.strip()]
                        if reports:
                            header["quantity_name"] = quantity
                            header["quantity_unit"] = unit
                            header["report_names"] = reports
            return header
        except (OSError, TimeoutError):
            if attempt >= attempts:
                raise
            time.sleep(delay)
    return {}


def load_topas_grid(
    csv_path: str | Path,
    retries: int = 1,
    retry_delay_sec: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load a single-report TOPAS CSV scorer grid."""

    path = Path(csv_path)
    header = parse_topas_header(path, retries=retries, retry_delay_sec=retry_delay_sec)

    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay_sec))
    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = np.loadtxt(str(path), comments="#", delimiter=",")
            break
        except (OSError, TimeoutError):
            if attempt >= attempts:
                raise
            time.sleep(delay)

    if raw is None:
        raise ValueError(f"Could not read data rows from {path}")
    if raw.size == 0:
        raise ValueError(f"No data rows in {path}")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns (ix,iy,iz,dose) in {path}")

    ix = raw[:, 0].astype(int)
    iy = raw[:, 1].astype(int)
    iz = raw[:, 2].astype(int)
    dose = raw[:, 3].astype(float)

    nx = int(header.get("nx", int(ix.max()) + 1))
    ny = int(header.get("ny", int(iy.max()) + 1))
    nz = int(header.get("nz", int(iz.max()) + 1))
    grid = np.zeros((nx, ny, nz), dtype=np.float64)
    grid[ix, iy, iz] = dose

    if "dx_cm" not in header or "dy_cm" not in header or "dz_cm" not in header:
        raise ValueError(f"Could not read voxel spacing from TOPAS header in {path}")

    return grid, header


def load_topas_report_grids(
    csv_path: str | Path,
    retries: int = 1,
    retry_delay_sec: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Load a TOPAS CSV containing one or more scorer report columns."""

    path = Path(csv_path)
    header = parse_topas_header(path, retries=retries, retry_delay_sec=retry_delay_sec)

    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay_sec))
    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = np.loadtxt(str(path), comments="#", delimiter=",")
            break
        except (OSError, TimeoutError):
            if attempt >= attempts:
                raise
            time.sleep(delay)

    if raw is None:
        raise ValueError(f"Could not read data rows from {path}")
    if raw.size == 0:
        raise ValueError(f"No data rows in {path}")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns (ix,iy,iz,report...) in {path}")

    ix = raw[:, 0].astype(int)
    iy = raw[:, 1].astype(int)
    iz = raw[:, 2].astype(int)
    report_data = raw[:, 3:].astype(float)

    nx = int(header.get("nx", int(ix.max()) + 1))
    ny = int(header.get("ny", int(iy.max()) + 1))
    nz = int(header.get("nz", int(iz.max()) + 1))

    report_names = list(header.get("report_names", []))
    if len(report_names) != report_data.shape[1]:
        report_names = [f"report_{idx}" for idx in range(report_data.shape[1])]

    report_grids: Dict[str, np.ndarray] = {}
    for idx, report_name in enumerate(report_names):
        grid = np.zeros((nx, ny, nz), dtype=np.float64)
        grid[ix, iy, iz] = report_data[:, idx]
        report_grids[str(report_name)] = grid

    if "dx_cm" not in header or "dy_cm" not in header or "dz_cm" not in header:
        raise ValueError(f"Could not read voxel spacing from TOPAS header in {path}")

    return report_grids, header

"""Shared helpers for native unified-package workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def package_root() -> Path:
    """Return the root of the unified workspace."""

    return Path(__file__).resolve().parents[3]


def default_run_root(folder_name: str) -> Path:
    """Return a standard run directory under the caller's current working directory."""

    return Path.cwd() / "runs" / str(folder_name)


def package_asset_path(*relative_parts: str) -> Path:
    """Resolve a packaged asset to a filesystem path."""

    return Path(__file__).resolve().parents[1] / "assets" / Path(*relative_parts)


def write_json(path: str | Path, payload: Any) -> None:
    """Write formatted JSON with parent creation."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def autodetect_legacy_spectrum_csv() -> Path | None:
    """Try to locate the existing representative 6 MV spectrum CSV."""

    candidates = [package_asset_path("data", "linac_6mv_representative_spectrum.csv")]
    root = package_root()
    candidates.append(root.parent / "vhee_topas" / "data" / "linac_6mv_representative_spectrum.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

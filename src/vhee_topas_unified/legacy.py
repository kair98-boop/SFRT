"""Compatibility layer for calling the current legacy workflow scripts.

This module gives the new package a stable CLI surface while the old code is
still being migrated out of the research repo.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class WorkflowSpec:
    """A legacy workflow exposed through the unified CLI."""

    name: str
    script_relpath: str
    description: str


WORKFLOWS: dict[str, WorkflowSpec] = {
    "legacy-simple-physical": WorkflowSpec(
        name="legacy-simple-physical",
        script_relpath="scripts/run_phase13_headneck_voxel_lattice.py",
        description="Run the legacy script for the simple voxelized head-and-neck SFRT workflow.",
    ),
    "detailed-physical": WorkflowSpec(
        name="detailed-physical",
        script_relpath="scripts/run_phase14_detailed_headneck_voxel_lattice.py",
        description="Run the detailed heterogeneous physical planning workflow.",
    ),
    "bioaware": WorkflowSpec(
        name="bioaware",
        script_relpath="scripts/run_phase15_detailed_headneck_bioaware.py",
        description="Apply the calibrated biology model to the detailed plan.",
    ),
    "bioopt": WorkflowSpec(
        name="bioopt",
        script_relpath="scripts/run_phase16_bio_guided_lattice_optimization.py",
        description="Run the biology-guided lattice optimization loop.",
    ),
    "candidate-tradeoff": WorkflowSpec(
        name="candidate-tradeoff",
        script_relpath="scripts/generate_phase16_candidate_plan_tradeoff_report.py",
        description="Compare candidate plans in physical and biological space.",
    ),
    "legacy-detailed-phantom": WorkflowSpec(
        name="legacy-detailed-phantom",
        script_relpath="scripts/generate_detailed_headneck_phantom.py",
        description="Run the legacy script that generates the detailed anatomical phantom.",
    ),
    "legacy-material-phantom": WorkflowSpec(
        name="legacy-material-phantom",
        script_relpath="scripts/generate_detailed_headneck_topas_phantom.py",
        description="Run the legacy script that builds the TOPAS material phantom.",
    ),
}


class LegacyRootError(RuntimeError):
    """Raised when the legacy repo cannot be found."""


def package_root() -> Path:
    """Return the root of the new unified package workspace."""

    return Path(__file__).resolve().parents[2]


def autodetect_legacy_root() -> Path | None:
    """Try to find the current legacy repo without hard-coding a machine path."""

    env_value = os.environ.get("VHEE_TOPAS_LEGACY_ROOT")
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())

    root = package_root()
    candidates.extend(
        [
            root.parent / "vhee_topas",
            root / "legacy" / "vhee_topas",
        ]
    )

    for candidate in candidates:
        if (candidate / "scripts").is_dir() and (candidate / "topas").is_dir():
            return candidate.resolve()
    return None


def resolve_legacy_root(explicit_root: str | Path | None) -> Path:
    """Resolve the legacy repo path or raise a clear error."""

    if explicit_root is not None:
        candidate = Path(explicit_root).expanduser().resolve()
        if (candidate / "scripts").is_dir():
            return candidate
        raise LegacyRootError(
            f"Provided legacy root does not look valid: {candidate}"
        )

    detected = autodetect_legacy_root()
    if detected is not None:
        return detected

    raise LegacyRootError(
        "Could not locate the legacy `vhee_topas` repo. "
        "Use --legacy-root or set VHEE_TOPAS_LEGACY_ROOT."
    )


def list_workflows() -> Iterable[WorkflowSpec]:
    """Return the workflows in a stable CLI order."""

    order = [
        "legacy-simple-physical",
        "detailed-physical",
        "bioaware",
        "bioopt",
        "candidate-tradeoff",
        "legacy-detailed-phantom",
        "legacy-material-phantom",
    ]
    for key in order:
        yield WORKFLOWS[key]


def resolve_workflow(name: str) -> WorkflowSpec:
    """Look up one workflow by name."""

    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(WORKFLOWS))
        raise KeyError(f"Unknown workflow '{name}'. Valid choices: {valid}") from exc


def workflow_script_path(workflow: WorkflowSpec, legacy_root: Path) -> Path:
    """Resolve the script path for a workflow."""

    script_path = legacy_root / workflow.script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Workflow script not found: {script_path}")
    return script_path


def build_legacy_command(
    workflow_name: str,
    *,
    legacy_root: Path,
    python_executable: str | None = None,
    workflow_args: Iterable[str] = (),
) -> list[str]:
    """Build the subprocess command for one legacy workflow."""

    workflow = resolve_workflow(workflow_name)
    script_path = workflow_script_path(workflow, legacy_root)
    python_bin = python_executable or sys.executable
    return [python_bin, str(script_path), *list(workflow_args)]


def run_legacy_workflow(
    workflow_name: str,
    *,
    legacy_root: Path,
    python_executable: str | None = None,
    workflow_args: Iterable[str] = (),
    dry_run: bool = False,
) -> int:
    """Run a legacy workflow through the unified CLI."""

    command = build_legacy_command(
        workflow_name,
        legacy_root=legacy_root,
        python_executable=python_executable,
        workflow_args=workflow_args,
    )
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(legacy_root), check=False)
    return int(completed.returncode)

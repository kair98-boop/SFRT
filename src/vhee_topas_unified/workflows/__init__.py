"""Registry for native package-owned workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .detailed_phantom import main as run_detailed_phantom
from .material_phantom import main as run_material_phantom
from .plan_preview import main_detailed as run_detailed_plan_preview
from .plan_preview import main_simple as run_simple_plan_preview
from .simple_physical import main as run_simple_physical


@dataclass(frozen=True)
class NativeWorkflowSpec:
    """One workflow implemented directly inside the new package."""

    name: str
    description: str
    runner: Callable[[list[str] | None], int]


WORKFLOWS: dict[str, NativeWorkflowSpec] = {
    "simple-physical": NativeWorkflowSpec(
        name="simple-physical",
        description="Run the native simple voxelized head-and-neck SFRT physical workflow.",
        runner=run_simple_physical,
    ),
    "detailed-phantom": NativeWorkflowSpec(
        name="detailed-phantom",
        description="Generate the detailed anatomical phantom using the native package modules.",
        runner=run_detailed_phantom,
    ),
    "material-phantom": NativeWorkflowSpec(
        name="material-phantom",
        description="Build the detailed TOPAS material phantom using the native package modules.",
        runner=run_material_phantom,
    ),
    "simple-plan-preview": NativeWorkflowSpec(
        name="simple-plan-preview",
        description="Build a native simple-phantom SFRT source-plan preview bundle.",
        runner=run_simple_plan_preview,
    ),
    "detailed-plan-preview": NativeWorkflowSpec(
        name="detailed-plan-preview",
        description="Build a native detailed-phantom SFRT source-plan preview bundle.",
        runner=run_detailed_plan_preview,
    ),
}


def list_native_workflows() -> Iterable[NativeWorkflowSpec]:
    """Return native workflows in a stable display order."""

    order = [
        "simple-physical",
        "detailed-phantom",
        "material-phantom",
        "simple-plan-preview",
        "detailed-plan-preview",
    ]
    for key in order:
        yield WORKFLOWS[key]


def is_native_workflow(name: str) -> bool:
    """Return True if the workflow is package-native."""

    return name in WORKFLOWS


def resolve_native_workflow(name: str) -> NativeWorkflowSpec:
    """Resolve one native workflow by name."""

    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(WORKFLOWS))
        raise KeyError(f"Unknown native workflow '{name}'. Valid choices: {valid}") from exc


def run_native_workflow(name: str, workflow_args: list[str] | None = None) -> int:
    """Run one native workflow."""

    spec = resolve_native_workflow(name)
    return int(spec.runner(list(workflow_args or [])))

"""Top-level CLI entrypoint for the unified package."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

from . import __version__
from .legacy import (
    LegacyRootError,
    autodetect_legacy_root,
    build_legacy_command,
    list_workflows as list_legacy_workflows,
    resolve_legacy_root,
    run_legacy_workflow,
)
from .workflows import is_native_workflow, list_native_workflows, run_native_workflow


def _strip_remainder_leading_dash(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _all_workflow_names() -> list[str]:
    native = [workflow.name for workflow in list_native_workflows()]
    legacy = [workflow.name for workflow in list_legacy_workflows()]
    return native + [name for name in legacy if name not in native]


def _iter_workflows_with_backend():
    for workflow in list_native_workflows():
        yield "native", workflow
    for workflow in list_legacy_workflows():
        if workflow.name not in {native.name for native in list_native_workflows()}:
            yield "legacy", workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vhee-topas-unified",
        description=(
            "Unified CLI for SFRT physical planning, biology-aware evaluation, "
            "and optimization workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser(
        "info",
        help="Show package paths, detected legacy repo, and available workflows.",
    )
    info_parser.add_argument(
        "--legacy-root",
        type=Path,
        default=None,
        help="Optional explicit path to the current legacy vhee_topas repo.",
    )

    subparsers.add_parser(
        "list-workflows",
        help="List the workflows currently exposed through the unified CLI.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one native or legacy workflow through the unified CLI surface.",
    )
    run_parser.add_argument(
        "workflow",
        choices=_all_workflow_names(),
        help="Workflow name to run.",
    )
    run_parser.add_argument(
        "--legacy-root",
        type=Path,
        default=None,
        help="Optional explicit path to the current legacy vhee_topas repo.",
    )
    run_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the legacy workflow.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the workflow and print the command without executing it.",
    )
    run_parser.add_argument(
        "--show-command",
        action="store_true",
        help="Print the fully resolved command before running it.",
    )

    return parser


def cmd_info(args: argparse.Namespace) -> int:
    print(f"Unified package root: {Path(__file__).resolve().parents[2]}")
    print(f"CLI module: {Path(__file__).resolve()}")

    detected = autodetect_legacy_root()
    print(f"Auto-detected legacy root: {detected if detected is not None else 'not found'}")
    try:
        resolved = resolve_legacy_root(args.legacy_root)
        print(f"Resolved legacy root: {resolved}")
    except LegacyRootError as exc:
        print(f"Resolved legacy root: unavailable ({exc})")

    print("Available workflows:")
    for backend, workflow in _iter_workflows_with_backend():
        print(f"  - {workflow.name} [{backend}]: {workflow.description}")
    return 0


def cmd_list_workflows() -> int:
    for backend, workflow in _iter_workflows_with_backend():
        print(f"{workflow.name}\t{backend}\t{workflow.description}")
    return 0


def cmd_run(args: argparse.Namespace, workflow_args: list[str]) -> int:
    workflow_args = _strip_remainder_leading_dash(list(workflow_args))
    if is_native_workflow(args.workflow):
        if args.show_command or args.dry_run:
            print("Resolved native workflow:")
            print(f"  {args.workflow} {' '.join(workflow_args)}".rstrip())
        if args.dry_run:
            return 0
        return run_native_workflow(args.workflow, workflow_args)

    try:
        legacy_root = resolve_legacy_root(args.legacy_root)
    except LegacyRootError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    command = build_legacy_command(
        args.workflow,
        legacy_root=legacy_root,
        python_executable=args.python,
        workflow_args=workflow_args,
    )
    if args.show_command or args.dry_run:
        print("Resolved command:")
        print("  " + shlex.join(command))

    if args.dry_run:
        return 0

    return run_legacy_workflow(
        args.workflow,
        legacy_root=legacy_root,
        python_executable=args.python,
        workflow_args=workflow_args,
        dry_run=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.command is None:
        if unknown:
            parser.error(f"Unrecognized arguments: {' '.join(unknown)}")
        parser.print_help()
        return 0
    if args.command == "info":
        if unknown:
            parser.error(f"Unrecognized arguments for info: {' '.join(unknown)}")
        return cmd_info(args)
    if args.command == "list-workflows":
        if unknown:
            parser.error(f"Unrecognized arguments for list-workflows: {' '.join(unknown)}")
        return cmd_list_workflows()
    if args.command == "run":
        return cmd_run(args, unknown)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

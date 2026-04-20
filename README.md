# vhee-topas-unified

`vhee-topas-unified` is a refactored Python package for spatially fractionated
radiotherapy research workflows. It pulls reusable phantom generation, planning,
TOPAS I/O, biology, and metric code out of the older `vhee_topas` research
scripts and exposes them through one installable package and one CLI.

This repository is intended to become the cleaner public home for the model as
the remaining legacy workflows are migrated across.

## Current Status

What is native in this package today:

- core biology utilities
- DVH and structure metrics
- simple and detailed synthetic head-and-neck phantoms
- TOPAS material-tag generation and ImageCube helpers
- spectrum loading, TOPAS CSV readers, and TOPAS case helpers
- lattice/source-plan generation
- biology-guided lattice candidate selection helpers
- native CLI workflows for:
  - `simple-physical`
  - `detailed-phantom`
  - `material-phantom`
  - `simple-plan-preview`
  - `detailed-plan-preview`

What still uses the legacy bridge:

- `detailed-physical`
- `bioaware`
- `bioopt`
- `candidate-tradeoff`

## Installation

```bash
git clone <your-github-url> vhee-topas-unified
cd vhee-topas-unified
python -m pip install -e .
```

The package currently requires:

- Python `>=3.10`
- `numpy`

Optional runtime requirements for physical dose workflows:

- a working TOPAS installation
- Geant4 data available to TOPAS

## Quickstart

Show the CLI surface:

```bash
vhee-topas-unified info
vhee-topas-unified list-workflows
```

Generate the detailed anatomy natively:

```bash
vhee-topas-unified run detailed-phantom
vhee-topas-unified run material-phantom
```

Generate source-plan previews without running TOPAS:

```bash
vhee-topas-unified run simple-plan-preview
vhee-topas-unified run detailed-plan-preview
```

Run the first native physical workflow:

```bash
vhee-topas-unified run simple-physical
```

Use a legacy workflow when needed:

```bash
vhee-topas-unified run detailed-physical --legacy-root /path/to/vhee_topas -- --skip-existing
```

## Repo Layout

```text
src/vhee_topas_unified/
  biology/
  phantom/
  planning/
  metrics/
  io/
  workflows/
  cli.py

tests/
.github/workflows/ci.yml
ARCHITECTURE_PLAN.md
```

## Testing

Run the smoke-test suite locally:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml)
installs the package, compiles the source tree, and runs the same smoke tests.

## Design Notes

- Native workflows use package-managed assets where possible.
- Generated outputs are written under `runs/` and are ignored by Git.
- The legacy bridge remains available so migration can continue without blocking
  current research workflows.
- The longer-term migration map is in [ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md).

## Known Limitations

- Not all end-to-end workflows are native yet.
- The richer biology-aware and optimization paths still depend on the sibling
  legacy repo.
- Physical workflows require a local TOPAS executable and Geant4 data.

## Near-Term Roadmap

1. Migrate `detailed-physical` into the native workflow layer.
2. Migrate `bioaware` into the native workflow layer.
3. Migrate the optimization loop into the native workflow layer.
4. Expand tests around workflow outputs and CLI behavior.

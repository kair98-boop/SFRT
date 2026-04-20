# Architecture Plan

## Why this refactor exists

The current `vhee_topas` codebase already has a real core model, but the
reusable logic is spread across phase-specific scripts. The aim here is to
separate:

- reusable model code
- reusable planning / phantom / metric utilities
- experiment-specific workflow entrypoints
- generated outputs

## Current code map

### Core biology

- `vhee_topas/scripts/bystander_multispecies_pde_solver.py`
  - uptake tensors
  - state modifiers
  - state-dependent emission
  - PDE solve
  - temporal hazard
  - phase-7 survival
  - effective-dose inversion

### Physical and plan utilities

- `vhee_topas/scripts/run_phase13_headneck_voxel_lattice.py`
  - simple head-neck phantom
  - lattice spot picking
  - source generation
  - DVH and structure metrics
  - end-to-end simple workflow

- `vhee_topas/scripts/run_phase14_detailed_headneck_voxel_lattice.py`
  - detailed phantom workflow
  - TOPAS ImageCube case rendering
  - physical dose comparison

### Detailed anatomy and materials

- `vhee_topas/scripts/generate_detailed_headneck_phantom.py`
- `vhee_topas/scripts/generate_detailed_headneck_topas_phantom.py`

### Biology-aware detailed evaluation

- `vhee_topas/scripts/run_phase15_detailed_headneck_bioaware.py`

### Biology-guided optimization

- `vhee_topas/scripts/run_phase16_bio_guided_lattice_optimization.py`

## Target package layout

```text
src/vhee_topas_unified/
  biology/
    pde.py
    sinks.py
    survival.py
    constants.py
  phantom/
    simple_headneck.py
    detailed_headneck.py
    materials.py
  planning/
    lattice.py
    source_plan.py
    optimize.py
  metrics/
    dvh.py
    structures.py
  io/
    topas_case.py
    topas_grid.py
    spectrum.py
  workflows/
    detailed_physical.py
    detailed_bioaware.py
    bioopt.py
  cli.py
```

## Planned module moves

### biology/pde.py

From:

- `vhee_topas/scripts/bystander_multispecies_pde_solver.py`

Move:

- `solve_multispecies_pde_3d`
- `run_pde_temporal_integration`
- related helpers

### biology/sinks.py

From:

- `vhee_topas/scripts/bystander_multispecies_pde_solver.py`
- `vhee_topas/scripts/run_phase15_detailed_headneck_bioaware.py`

Move:

- vessel uptake builders
- anatomical uptake tensor assembly

### biology/survival.py

From:

- `vhee_topas/scripts/bystander_multispecies_pde_solver.py`

Move:

- `calculate_phase7_survival`
- `calculate_systemic_immune_penalty`
- `calculate_effective_dose`

### phantom/detailed_headneck.py

From:

- `vhee_topas/scripts/generate_detailed_headneck_phantom.py`
- `vhee_topas/scripts/run_phase14_detailed_headneck_voxel_lattice.py`

Move:

- detailed anatomy builder
- planning target construction
- hypoxia mask construction

### phantom/materials.py

From:

- `vhee_topas/scripts/generate_detailed_headneck_topas_phantom.py`

Move:

- material specs
- structure-to-material mapping

### planning/source_plan.py

From:

- `vhee_topas/scripts/run_phase13_headneck_voxel_lattice.py`

Move:

- source dataclass
- lattice spot picking
- source block generation

### planning/optimize.py

From:

- `vhee_topas/scripts/run_phase16_bio_guided_lattice_optimization.py`

Move:

- objective function
- candidate selection
- spot update logic

### metrics/dvh.py

From:

- `vhee_topas/scripts/run_phase13_headneck_voxel_lattice.py`

Move:

- `compute_dvh`
- dose-at-volume helpers

### metrics/structures.py

From:

- `vhee_topas/scripts/run_phase13_headneck_voxel_lattice.py`

Move:

- `compute_structure_metrics`

### io/topas_case.py

From:

- `vhee_topas/scripts/run_phase13_headneck_voxel_lattice.py`
- `vhee_topas/scripts/run_phase14_detailed_headneck_voxel_lattice.py`
- `vhee_topas/scripts/run_linac_6mv_polyenergetic_clinical_sfrt.py`

Move:

- write image cube
- render TOPAS case text
- run TOPAS

### io/topas_grid.py

From:

- `vhee_topas/scripts/analyze_topas_outputs.py`

Move:

- `load_topas_grid`

### io/spectrum.py

From:

- `vhee_topas/scripts/run_linac_6mv_polyenergetic_clinical_sfrt.py`

Move:

- `load_spectrum`

## First implementation milestone

Build a minimal importable package that supports this path:

1. load detailed phantom
2. build or load SFRT plan
3. read physical dose
4. build anatomical sink field
5. run biology model
6. compute DVHs and structure metrics

## Migration status

### Completed first extraction

The new package now contains:

- `biology/constants.py`
- `biology/common.py`
- `biology/sinks.py`
- `biology/emission.py`
- `biology/survival.py`
- `biology/pde.py`
- `metrics/dvh.py`
- `metrics/structures.py`
- `phantom/common.py`
- `phantom/simple_headneck.py`
- `phantom/detailed_headneck.py`
- `phantom/materials.py`
- `io/spectrum.py`
- `io/topas_grid.py`
- `io/topas_case.py`
- `planning/lattice.py`
- `planning/source_plan.py`
- `workflows/simple_physical.py`
- `workflows/detailed_phantom.py`
- `workflows/material_phantom.py`
- `workflows/plan_preview.py`

What is already migrated:

- simple head-neck phantom builder
- detailed head-neck phantom builder
- detailed planning-phantom wrapper
- material spec definitions
- structure-to-material tag mapping
- ImageCube writer
- TOPAS materials include rendering
- representative spectrum loader
- TOPAS CSV header parsing
- single-report and multi-report TOPAS grid loading
- TOPAS environment discovery / case-template rendering helpers
- initial lattice spot picking
- direct source-plan generation
- legacy source-table loading
- biology-guided candidate scoring and next-spot selection
- native detailed-phantom workflow
- native material-phantom workflow
- native simple/detailed plan-preview workflows
- native simple physical dose workflow
- dose-at-volume helper
- DVH calculation
- structure summary metrics
- CFL stability helper
- 3D Laplacian helper
- multispecies PDE solve
- multispecies PDE solve with temporal hazard integration
- uptake-field builders
- state modifier builders
- state-dependent emission builder
- effective-dose inversion
- ICD / immune-penalty calculation
- phase-7 survival calculation
- LQ survival helper

What is still bridged to the legacy repo:

- advanced / assay-observable PDE variants not yet ported

This is intentional for the first pass so the public API can stabilize before
the heavier PDE code is copied and tested module-by-module.

## What should stay out of the package

- `runs/`
- one-off figure generators
- manuscript-only formatting scripts
- legacy benchmark-specific plotting that is not part of the reusable model

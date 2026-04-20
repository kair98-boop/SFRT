from __future__ import annotations

import unittest

import numpy as np

from vhee_topas_unified.phantom import build_detailed_plan_phantom, build_simple_headneck_phantom
from vhee_topas_unified.planning import (
    LatticePlanSettings,
    build_candidate_centers,
    build_plan_sources,
    build_safe_candidate_centers,
    build_structure_points_mm,
    choose_next_spots,
    compute_plan_objective,
    pick_lattice_spots,
    render_source_block,
    score_oar_exceedances,
)


class PlanningTests(unittest.TestCase):
    def test_simple_plan_sources_build(self) -> None:
        phantom = build_simple_headneck_phantom()
        spots = pick_lattice_spots(
            phantom["structures"]["GTV"],
            phantom["axes_mm"],
            spacing_mm=(18.0, 20.0, 18.0),
            spot_radius_mm=4.0,
            limit=4,
        )
        settings = LatticePlanSettings(
            size_x_cm=18.0,
            size_y_cm=24.0,
            size_z_cm=16.0,
            spot_radius_mm=4.0,
            base_margin_mm=6.0,
            base_history_fraction=0.42,
            histories=1_000_000,
        )
        plan = build_plan_sources(settings, phantom["axes_mm"], phantom["structures"]["PTV"], spots)
        self.assertEqual(len(plan["sources"]), 13)
        block = render_source_block(plan["sources"][:1], [1.0], [1.0])
        self.assertIn("Source_AP_BASE", block)

    def test_candidate_selection_builds(self) -> None:
        phantom = build_detailed_plan_phantom()
        structures = phantom["structures"]
        axes_mm = phantom["axes_mm"]
        candidates = build_candidate_centers(
            structures["GTV"],
            axes_mm,
            spot_radius_mm=8.0,
            candidate_step_mm=6.0,
        )
        points = build_structure_points_mm(
            structures,
            axes_mm,
            [
                "SPINAL_CORD",
                "BRAINSTEM",
                "PAROTID_R",
                "PAROTID_L",
                "THYROID",
                "PARATHYROIDS",
                "BRAIN",
                "BLOOD_BRAIN_BARRIER",
                "MANDIBLE",
            ],
        )
        safe = build_safe_candidate_centers(
            candidates,
            axes_mm,
            points,
            hard_min_dist_cord_mm=55.0,
            hard_min_dist_brainstem_mm=50.0,
        )
        vessel_coords = build_structure_points_mm(structures, axes_mm, ["ARTERIES", "VEINS"])
        vessel_union = np.vstack([vessel_coords["ARTERIES"], vessel_coords["VEINS"]]).astype(np.float32)
        uptake = np.zeros((2, *structures["GTV"].shape), dtype=np.float32)
        uptake[1, structures["VEINS"]] = 0.9
        uptake[1, structures["ARTERIES"]] = 0.7
        selected, debug = choose_next_spots(
            prev_effective_dose=np.zeros(structures["GTV"].shape, dtype=np.float32),
            structures=structures,
            axes_mm=axes_mm,
            uptake_tensor=uptake,
            candidate_indices=safe,
            num_spots=4,
            min_spacing_mm=18.0,
            target_effective_gy=28.0,
            prev_selected_mm=[],
            oar_weights={
                "SPINAL_CORD": 1.2,
                "BRAINSTEM": 0.9,
                "PAROTID_R": 1.0,
                "PAROTID_L": 0.25,
                "THYROID": 0.75,
                "PARATHYROIDS": 0.55,
                "BRAIN": 0.7,
                "BLOOD_BRAIN_BARRIER": 0.45,
                "MANDIBLE": 0.5,
            },
            structure_points_mm=points,
            vessel_coords_mm=vessel_union,
            history_counts={},
        )
        self.assertEqual(len(selected), 4)
        self.assertIn("candidate_rankings", debug)

    def test_objective_helpers(self) -> None:
        effective_metrics = {
            "GTV": {"d95_gy": 20.0, "d50_gy": 30.0},
            "PTV": {"d95_gy": 18.0},
            "SPINAL_CORD": {"d2_gy": 34.0},
            "BRAINSTEM": {"d2_gy": 7.0},
            "PAROTID_R": {"mean_gy": 19.0},
            "PAROTID_L": {"mean_gy": 4.0},
            "THYROID": {"mean_gy": 14.0},
            "PARATHYROIDS": {"mean_gy": 13.0},
            "BRAIN": {"mean_gy": 4.5},
            "BLOOD_BRAIN_BARRIER": {"mean_gy": 4.5},
            "MANDIBLE": {"mean_gy": 14.0},
        }
        objective, penalties = compute_plan_objective(effective_metrics)
        weights, details = score_oar_exceedances(effective_metrics)
        self.assertGreater(objective, 0.0)
        self.assertIn("SPINAL_CORD", penalties)
        self.assertEqual(weights["SPINAL_CORD"], 1.2)
        self.assertEqual(details["SPINAL_CORD"][0], "d2_gy")


if __name__ == "__main__":
    unittest.main()

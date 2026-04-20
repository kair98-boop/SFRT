from __future__ import annotations

import unittest

from vhee_topas_unified.phantom import (
    build_detailed_headneck_phantom,
    build_detailed_plan_phantom,
    build_simple_headneck_phantom,
)


class PhantomTests(unittest.TestCase):
    def test_simple_phantom_builds(self) -> None:
        phantom = build_simple_headneck_phantom()
        self.assertEqual(phantom["tag_grid"].shape, (90, 120, 80))
        self.assertIn("PTV", phantom["structures"])
        self.assertIn("GTV", phantom["structures"])
        self.assertIn("TUMOUR", phantom["structures"])
        self.assertGreater(int(phantom["structures"]["PTV"].sum()), 0)

    def test_detailed_phantom_builds(self) -> None:
        phantom = build_detailed_headneck_phantom()
        self.assertEqual(phantom["tag_grid"].shape, (133, 173, 120))
        self.assertGreater(int(phantom["structures"]["ARTERIES"].sum()), 0)
        self.assertGreater(int(phantom["structures"]["VEINS"].sum()), 0)
        self.assertGreater(int(phantom["structures"]["BRAIN"].sum()), 0)
        self.assertGreater(int(phantom["structures"]["BLOOD_BRAIN_BARRIER"].sum()), 0)
        self.assertIn("coarse_tag_meanings", phantom["meta"])

    def test_detailed_plan_phantom_adds_targets(self) -> None:
        phantom = build_detailed_plan_phantom()
        self.assertIn("GTV", phantom["structures"])
        self.assertIn("PTV", phantom["structures"])
        self.assertIn("HYPOXIA", phantom["structures"])
        self.assertGreater(int(phantom["structures"]["GTV"].sum()), 0)
        self.assertGreater(int(phantom["structures"]["PTV"].sum()), 0)


if __name__ == "__main__":
    unittest.main()

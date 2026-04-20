from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vhee_topas_unified.io import (
    format_physics_modules,
    load_spectrum,
    load_topas_grid,
    load_topas_report_grids,
    parse_topas_header,
    render_case_text,
)
from vhee_topas_unified.workflows.common import package_asset_path


class IoTests(unittest.TestCase):
    def test_load_packaged_spectrum(self) -> None:
        spectrum_path = package_asset_path("data", "linac_6mv_representative_spectrum.csv")
        energies, weights = load_spectrum(spectrum_path)
        self.assertEqual(len(energies), 16)
        self.assertAlmostEqual(sum(weights), 1.0, places=7)

    def test_parse_and_load_single_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.csv"
            path.write_text(
                "# X in 2 bins of 0.10 cm\n"
                "# Y in 2 bins of 0.20 cm\n"
                "# Z in 2 bins of 0.30 cm\n"
                "0,0,0,1.5\n"
                "1,1,1,2.5\n",
                encoding="utf-8",
            )
            header = parse_topas_header(path)
            grid, loaded_header = load_topas_grid(path)
            self.assertEqual(header["nx"], 2)
            self.assertEqual(grid.shape, (2, 2, 2))
            self.assertEqual(float(grid[0, 0, 0]), 1.5)
            self.assertEqual(float(grid[1, 1, 1]), 2.5)
            self.assertEqual(loaded_header["dz_cm"], 0.3)

    def test_load_multi_report_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.csv"
            path.write_text(
                "# X in 2 bins of 0.10 cm\n"
                "# Y in 2 bins of 0.20 cm\n"
                "# Z in 2 bins of 0.30 cm\n"
                "# DoseToMedium ( Gy ) : Sum Mean Standard_Deviation\n"
                "0,0,0,1.0,0.5,0.1\n"
                "1,1,1,2.0,1.5,0.2\n",
                encoding="utf-8",
            )
            report_grids, report_header = load_topas_report_grids(path)
            self.assertIn("Standard_Deviation", report_grids)
            self.assertEqual(float(report_grids["Standard_Deviation"][1, 1, 1]), 0.2)
            self.assertEqual(report_header["quantity_unit"], "Gy")

    def test_render_case_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "template.txt"
            path.write_text("A=__A__\nB=__B__\n", encoding="utf-8")
            rendered = render_case_text(path, {"__A__": "1", "__B__": "two"})
            self.assertEqual(rendered, "A=1\nB=two\n")
        self.assertEqual(format_physics_modules("em_opt4_only"), '1 "g4em-standard_opt4"')


if __name__ == "__main__":
    unittest.main()

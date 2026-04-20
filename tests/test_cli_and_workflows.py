from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from vhee_topas_unified import cli


class CliAndWorkflowTests(unittest.TestCase):
    def test_list_workflows_contains_native_entries(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            rc = cli.main(["list-workflows"])
        output = stream.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("simple-physical\tnative", output)
        self.assertIn("detailed-phantom\tnative", output)

    def test_native_workflows_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            self.assertEqual(
                cli.main(["run", "detailed-phantom", "--", "--run-root", str(root / "detailed")]),
                0,
            )
            self.assertTrue((root / "detailed" / "phantom" / "detailed_headneck_summary.json").exists())

            self.assertEqual(
                cli.main(["run", "material-phantom", "--", "--run-root", str(root / "material")]),
                0,
            )
            self.assertTrue((root / "material" / "case" / "materials.txt").exists())

            self.assertEqual(
                cli.main(["run", "simple-plan-preview", "--", "--run-root", str(root / "simple-preview")]),
                0,
            )
            summary_path = root / "simple-preview" / "analysis" / "plan_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text())
            self.assertEqual(payload["workflow"], "simple-plan-preview")

            self.assertEqual(
                cli.main(["run", "detailed-plan-preview", "--", "--run-root", str(root / "detailed-preview")]),
                0,
            )
            self.assertTrue((root / "detailed-preview" / "analysis" / "plan_sources.csv").exists())


if __name__ == "__main__":
    unittest.main()

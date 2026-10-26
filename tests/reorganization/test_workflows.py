import unittest
from pathlib import Path

from data_factory.workflow import dataset_command
from execution.native import canonical_command
from research.ardy.qualification.run_pipeline import _resolve_calibration as resolve_qualification_calibration
from research.ardy.runtime_support.run_pipeline import _resolve_calibration as resolve_runtime_calibration
from research.ardy.workflow import build_commands as ardy_commands
from research.kimodo.workflow import build_commands as kimodo_commands
from va_research.registry import get_family


ROOT = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def test_kimodo_commands_are_existing_entry_points(self):
        commands = kimodo_commands(project_root=ROOT, prompt="grasp", output=ROOT / "tmp/run", seed=1, case="center")
        for command in commands.values():
            self.assertTrue(any(Path(value).is_file() and value.endswith(".py") for value in command))
        self.assertIn("--task", commands["generation"])

    def test_ardy_commands_are_existing_entry_points(self):
        commands = ardy_commands(project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run", seed=1, case="center", gpu=0)
        self.assertTrue(Path(commands["full_pipeline"][2]).is_file())
        self.assertEqual(commands["full_pipeline"][0].split("/")[-1], "python3")

    def test_opposition_centering_is_explicitly_forwarded(self):
        kimodo = kimodo_commands(
            project_root=ROOT, prompt="grasp", output=ROOT / "tmp/run",
            seed=1, case="right", opposition_centering=True)
        self.assertIn("--opposition-centering", kimodo["export"])
        default_kimodo = kimodo_commands(
            project_root=ROOT, prompt="grasp", output=ROOT / "tmp/run",
            seed=1, case="right")
        self.assertNotIn("--opposition-centering", default_kimodo["export"])

        ardy = ardy_commands(
            project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run",
            seed=1, case="right", gpu=0, opposition_centering=True)
        self.assertIn("--opposition-centering", ardy["full_pipeline"])
        default_ardy = ardy_commands(
            project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run",
            seed=1, case="right", gpu=0)
        self.assertNotIn("--opposition-centering", default_ardy["full_pipeline"])

    def test_collision_hand_projection_is_explicitly_forwarded(self):
        kimodo = kimodo_commands(
            project_root=ROOT, prompt="grasp", output=ROOT / "tmp/run",
            seed=1, case="right", collision_aware_hand=True)
        self.assertIn("--collision-aware-hand", kimodo["export"])
        default_kimodo = kimodo_commands(
            project_root=ROOT, prompt="grasp", output=ROOT / "tmp/run",
            seed=1, case="right")
        self.assertNotIn("--collision-aware-hand", default_kimodo["export"])

        ardy = ardy_commands(
            project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run",
            seed=1, case="right", gpu=0, collision_aware_hand=True)
        self.assertIn("--collision-aware-hand", ardy["full_pipeline"])
        default_ardy = ardy_commands(
            project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run",
            seed=1, case="right", gpu=0)
        self.assertNotIn("--collision-aware-hand", default_ardy["full_pipeline"])

    def test_ardy_eef_lift_offset_is_explicitly_forwarded_and_bounded(self):
        ardy = ardy_commands(
            project_root=ROOT, config=ROOT / "config.yaml", output=ROOT / "tmp/run",
            seed=1, case="left", gpu=7,
            eef_lift_offset_m=(0.025, -0.015, 0.05),
            eef_lift_offset_window_s=(10.0, 11.0))
        command = ardy["full_pipeline"]
        self.assertIn("--eef-lift-offset-m", command)
        start = command.index("--eef-lift-offset-m")
        self.assertEqual(command[start + 1:start + 4], ["0.025", "-0.015", "0.05"])
        self.assertIn("--eef-lift-offset-window-s", command)
        with self.assertRaises(ValueError):
            ardy_commands(project_root=ROOT, config=ROOT / "config.yaml",
                          output=ROOT / "tmp/run", seed=1, case="left", gpu=7,
                          eef_lift_offset_m=(0.1, 0.1, 0.1))

    def test_ardy_pipeline_forwards_configured_calibration_and_cli_wins(self):
        config = ROOT / "tmp" / "ardy" / "config.json"
        cfg = {"adapter": {"calibration": "calibration.json"}}
        expected = (config.parent / "calibration.json").resolve()
        for resolve in (resolve_qualification_calibration, resolve_runtime_calibration):
            self.assertEqual(resolve(None, cfg, config), expected)
            explicit = ROOT / "tmp" / "explicit-calibration.json"
            self.assertEqual(resolve(explicit, cfg, config), explicit.resolve())
            self.assertIsNone(resolve(None, {"adapter": {"calibration": None}}, config))

    def test_execution_command_is_native_only(self):
        command = canonical_command(ROOT / "reference", ROOT / "output")
        self.assertIn("--backend", command)
        self.assertIn("native", command)

    def test_dataset_command_requires_seed_for_count_split(self):
        with self.assertRaises(ValueError):
            dataset_command(ROOT / "canonical", ROOT / "dataset", val_count=1)

    def test_registry_preserves_v4_contract(self):
        family = get_family("v4")
        self.assertIn("68", family.input_contract)
        self.assertIn("10x64", family.output_contract)

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Host inspection Python intentionally lacks scientific deps.
    raise unittest.SkipTest("contract tensor tests require the validated scientific environment")

from contracts import CanonicalReference, ReferenceValidationError, RunManifest
from contracts.policy import PolicyPlan


def reference():
    t = np.arange(7, dtype=np.float64) / 50
    return CanonicalReference(
        time_s=t,
        body_ref_q=np.zeros((7, 29)),
        body_ref_dq=np.zeros((7, 29)),
        root_ref_pos=np.zeros((7, 3)),
        root_ref_quat=np.tile([1.0, 0.0, 0.0, 0.0], (7, 1)),
        hand_ref_q=np.zeros((7, 14)),
        metadata={"reference_hz": 50, "quaternion_convention": "wxyz", "producer_type": "kimodo"},
    )


class ContractTests(unittest.TestCase):
    def test_reference_contract_accepts_canonical_shapes(self):
        item = reference()
        self.assertIs(item.validate(), item)

    def test_reference_contract_rejects_wrong_frequency(self):
        item = reference()
        item.metadata["reference_hz"] = 25
        with self.assertRaises(ReferenceValidationError):
            item.validate()

    def test_policy_plan_contract(self):
        item = PolicyPlan(np.zeros((10, 64)), np.zeros((50, 14)), 5)
        self.assertIs(item.validate(), item)
        with self.assertRaises(ValueError):
            PolicyPlan(np.zeros((9, 64)), np.zeros((50, 14)), 5).validate()

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            original = RunManifest("run_001", "kimodo", command=["echo", "ok"])
            original.write(path)
            self.assertEqual(RunManifest.read(path).run_id, "run_001")

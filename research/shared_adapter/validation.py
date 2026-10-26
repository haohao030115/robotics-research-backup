"""One physical/dataset acceptance policy for every producer."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class ValidationResult:
    physical_success: bool
    contact_safe: bool
    dataset_valid: bool
    expert_valid: bool
    reasons: tuple[str, ...] = ()
    def as_dict(self): return asdict(self)

class UnifiedValidationPolicy:
    VERSION = "strict-unified-validation-v1"
    def __init__(self, *, lift_m=.10, hold_s=2.0, contact_force_noise_n=1e-4):
        self.lift_m=float(lift_m); self.hold_s=float(hold_s); self.contact_force_noise_n=float(contact_force_noise_n)

    def evaluate(self, *, physical_success, contacts=(), dataset_checks=None, replay=None):
        reasons=[]; allowed={"hand_cube", "cube_table", "foot_support"}
        meaningful=[]
        for c in contacts or ():
            force=float(c.get("force_n", 0.)) if isinstance(c,dict) else 0.
            kind=c.get("kind") if isinstance(c,dict) else None
            if force <= self.contact_force_noise_n: continue
            meaningful.append(kind)
            if kind not in allowed: reasons.append("disallowed_contact:"+str(kind))
        contact_safe=not reasons
        checks=dataset_checks or {}; dataset_valid=all(bool(v) for v in checks.values()) if checks else True
        if not dataset_valid: reasons.extend("dataset:"+k for k,v in checks.items() if not v)
        if replay and not replay.get("complete", True): reasons.append("incomplete_replay")
        expert=bool(physical_success and contact_safe and dataset_valid)
        return ValidationResult(bool(physical_success), contact_safe, dataset_valid, expert, tuple(dict.fromkeys(reasons)))

    def evaluate_native_result(self, result: dict, *, dataset_checks=None):
        """Apply the same gate to Kimodo/ARDY native replay summaries.

        Both native evaluators expose ``support_contact_steps`` and the same
        lift/hold fields, although their JSON uses different diagnostic names.
        A positive support or robot/environment-contact count is always
        rejected; no producer-specific exception is made for a hand/table or
        robot-self contact.
        """
        # Keep the historical support/table aliases, but do not let a present
        # ``None`` value mask the other alias.  Native replay summaries expose
        # self/environment contact counts separately; all of them are strict
        # collision gates even when physical_success was computed elsewhere.
        # Kimodo/ARDY place these diagnostics in
        # ``original_shared_evaluator_report`` rather than duplicating every
        # count at the top level, so inspect both representations.
        reports = [result]
        shared = result.get("original_shared_evaluator_report")
        if isinstance(shared, dict):
            reports.append(shared)

        def positive_count(*names):
            for report in reports:
                for name in names:
                    if name not in report or report[name] is None:
                        continue
                    try:
                        if int(report[name]) > 0:
                            return True
                    except (TypeError, ValueError):
                        continue
            return False

        contacts = []
        if positive_count("support_contact_steps", "robot_table_contact_steps"):
            contacts.append({"kind": "robot_table", "force_n": 1.0})
        if positive_count("robot_self_contact_steps"):
            contacts.append({"kind": "robot_self", "force_n": 1.0})
        if positive_count("left_arm_environment_contact_steps"):
            contacts.append({"kind": "left_arm_environment", "force_n": 1.0})
        if positive_count("inactive_arm_environment_contact_steps"):
            contacts.append({"kind": "inactive_arm_environment", "force_n": 1.0})
        complete = bool(result.get("completed", result.get("frame_coverage_complete", True)))
        checks = dict(dataset_checks or {})
        if result.get("frame_coverage_complete") is False: checks["frame_coverage"] = False
        if result.get("finite") is False: checks["finite"] = False
        # These fields are emitted by the native validators and are explicit
        # evidence gates.  Preserve compatibility with older summaries by
        # checking only when a field is present; a present failure must never
        # be hidden by a physical_success flag computed upstream.
        if result.get("integrity_pass") is False:
            checks["integrity_pass"] = False
        if result.get("encoder_mode_0_verified") is False:
            checks["encoder_mode_0_verified"] = False
        if result.get("foreign_processes"):
            checks["foreign_processes_empty"] = False
        for field in ("policy_reference_matches", "policy_target_verified",
                      "target_verified", "hand_target_verified"):
            if result.get(field) is False:
                checks[field] = False
        return self.evaluate(physical_success=bool(result.get("physical_success", False)),
                             contacts=contacts, dataset_checks=checks,
                             replay={"complete": complete})

    def write(self, result, path, *, metadata=None):
        from pathlib import Path
        Path(path).write_text(json.dumps({"policy_version":self.VERSION,"policy":self.__dict__,
            "result":result.as_dict(),"metadata":metadata or {}}, indent=2, sort_keys=True)+'\n')

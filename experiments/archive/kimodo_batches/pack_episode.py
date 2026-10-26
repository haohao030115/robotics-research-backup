"""Package a verified Kimodo replay in Expert Reference Interface v1."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


FIELDS = {
    "time_s": (810,),
    "body_ref_q": (810, 29),
    "body_ref_dq": (810, 29),
    "root_ref_pos": (810, 3),
    "root_ref_quat": (810, 4),
    "hand_ref_q": (810, 14),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_arrays(arrays):
    if set(arrays) != set(FIELDS):
        raise ValueError("Expert reference fields differ from v1 interface")
    for key, shape in FIELDS.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}: shape or nonfinite values")
    time = arrays["time_s"]
    if not np.array_equal(time, np.arange(810) / 50):
        raise ValueError("Time axis must cover 810 frames at 50 Hz")
    if not np.allclose(np.linalg.norm(arrays["root_ref_quat"], axis=1), 1, atol=1e-5):
        raise ValueError("Root quaternions are not normalized")


def joint_limits(reference, setting):
    from export_kimodo_reference import canonical

    model, _, body_names, hand_names = canonical(setting)
    if setting["body_joint_order"] != body_names or setting["hand_joint_order"] != hand_names:
        raise ValueError("Joint names differ from canonical replay order")
    for field, names in (("body_ref_q", body_names), ("hand_ref_q", hand_names)):
        values = reference[field]
        for col, name in enumerate(names):
            joint = model.joint(name)
            if joint.limited[0] and not np.all(
                (values[:, col] >= joint.range[0] - 1e-6)
                & (values[:, col] <= joint.range[1] + 1e-6)
            ):
                raise ValueError(f"Joint limit exceeded: {name}")
    return body_names, hand_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()
    trial = args.trial.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite {out}")
    q = trial / "qualification"
    replay = q / "replay"
    result = json.loads((replay / "validation_task_criteria_verified.json").read_text())
    if not result["physical_success"] or not result["expert_valid"]:
        raise ValueError("Only successful canonical replays become expert references")
    if result["checks_passed"] != result["checks_total"]:
        raise ValueError("Canonical validation checks are incomplete")
    original = json.loads((replay / "result.json").read_text())
    if original["recorded_frames"] != 810 or not original["completed"]:
        raise ValueError("Physical replay did not cover the whole reference")
    setting = json.loads((q / "candidate/setting.json").read_text())
    generation = json.loads((q / "generation/generation.json").read_text())
    config = json.loads((trial / "config/frozen_config.json").read_text())
    if generation["seed"] != config["seed"] or setting["seed"] != config["seed"]:
        raise ValueError("Seed linkage is inconsistent")
    if not generation["model_generation_run"] or generation["model_call_count"] != 5:
        raise ValueError("This is not a fresh five-segment Kimodo generation")
    with np.load(q / "candidate/reference.npz", allow_pickle=False) as source:
        arrays = {
            "time_s": source["timestamps"].copy(),
            "body_ref_q": source["body_ref_q"].copy(),
            "body_ref_dq": source["body_ref_dq"].copy(),
            "root_ref_pos": source["root_ref_pos"].copy(),
            "root_ref_quat": source["root_ref_quat"].copy(),
            "hand_ref_q": source["hand_ref_q"].copy(),
        }
    verify_arrays(arrays)
    with np.load(replay / "input_reference.npz", allow_pickle=False) as played:
        for field, value in arrays.items():
            key = "timestamps" if field == "time_s" else field
            if not np.array_equal(value, played[key]):
                raise ValueError(f"Delivered {field} differs from physically replayed reference")
    body_names, hand_names = joint_limits(arrays, setting)
    metadata = {
        "schema_name": "kimodo_expert_reference",
        "schema_version": 1,
        "episode_id": args.episode_id,
        "reference_hz": 50,
        "num_frames": 810,
        "duration_s": 16.18,
        "body_dof": 29,
        "hand_dof": 14,
        "body_joint_names": body_names,
        "hand_joint_names": hand_names,
        "root_quat_order": "wxyz",
        "units": {
            "time": "s",
            "position": "m",
            "joint_position": "rad",
            "joint_velocity": "rad/s",
        },
        "robot": "G1",
        "scene": "canonical_tabletop",
        "initial_pose": "canonical_reset",
        "cube_initial_pos": json.loads((q / "task/task.json").read_text())[
            "cube_initial_position"
        ],
        "reference_complete": True,
    }
    adaptation = json.loads((q / "candidate/export_metadata.json").read_text())
    calibration = json.loads((trial / "config/development_calibration.json").read_text())
    provenance = {
        "generator": "Kimodo-G1-RP-v1",
        "generator_checkpoint": generation["checkpoint_hashes"],
        "seed": config["seed"],
        "prompts": generation["segment_prompts"],
        "raw_generation_id": generation["task_id"],
        "raw_generation_path_or_hash": generation["raw_sha256"],
        "raw_generation_path": str(q / "generation/raw_motion.npz"),
        "raw_generation_sha256": sha(q / "generation/raw_motion.npz"),
        "offline_processing": {
            "retarget": True,
            "offline_ik": True,
            "wrist_clearance": config["adapter"]["clearance_m"],
            "clearance_decay_s": config["adapter"]["clearance_decay_s"],
            "approach_refinement": True,
            "contact_refinement": True,
            "tracking_calibration": calibration,
            "hand_strategy": config["adapter"]["hand_preclose"],
            "body_ref_dq_source": "derived_by_finite_difference",
            "adaptation_measurements": adaptation.get("postprocessing"),
        },
        "canonical_replay_validation_path": str(replay / "validation_task_criteria_verified.json"),
        "canonical_replay_validation_sha256": sha(
            replay / "validation_task_criteria_verified.json"
        ),
        "classification": "fixed_adapter_batch_experiment",
        "formal_task9_independent_model_comparison": False,
        "shared_generic_text_encoder_authorized": True,
    }
    validation = {
        "reference_schema_valid": True,
        "finite": True,
        "timestamps_monotonic": True,
        "joint_limits_valid": True,
        "quaternion_valid": True,
        "reference_complete": True,
        "tested_with_canonical_replay": True,
        "physical_success": True,
        "expert_valid": True,
        "cube_table_contact_during_qualifying_hold_steps": result[
            "cube_table_contact_during_best_hold_steps"
        ],
        "max_lift_m": result["max_lift_m"],
        "max_continuous_hold_seconds": result["max_continuous_hold_seconds"],
        "canonical_checks_passed": result["checks_passed"],
        "canonical_checks_total": result["checks_total"],
        "validation_notes": "Final body and hand arrays match the physically replayed input exactly.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending_episode_", dir=out.parent))
    try:
        np.savez_compressed(pending / "reference.npz", **arrays)
        dump(pending / "metadata.json", metadata)
        dump(pending / "provenance.json", provenance)
        dump(pending / "reference_validation.json", validation)
        os.rename(pending, out)
    except BaseException:
        for child in pending.iterdir():
            child.unlink()
        pending.rmdir()
        raise
    print(json.dumps({"episode_id": args.episode_id, "path": str(out), "success": True}))


if __name__ == "__main__":
    main()

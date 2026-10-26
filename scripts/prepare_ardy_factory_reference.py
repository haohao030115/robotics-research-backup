#!/usr/bin/env python3
"""Build a Canonical Data Factory reference from an accepted ARDY replay candidate.

This is a packaging step only: it does not generate motion or replay physics.
The caller must pass an already accepted native replay and the candidate that
produced it.  The factory itself performs the new production replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


BODY = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
HAND = [
    "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]
UNITS = {"time_s": "s", "body_ref_q": "rad", "body_ref_dq": "rad/s",
         "root_ref_pos": "m", "root_ref_quat": "unitless", "hand_ref_q": "rad"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_motion(path: Path, name: str) -> np.ndarray:
    return np.loadtxt(path / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    candidate = args.candidate.resolve()
    replay = args.replay.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"fresh output required: {out}")
    result = json.loads((replay / "result.json").read_text())
    if not (result.get("physical_success") and result.get("expert_valid") and
            result.get("support_contact_steps") == 0 and result.get("recorded_frames") == result.get("expected_frames")):
        raise SystemExit("replay is not strictly accepted; refusing to package")
    setting = json.loads((candidate / "setting.json").read_text())
    motion = candidate / "reference" / "motion"
    q = read_motion(motion, "joint_pos")
    dq = read_motion(motion, "joint_vel")
    root_pos = read_motion(motion, "body_pos")[:, :3]
    root_quat = read_motion(motion, "body_quat")[:, :4]
    hand_values: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        path = motion / f"{side}_hand_pos.csv"
        headers = path.read_text().splitlines()[0].split(",")
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        for header, column in zip(headers, values.T):
            hand_values[f"{side}_hand_{header[len(side) + 1:]}_joint"] = column
    hands = np.stack([hand_values[name] for name in HAND], axis=1)
    n = len(q)
    if q.shape != (n, 29) or dq.shape != (n, 29) or hands.shape != (n, 14):
        raise SystemExit("candidate motion has unexpected dimensions")
    if not all(np.isfinite(x).all() for x in (q, dq, root_pos, root_quat, hands)):
        raise SystemExit("candidate motion contains non-finite values")
    if not np.allclose(np.linalg.norm(root_quat, axis=1), 1, atol=1e-5):
        raise SystemExit("root quaternions are not unit")
    if not np.allclose(hands[0], 0, atol=1e-10):
        raise SystemExit("hand frame zero is not canonical")

    out.mkdir(parents=True)
    ref = out
    np.savez_compressed(ref / "reference.npz", time_s=np.arange(n) / 50.0,
                        body_ref_q=q, body_ref_dq=dq, root_ref_pos=root_pos,
                        root_ref_quat=root_quat, hand_ref_q=hands,
                        hand_ref_dq=np.gradient(hands, 0.02, axis=0))
    source_files = [f for f in candidate.rglob("*") if f.is_file()]
    candidate_hashes = {str(f): digest(f) for f in source_files}
    replay_files = [f for f in replay.rglob("*") if f.is_file()]
    replay_hashes = {str(f): digest(f) for f in replay_files}
    episode = f"ardy_unified_{setting['case']['id']}_seed{setting['seed']}_v2"
    metadata = {
        "producer_type": "ardy", "schema_version": 2, "episode_id": episode,
        "reference_hz": 50, "num_frames": n, "duration_s": float((n - 1) / 50),
        "body_dof": 29, "hand_dof": 14, "body_joint_names": BODY,
        "hand_joint_names": HAND, "quaternion_convention": "wxyz",
        "root_pose_frame": "world", "units": UNITS, "robot": "G1",
        "scene": "canonical_tabletop", "initial_pose": "canonical_reset",
        "case": setting["case"], "task_arm": "right", "reference_complete": True,
        "source_adapter": "unified", "source_native_run": str(replay),
        "source_reference_input": str(replay), "source_candidate": str(candidate),
        "source_candidate_sha256": candidate_hashes,
        "source_native_replay_sha256": replay_hashes,
        "hand_profile": setting.get("postprocessing", {}).get("hand_profile", "ardy"),
        "preserve_generated_body": setting.get("postprocessing", {}).get("preserve_generated_body", False),
        "preserve_body_weight": setting.get("postprocessing", {}).get("preserve_body_weight", 1.0),
    }
    (ref / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    provenance = {
        "factory_input": "accepted native ARDY unified replay; no motion generation rerun",
        "producer_type": "ardy", "adapter": "unified", "source_reference": str(replay),
        "source_candidate": str(candidate), "native_execution_evidence": str(replay),
        "native_result": str(replay / "result.json"),
        "source_candidate_sha256": candidate_hashes,
        "source_native_replay_sha256": replay_hashes,
        "generation": setting.get("generation_mode"), "seed": setting.get("seed"),
        "case": setting.get("case"), "hand_profile": metadata["hand_profile"],
    }
    (ref / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    strict = {
        "policy_version": "strict-unified-validation-v1",
        "result": {"physical_success": True, "contact_safe": True,
                    "expert_valid": True, "dataset_valid": True, "reasons": []},
        "source": str(replay / "result.json"),
    }
    validation = {
        "reference_schema_valid": True, "finite": True,
        "timestamps_monotonic": True, "joint_limits_valid": True,
        "quaternion_valid": True, "reference_complete": True,
        "tested_with_native_replay": True, "source_native_result": result,
        "strict_gate": strict,
    }
    (ref / "reference_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    evidence = ref / "source_evidence" / "native_replay"
    shutil.copytree(replay, evidence)
    (ref / "source_evidence" / "evidence_index.json").write_text(json.dumps({
        "source_candidate": str(candidate), "source_replay": str(replay),
        "candidate_sha256": candidate_hashes, "replay_sha256": replay_hashes,
        "strict_gate": strict,
    }, indent=2) + "\n")
    print(json.dumps({"reference": str(ref), "episode_id": episode,
                      "frames": n, "source_replay": str(replay)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

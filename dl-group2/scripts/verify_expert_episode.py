#!/usr/bin/env python3
"""Audit saved references/rollouts and produce an honest A/B summary.

Physical contact/hold certification comes from the runtime's per-physics-step
evaluator, not from interpolating the 50 Hz dataset or trusting playback alone.
Exit 2 means the episode is not a validated expert, even if its files are valid.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from expert_trajectory import load_reference, write_json
from project_paths import DEFAULT_EPISODE
from episode_schema import validate_dataset_metadata, validate_images


def select_saved_run(episode, name):
    """Choose which measured rollout is exposed at the episode root, without relabeling it."""
    if Path(name).name != name:
        raise ValueError("Run name must be one directory name")
    run = episode / "validation" / name
    meta = json.loads((run / "metadata.json").read_text())
    report = json.loads((run / "report.json").read_text())
    parent = json.loads((episode / "metadata.json").read_text())
    for path in (episode / "candidates" / meta["candidate_type"]).iterdir():
        if path.is_file() and path.name != "metadata.json":
            shutil.copy2(path, episode / path.name)
    for path in run.glob("*.npy"):
        name = "actual_timestamps.npy" if path.name == "timestamps.npy" else path.name
        shutil.copy2(path, episode / name)
    parent.update(meta)
    parent.update(selected_candidate=meta["candidate_type"], selected_validation_run=run.name,
                  validation_report=str(run / "report.json"),
                  expert_valid=report["expert_valid"], failure_reason=report["failure_reason"])
    write_json(episode / "metadata.json", parent)
    info = episode / "info.txt"
    info.write_text(info.read_text()+f"\nfinal_expert_valid: {report['expert_valid']}\nselected_validation: {run.name}\n")


def verify(episode, check_csv=True):
    candidates = {}
    for name in ("actual", "target"):
        if not (episode / "candidates" / name).exists(): continue
        arrays, meta = load_reference(episode / "candidates" / name)
        candidates[name] = arrays
    source = json.loads((episode / "source/report.json").read_text())
    rows = []
    shape = {"body_q": 29, "body_dq": 29, "hand_q": 14, "hand_dq": 14,
             "root_pos": 3, "root_quat": 4, "cube_pos": 3, "cube_quat": 4}
    for path in sorted((episode / "validation").glob("*/report.json")):
        report = json.loads(path.read_text())
        meta = json.loads((path.parent / "metadata.json").read_text())
        if meta["candidate_type"] not in candidates:
            candidates[meta["candidate_type"]], _ = load_reference(episode / "candidates" / meta["candidate_type"])
        ref = candidates[meta["candidate_type"]]
        frames = np.load(path.parent / "reference_frame.npy")
        assert frames.ndim == 1 and np.issubdtype(frames.dtype, np.integer)
        assert np.all((frames >= 0) & (frames < len(ref["timestamps"])))
        assert len(set(frames.tolist())) == len(frames)
        for key, width in shape.items():
            a = np.load(path.parent / f"{key}.npy")
            assert a.shape == (len(frames), width) and np.isfinite(a).all(), (path, key)
            if key.endswith("quat"):
                assert np.allclose(np.linalg.norm(a, axis=1), 1., atol=1e-5)
        ts = np.load(path.parent / "timestamps.npy")
        assert np.array_equal(ts, ref["timestamps"][frames])
        applied = np.load(path.parent / "hand_applied_ref_q.npy")
        assert np.allclose(applied, ref["hand_ref_q"][frames], atol=1e-8, rtol=0)
        assert report["recorded_frames"] == len(frames)
        certified = bool(report["expert_valid"])
        if certified:
            assert meta["expert_valid"] and meta["source_success"]
            assert meta["elastic_band"] == "off"
            assert meta["initialization_support"] == "none"
            assert meta["pelvis_supported_during_final_rollout"] == "none"
            assert not meta["pelvis_supported_during_source_generation"]
            assert not np.any(np.load(path.parent / "elastic_band_enabled.npy"))
            assert report["episode_completed"] and report["hand_synchronized"]
            assert frames.tolist() == list(range(len(ref["timestamps"])))
            assert not report["failure_reason"] and not report["fall_steps"]
            assert report["physical_grasp_success"] and not report["cube_attached"]
            assert not report["cube_teleported_during_rollout"]
            assert report["final_lift_m"] >= .10
            assert report["final_continuous_hold_seconds"] >= 2.
            assert report["sonic_target_max_error_rad"] <= 1e-5
        rows.append({"run": path.parent.name, "candidate": meta["candidate_type"],
                     "band": meta["elastic_band"], "kp_scale": meta["sonic_kp_scale"],
                     "kd_scale": meta.get("sonic_kd_scale",1.),
                     "pelvis_support": meta["pelvis_supported_during_final_rollout"],
                     "physics_hz": meta.get("physics_hz", 500),
                     **{k: report[k] for k in ("recorded_frames", "body_reference_rmse_rad",
                         "max_lift_m", "final_lift_m", "final_continuous_hold_seconds",
                         "fall_steps", "robot_table_contact_steps", "hand_synchronized",
                         "expert_valid", "failure_reason")}})
    parent = json.loads((episode / "metadata.json").read_text())
    if parent.get('schema_version') == 3:
        validate_dataset_metadata(parent)
        initial_cube=np.load(episode/'source/cube_pos.npy')[0]
        initial_quat=np.load(episode/'source/cube_quat.npy')[0]
        assert np.allclose(parent['task_config']['cube_initial_position'], initial_cube, atol=1e-12)
        assert np.allclose(parent['task_config']['cube_initial_quaternion_wxyz'], initial_quat, atol=1e-12)
    source_physical = source.get("force_checked", source)
    source_success = source.get("source_success", source.get("original", {}).get("success", False))
    assert parent["source_success"] == bool(source_success and source_physical["physical_grasp_success"])
    selected = parent.get("selected_validation_run")
    selected_row = next((r for r in rows if r["run"] == selected), None)
    assert parent["expert_valid"] == bool(selected_row and selected_row["expert_valid"])
    if selected_row:
        root_refs, _ = load_reference(episode)
        for key, a in root_refs.items():
            assert np.array_equal(a, candidates[selected_row["candidate"]][key]), key
        for key, width in shape.items():
            a = np.load(episode / f"{key}.npy")
            original = np.load(episode / "validation" / selected / f"{key}.npy")
            assert a.shape == (selected_row["recorded_frames"],width) and np.isfinite(a).all()
            assert np.array_equal(a,original), key
        assert np.array_equal(np.load(episode/"actual_timestamps.npy"),
                              np.load(episode/"validation"/selected/"timestamps.npy"))
    image_audit=None
    if check_csv and selected_row and parent.get('schema_version') == 3 and (episode/'csv_manifest.json').exists():
        image_audit=validate_images(episode/'validation'/selected, root_refs['timestamps'])
        if parent.get('expert_valid') and parent.get('initial_pose',{}).get('name')=='neutral_standing_v1':
            selected_report=json.loads((episode/'validation'/selected/'report.json').read_text())
            assert selected_report['robot_self_contact_steps']==0
            assert selected_report['left_arm_environment_contact_steps']==0
    delta = (candidates["target"]["body_ref_q"]-candidates["actual"]["body_ref_q"]) if "target" in candidates else None
    if check_csv and selected_row and (episode / "csv_manifest.json").exists():
        # Check exported dataset states as well as binary trial arrays.
        for key in ("body_ref_q", "body_ref_dq", "hand_ref_q", "hand_ref_dq", "body_q", "body_dq", "hand_q", "hand_dq", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "cube_pos", "cube_quat"):
            csv = np.loadtxt(episode / f"{key}.csv", delimiter=",", skiprows=1, ndmin=2)
            arr = np.load(episode / f"{key}.npy")
            assert csv.shape == arr.shape and np.allclose(csv, arr, atol=6e-6, rtol=0), key
        candidate_path = episode / "candidates" / parent["selected_candidate"]
        for filename in ("joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv", "body_lin_vel.csv", "body_ang_vel.csv"):
            assert (episode / "sonic_reference" / filename).read_bytes() == (candidate_path / filename).read_bytes(), filename
    summary = {"files_valid": True, "expert_valid": parent["expert_valid"],
               "source_success": parent["source_success"], "image_alignment": image_audit,
               "source_physical_report": source_physical,
               "target_vs_actual_body_rmse_rad": float(np.sqrt(np.mean(delta**2))) if delta is not None else None,
               "selected_validation_run": selected, "runs": rows}
    incomplete = [{"run":p.parent.name,**json.loads(p.read_text())}
                  for p in sorted((episode/"validation").glob("*/INCOMPLETE.json"))]
    assert not any(r["expert_valid"] for r in incomplete)
    summary["incomplete_runs"] = incomplete
    write_json(episode / "validation_summary.json", summary)
    lines = ["# Physical SONIC + hand validation", "",
             f"Expert valid: **{parent['expert_valid']}**. Source grasp success: **{parent['source_success']}**.", "",
             "No success episode is certified unless the complete SONIC + hand physical rollout passes.", "",
             f"Source final lift: {source_physical['final_lift_m']:.5f} m; final continuous hold: {source_physical['final_continuous_hold_seconds']:.3f} s.", "",
             f"Source controller: {parent.get('source_controller', 'unknown')}. Tracking accuracy alone does not certify a grasp.", "",
             "| Run | Candidate | Band | Pelvis support | Physics Hz | Kp scale | Frames | Body RMSE rad | Max lift m | Final hold s | Fall steps | Table contact steps | Hand sync | Expert |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        rmse = f"{r['body_reference_rmse_rad']:.5f}" if r['body_reference_rmse_rad'] is not None else "N/A"
        lines.append(f"| {r['run']} | {r['candidate']} | {r['band']} | {r['pelvis_support']} | {r['physics_hz']} | {r['kp_scale']} | {r['recorded_frames']} | "
                     f"{rmse} | {r['max_lift_m']:.5f} | {r['final_continuous_hold_seconds']:.3f} | "
                     f"{r['fall_steps']} | {r['robot_table_contact_steps']} | {r['hand_synchronized']} | {r['expert_valid']} |")
    lines += ["", "## Failure reasons", ""]
    lines += [f"- `{r['run']}`: {', '.join(r['failure_reason']) or 'none'}" for r in rows]
    lines += [f"- INCOMPLETE `{r['run']}`: {r['failure_reason']}" for r in incomplete]
    lines += ["", "## Interpretation", "",
              "Compare actual/target and band ON/OFF under matching physics rate, pelvis support and gains. Partial-frame or time-lag trials are development results; a welded-pelvis run is distinct from a free-base band-OFF run. Derived tracking-compensation references are not original measured source references.", "",
              f"Root actual arrays expose `{selected}`. Other measured trials and images remain in their own validation folders. Current certified successful episode path: {'see selected validation run' if parent['expert_valid'] else 'none'}."]
    (episode / "VALIDATION.md").write_text("\n".join(lines)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--select-run", help="Expose this saved run at the episode root; preserves its success/failure status")
    args = parser.parse_args()
    if args.select_run:
        select_saved_run(args.episode.resolve(), args.select_run)
    result = verify(args.episode.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["expert_valid"] else 2)

"""Run a predeclared Kimodo expert-reference batch without an SSH session."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write_atomic(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_used_mib(gpu):
    text = subprocess.check_output(["mthreads-gmi"], text=True, timeout=8)
    match = re.search(rf"^\s*{gpu}\s+MTT S4000[^\n]*?\s+(\d+)MiB\(", text, re.M)
    if not match:
        raise RuntimeError(f"Cannot parse physical GPU {gpu} memory")
    return int(match.group(1))


def controller_busy():
    result = subprocess.run(
        ["pgrep", "-af", "g1_deploy_onnx_ref|run_sim_loop.py|run_sonic_grasp.py|replay_validate_kimodo.py"],
        capture_output=True,
        text=True,
    )
    return any("pgrep -af" not in line for line in result.stdout.splitlines())


def wait_for_shared_resources(gpu, state):
    while True:
        try:
            memory = gpu_used_mib(gpu)
            busy = controller_busy()
            if memory < 512 and not busy:
                return
            state["resource_wait"] = {"gpu_used_mib": memory, "controller_busy": busy}
        except Exception as exc:
            state["resource_wait"] = {"error": str(exc)}
        write_atomic(ROOT / "status.json", state)
        time.sleep(20)


def patch_generation_runner(code):
    replacements = {
        "c['seed']==1": "c['seed'] in range(2, 4)",
        "seed=1,trial_count=1": "seed=c['seed'],trial_count=1",
        "g['seed']!=1": "g['seed']!=c['seed']",
        "mode='development_stability_probe'": "mode='fixed_adapter_batch_trial'",
    }
    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"Runner patch anchor missing: {old}")
        code = code.replace(old, new)
    compile(code, "development_full_once.py", "exec")
    return code


def patch_shell_runner(shell, source_root, trial_root, old_tmp, new_tmp):
    if old_tmp not in shell:
        raise RuntimeError(f"Shell TMPDIR anchor missing: {old_tmp}")
    if "development_full_once.py" not in shell:
        raise RuntimeError("Shell generation entry point missing")
    return shell.replace(source_root, trial_root).replace(old_tmp, new_tmp)


def prepare_trial(seed, index, plan):
    source = Path(plan["accepted_profile"])
    trial = ROOT / "trials" / f"seed_{seed:03d}"
    if trial.exists():
        raise FileExistsError(f"Trial already exists: {trial}")
    trial.mkdir(parents=True)
    (trial / "qualification").mkdir()
    (trial / "assets").symlink_to(Path(plan["frozen_asset_root"]), target_is_directory=True)
    for part in ("code", "config", "tests"):
        shutil.copytree(source / part, trial / part)
    for file in (trial / "config").glob("*.json"):
        file.write_text(file.read_text().replace(str(source), str(trial)))
    config_file = trial / "config/frozen_config.json"
    config = read(config_file)
    node_tmp = f"/tmp/task9_q4_kimodo_{seed:03d}"
    Path(node_tmp).mkdir(exist_ok=False)
    config["seed"] = seed
    config["source_task"] = str(trial / "qualification/task")
    config["source_generation"] = str(trial / "qualification/generation")
    config["reuse_actual_generation"] = False
    config["development_iteration"] = index + 1
    config["development_prior_replay_count"] = plan["known_prior_development_replays"]
    config["retry_count"] = 0
    config["manual_modification_count"] = 0
    config["batch_mode"] = "task9_qualification4_trial"
    config["formal_qualification"] = False
    config["read_allowlist"] = [
        item.replace(plan["accepted_node_tmp"], node_tmp)
        for item in config["read_allowlist"]
    ]
    config["adapter"]["calibration"] = str(trial / "config/development_calibration.json")
    write_atomic(config_file, config)
    cal_file = trial / "config/development_calibration.json"
    calibration = read(cal_file)
    calibration["task_id"] = f"kimodo-center-seed{seed}-16p2-v1"
    calibration["transfer_note"] = (
        "Fixed own run_01 seed0 calibration, copied before this batch; "
        "never updated from a batch outcome."
    )
    write_atomic(cal_file, calibration)
    write_atomic(
        trial / "config/config_equivalence.json",
        {
            "fixed_adapter_parameters": True,
            "new_generation_seed": seed,
            "model_settings": "same checkpoint, prompts, denoising, CFG and segmentation",
            "tracking_bias": "fixed own seed0 development replay",
        },
    )
    script = trial / "code/development_full_once.py"
    script.write_text(patch_generation_runner(script.read_text()))
    runner = trial / "run_once.sh"
    shell = patch_shell_runner(
        (source / "run_stability_once.sh").read_text(),
        str(source),
        str(trial),
        plan["accepted_node_tmp"],
        node_tmp,
    )
    runner.write_text(shell)
    runner.chmod(0o755)
    write_atomic(
        trial / "declaration.json",
        {
            "case": "center",
            "seed": seed,
            "episode_id": f"kimodo_episode_{index + 1:04d}",
            "one_generation": True,
            "one_candidate": True,
            "one_replay": True,
            "retry_count": 0,
            "post_start_manual_changes": 0,
            "predeclared_config_sha256": sha(config_file),
            "calibration_sha256": sha(cal_file),
        },
    )
    return trial


def summarize_trial(trial, seed, index, launch_exit, package_error, episode):
    q = trial / "qualification"
    generation = q / "generation/generation.json"
    verdict = q / "replay/validation_task_criteria_verified.json"
    timing = q / "timing.json"
    end = q / "execution_end.json"
    result = {
        "trial_index": index + 1,
        "seed": seed,
        "episode_id": f"kimodo_episode_{index + 1:04d}",
        "trial_path": str(trial),
        "launch_exit_code": launch_exit,
        "attempted": True,
        "expert_reference_delivered": episode is not None,
        "episode_path": str(episode) if episode else None,
        "package_error": package_error,
    }
    if generation.exists():
        g = read(generation)
        result.update(
            model_sampling_calls=g.get("model_call_count"),
            generation_seconds=g.get("generation_wall_seconds"),
            raw_sha256=g.get("raw_sha256"),
        )
    if verdict.exists():
        v = read(verdict)
        result.update(
            physical_success=v.get("physical_success"),
            expert_valid=v.get("expert_valid"),
            max_lift_m=v.get("max_lift_m"),
            hold_seconds=v.get("max_continuous_hold_seconds"),
            canonical_checks_passed=v.get("checks_passed"),
            canonical_checks_total=v.get("checks_total"),
        )
    else:
        result.update(physical_success=None, expert_valid=None)
    if timing.exists():
        t = read(timing)
        result.update(
            model_load_seconds=t.get("model_load_seconds"),
            adapter_seconds=t.get("adapter_postprocessing_seconds"),
            sonic_init_seconds=t.get("sonic_initialization_seconds"),
            replay_wall_seconds=t.get("replay_wall_seconds"),
            total_wall_seconds=t.get("total_wall_seconds"),
        )
    if end.exists():
        result["failure_reason"] = read(end).get("failure")
    if package_error and not result.get("failure_reason"):
        result["failure_reason"] = package_error
    return result


def write_results(rows):
    fields = [
        "trial_index", "seed", "episode_id", "physical_success", "expert_valid",
        "expert_reference_delivered", "max_lift_m", "hold_seconds",
        "model_load_seconds", "generation_seconds", "adapter_seconds",
        "sonic_init_seconds", "replay_wall_seconds", "total_wall_seconds",
        "model_sampling_calls", "failure_reason", "episode_path", "trial_path",
    ]
    pending = ROOT / "results.csv.pending"
    with pending.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(pending, ROOT / "results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if plan["seeds"] != list(range(2, 4)):
        raise ValueError("Unexpected predeclared seed schedule")
    ledger = ROOT / "batch_execution.json"
    with ledger.open("x") as output:
        json.dump({"plan_sha256": sha(args.plan), "start_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}, output, indent=2)
    state = {
        "status": "RUNNING",
        "phase": "first_four",
        "planned_trials": len(plan["seeds"]),
        "completed_trials": 0,
        "successes": 0,
        "failed_trials": 0,
        "expert_references_delivered": 0,
        "attempts": [],
        "background_process_pid": os.getpid(),
    }
    write_atomic(ROOT / "status.json", state)
    rows = []
    for index, seed in enumerate(plan["seeds"]):
        state["phase"] = "first_four" if index < 4 else "remaining_in_background"
        state["next_seed"] = seed
        write_atomic(ROOT / "status.json", state)
        wait_for_shared_resources(plan["physical_gpu"], state)
        trial = prepare_trial(seed, index, plan)
        print(f"START seed={seed} trial={index + 1}", flush=True)
        launch_exit = None
        package_error = None
        episode = None
        with (trial / "run.log").open("x") as log:
            launch_exit = subprocess.run(["bash", str(trial / "run_once.sh")], cwd=trial, stdout=log, stderr=subprocess.STDOUT).returncode
        verdict = trial / "qualification/replay/validation_task_criteria_verified.json"
        if verdict.exists() and read(verdict).get("expert_valid"):
            episode = ROOT / "episodes" / f"kimodo_episode_{index + 1:04d}"
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = f"{trial / 'code'}:{trial / 'code/canonical/scripts'}:{trial / 'assets/unitree_sdk2_python'}"
            packaged = subprocess.run(
                [sys.executable, str(ROOT / "pack_episode.py"), "--trial", str(trial), "--out", str(episode), "--episode-id", episode.name],
                env=env,
                cwd=trial,
                capture_output=True,
                text=True,
            )
            (trial / "package.log").write_text(packaged.stdout + packaged.stderr)
            if packaged.returncode:
                package_error = f"Expert reference packaging failed: {packaged.stderr.strip()}"
                episode = None
        row = summarize_trial(trial, seed, index, launch_exit, package_error, episode)
        rows.append(row)
        with (ROOT / "attempts.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
        write_results(rows)
        state["attempts"] = rows
        state["completed_trials"] = len(rows)
        state["successes"] = sum(row.get("expert_valid") is True for row in rows)
        state["failed_trials"] = len(rows) - state["successes"]
        state["expert_references_delivered"] = sum(row["expert_reference_delivered"] for row in rows)
        state["observed_success_rate"] = state["successes"] / len(rows)
        state.pop("resource_wait", None)
        write_atomic(ROOT / "status.json", state)
        print(f"DONE seed={seed} expert_valid={row.get('expert_valid')} episode={episode}", flush=True)
    state["status"] = "COMPLETE"
    state["phase"] = "complete"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_atomic(ROOT / "status.json", state)
    print(json.dumps({key: state[key] for key in ("status", "completed_trials", "successes", "failed_trials", "expert_references_delivered", "observed_success_rate")}))


if __name__ == "__main__":
    main()

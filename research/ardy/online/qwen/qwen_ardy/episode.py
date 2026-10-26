from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Paths


def run_logged(command: list[str], cwd: Path, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (log_path.with_suffix(log_path.suffix + ".command.json")).write_text(
        json.dumps({"cwd": str(cwd), "command": command}, indent=2) + "\n"
    )
    with log_path.open("w") as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def generate_motion(paths: Paths, attempt_dir: Path, prompt: str, target_delta: list[float],
                    gpu: int, dry_run: bool = False) -> Path:
    motion_parent = attempt_dir / "motion"
    motion = motion_parent / "target_block"
    motion.mkdir(parents=True, exist_ok=True)
    constraint_cmd = [
        sys.executable, "-m", "qwen_ardy.constraint_builder",
        "--output-dir", str(motion), "--scene", str(paths.scene), "--ardy-repo", str(paths.ardy),
        "--arms-at-sides", "--delta-x", str(target_delta[0]),
        "--delta-y", str(target_delta[1]), "--delta-z", str(target_delta[2]),
    ]
    generator = paths.ardy / "scripts/generate.py"
    converter = paths.sonic / "gear_sonic_deploy/reference/convert_ardy.py"
    source = motion / "ardy_motion.npz"
    generate_cmd = [
        sys.executable, str(generator), prompt, "--model", "g1",
        "--checkpoints_dir", str(paths.ardy / "checkpoints"), "--duration", "12",
        "--seed", "0", "--output", str(source.with_suffix("")),
        "--constraints", str(motion / "constraints.json"),
    ]
    convert_cmd = [
        sys.executable, str(converter), str(source), "--ardy-repo", str(paths.ardy),
        "--output-dir", str(motion),
    ]
    plan = {"prompt": prompt, "constraint_delta_world_m": target_delta,
            "hand_control": "contact_driven; no hand CSV; no fixed timing",
            "commands": [constraint_cmd, generate_cmd, convert_cmd]}
    (attempt_dir / "generation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if dry_run:
        return motion_parent
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env["MUSA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TEXT_ENCODERS_DIR", str(paths.ardy / "text_encoders"))
    env["TEXT_ENCODER_MODE"] = "local"
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    run_logged(constraint_cmd, paths.project, attempt_dir / "constraints.log", env)
    run_logged(generate_cmd, paths.ardy, attempt_dir / "ardy_generation.log", env)
    run_logged(convert_cmd, paths.sonic, attempt_dir / "convert_motion.log", env)
    for forbidden in (motion / "left_hand_pos.csv", motion / "right_hand_pos.csv"):
        if forbidden.exists():
            raise RuntimeError(f"Timed hand trajectory is forbidden in staged mode: {forbidden}")
    return motion_parent


def run_episode(paths: Paths, attempt_dir: Path, motion_parent: Path, viewer: bool, gpu: int,
                dry_run: bool = False) -> Path:
    output = attempt_dir / "episode"
    command = [
        sys.executable, str(paths.project / "qwen_ardy/telemetry_runner.py"),
        "--arms-at-sides", "--robot-x", "0.24", "--robot-y", "-0.03", "--hand-kp", "6",
        "--motion-dir", str(motion_parent), "--scene", str(paths.scene), "--output-dir", str(output),
    ]
    if viewer:
        command.append("--viewer")
    (attempt_dir / "episode_command.json").write_text(json.dumps(command, indent=2) + "\n")
    if dry_run:
        return output
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env["MUSA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, cwd=paths.sonic, env=env, check=True)
    return output


def render_episode(paths: Paths, output: Path) -> list[Path]:
    command = [sys.executable,
               str(paths.sonic / "gear_sonic_deploy/reference/render_ardy_grasp.py"), str(output)]
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", "/root/.Xauthority")
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["MUJOCO_GL"] = "glfw"
    try:
        run_logged(command, paths.sonic, output / "render.log", env)
    except subprocess.CalledProcessError:
        return []
    return [p for p in (output / "frame_6.png", output / "frame_7.png", output / "frame_10.png") if p.exists()]

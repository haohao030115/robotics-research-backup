from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from .config import Paths
from .episode import generate_motion, render_episode, run_episode
from .models import PlannerDecision, semantic_prompt
from .qwen_vlm import QwenPlanner, heuristic_decision
from .state import summarize_episode, write_summary


def choose_planner(mode: str):
    if mode == "heuristic":
        return None
    if mode == "qwen" or (mode == "auto" and os.getenv("DASHSCOPE_API_KEY")):
        return QwenPlanner()
    return None


def main():
    parser = argparse.ArgumentParser(description="Episode-level Qwen → ARDY → SONIC grasp loop")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--planner", choices=["auto", "qwen", "heuristic"], default="auto")
    parser.add_argument("--gpu", type=int, choices=[4, 5, 6, 7], default=7)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--reuse-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    paths = Paths.discover()
    paths.validate()
    run_dir = args.run_dir or paths.project / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    planner = choose_planner(args.planner)
    decision = PlannerDecision(
        assessment="Initial validated semantic plan.", strategy="retry",
        semantic_action="approach_and_grasp", constraint_delta_world_m=[0, 0, 0], confidence=1,
    )
    history = []

    for attempt in range(args.max_attempts):
        attempt_dir = run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir()
        prompt = semantic_prompt(decision)
        (attempt_dir / "planner_decision.json").write_text(decision.model_dump_json(indent=2) + "\n")
        (attempt_dir / "ardy_prompt.txt").write_text(prompt + "\n")
        if attempt == 0 and args.reuse_baseline:
            motion_parent = paths.baseline_motion
        else:
            motion_parent = generate_motion(
                paths, attempt_dir, prompt, decision.constraint_delta_world_m,
                args.gpu, args.dry_run,
            )
        output = run_episode(paths, attempt_dir, motion_parent, args.viewer, args.gpu, args.dry_run)
        if args.dry_run:
            history.append({"attempt": attempt, "prompt": prompt, "motion": str(motion_parent)})
            decision = PlannerDecision(
                assessment="Dry-run synthetic correction.", strategy="move_closer",
                semantic_action="careful_reach", constraint_delta_world_m=[0.01, 0, 0], confidence=.5,
            )
            continue
        summary = summarize_episode(attempt, output)
        write_summary(summary, attempt_dir / "summary.json")
        images = render_episode(paths, output)
        history.append(summary.to_dict())
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(summary.to_dict(), indent=2), flush=True)
        if summary.success:
            print(f"SUCCESS on attempt {attempt}: {run_dir}", flush=True)
            return
        try:
            decision = planner.decide(summary, images) if planner else heuristic_decision(summary)
        except Exception as exc:
            (attempt_dir / "qwen_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            decision = heuristic_decision(summary)
        (attempt_dir / "next_planner_decision.json").write_text(
            decision.model_dump_json(indent=2) + "\n"
        )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if args.dry_run:
        print(f"Dry-run complete: {run_dir}")
        return
    raise SystemExit(f"No successful grasp after {args.max_attempts} attempts; see {run_dir}")


if __name__ == "__main__":
    main()

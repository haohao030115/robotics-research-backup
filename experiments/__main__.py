"""Entry point for safe planning and model-family inspection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.ardy.workflow import build_commands as ardy_commands
from research.kimodo.workflow import build_commands as kimodo_commands
from va_research.registry import get_family

from .run import plan
from .execute import execute_stages
from .workflows import canonical_commands, dataset_commands, expert_commands, va_commands
from .catalog import write_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    family = sub.add_parser("family")
    family.add_argument("name")
    p = sub.add_parser("plan")
    p.add_argument("--family", choices=("kimodo", "ardy"), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", default="grasp")
    p.add_argument("--config", type=Path, default=Path("research/ardy/config/default.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", default="center")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--adapter", choices=("legacy", "unified"), default="legacy")
    p.add_argument("--hand-profile", choices=("default", "ardy", "slow_close", "early_preshape"), default=None)
    r = sub.add_parser("run", help="execute an expert workflow")
    r.add_argument("--family", choices=("kimodo", "ardy"), required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--prompt", default="grasp")
    r.add_argument("--config", type=Path, default=Path("research/ardy/config/default.json"))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--case", default="center")
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--adapter", choices=("legacy", "unified"), default="legacy")
    r.add_argument("--hand-profile", choices=("default", "ardy", "slow_close", "early_preshape"), default=None)
    r.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("canonical", help="execute native canonical production")
    c.add_argument("--run-id", required=True); c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True); c.add_argument("--gpu", type=int, default=0)
    c.add_argument("--mode", choices=("audit", "production"), default="audit"); c.add_argument("--dry-run", action="store_true")
    d = sub.add_parser("dataset", help="compile an existing canonical root")
    d.add_argument("--run-id", required=True); d.add_argument("--canonical-root", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True); split = d.add_mutually_exclusive_group()
    split.add_argument("--all-train", action="store_true", help="use every accepted episode for training")
    split.add_argument("--val-count", type=int, help="held-out validation episode count")
    d.add_argument("--seed", type=int)
    d.add_argument("--dry-run", action="store_true")
    cat = sub.add_parser("catalog", help="index existing manifests and result artifacts")
    cat.add_argument("--root", type=Path, action="append", help="artifact root (repeatable; defaults to outputs and expert_datasets)")
    cat.add_argument("--output", type=Path, default=Path("docs/artifact_catalog.json"))
    v = sub.add_parser("va", help="execute VA training, offline evaluation, or rollout")
    v.add_argument("operation", choices=("train", "offline", "closed")); v.add_argument("--family", choices=("v4", "bcz_hand"), default="v4")
    v.add_argument("--run-id", required=True); v.add_argument("--output", type=Path, required=True)
    v.add_argument("--manifest", type=Path); v.add_argument("--checkpoint", type=Path); v.add_argument("--config", type=Path)
    v.add_argument("--device", default="musa:0"); v.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "family":
        item = get_family(args.name)
        print(item)
        return 0
    root = Path(__file__).resolve().parents[1]
    if args.command == "catalog":
        path = write_catalog(args.root or [root / "outputs", root / "expert_datasets"], args.output)
        print(path)
        return 0
    if args.command == "run":
        commands = expert_commands(args.family, root, args.output, prompt=args.prompt, config=args.config,
                                   seed=args.seed, case=args.case, gpu=args.gpu, adapter=args.adapter,
                                   hand_profile=args.hand_profile or ("ardy" if args.family == "ardy" else "default"))
        if args.dry_run:
            print(plan(args.run_id, args.family, commands, output=args.output)); return 0
        environment = None
        if args.family == "kimodo":
            paths = [
                root / "research/kimodo/generation",
                root / "research/kimodo/export",
                root / "research/kimodo/qualification",
                root / "research/kimodo/runtime_support/scripts",
                root / "kimodo-t6/g1-integration/task8_qualification_v2/code",
            ]
            formal_config = root / "workspace/kunqi/task9_formal40_20261008/kimodo" / "trials" / f"seed_{args.seed}" / "config" / "frozen_config.json"
            if not formal_config.is_file():
                formal_config = root / "workspace/kunqi/task9_formal40_20261008/kimodo_frozen_method_source/config/frozen_config.json"
            environment = {
                "RESEARCH_ROOT": str(root),
                "KIMODO_RUNTIME_SUPPORT": str(root / "research/kimodo/runtime_support"),
                "KIMODO_FROZEN_CONFIG": str(formal_config),
                "SONIC_REPO": str(root / "GR00T-WholeBodyControl"),
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        else:
            environment = {
                "RESEARCH_ROOT": str(root),
                "DL_GROUP2_ROOT": str(root / "dl-group2"),
                "SONIC_REPO": str(root / "GR00T-WholeBodyControl"),
                "ARDY_REPO": str(root / "workspace/fuyuhan/ardy"),
                "PYTHONPATH": os.pathsep.join([
                    str(root / "research/ardy/generation"),
                    str(root / "research/ardy/overlap"),
                    str(root / "research/ardy/adaptation"),
                    str(root / "research/ardy/qualification"),
                    str(root / "research/ardy/runtime_support"),
                ]),
            }
        return execute_stages(run_id=args.run_id, family=args.family, commands=commands,
                              output=args.output, root=root, environment=environment)
    if args.command == "canonical":
        commands = canonical_commands(args.reference, args.output, gpu=args.gpu, mode=args.mode)
        if args.dry_run:
            print(plan(args.run_id, "canonical", commands, output=args.output)); return 0
        return execute_stages(run_id=args.run_id, family="canonical", commands=commands,
                              output=args.output, root=root)
    if args.command == "dataset":
        if args.val_count is not None and args.seed is None:
            parser.error("--seed is required with --val-count")
        commands = dataset_commands(args.canonical_root, args.output, val_count=args.val_count, seed=args.seed, all_train=args.all_train)
        if args.dry_run:
            print(plan(args.run_id, "dataset", commands, output=args.output)); return 0
        return execute_stages(run_id=args.run_id, family="dataset", commands=commands,
                              output=args.output, root=root)
    if args.command == "va":
        commands = va_commands(args.family, args.operation, checkpoint=args.checkpoint,
                               manifest=args.manifest, output=args.output, config=args.config,
                               device=args.device)
        if args.dry_run:
            print(plan(args.run_id, f"va:{args.family}:{args.operation}", commands, output=args.output)); return 0
        return execute_stages(run_id=args.run_id, family=f"va:{args.family}:{args.operation}",
                              commands=commands, output=args.output, root=root)
    if args.family == "kimodo":
        commands = kimodo_commands(project_root=root, prompt=args.prompt, output=args.output, seed=args.seed, case=args.case)
    else:
        commands = ardy_commands(project_root=root, config=args.config, output=args.output, seed=args.seed, case=args.case, gpu=args.gpu)
    path = plan(args.run_id, args.family, commands, output=args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

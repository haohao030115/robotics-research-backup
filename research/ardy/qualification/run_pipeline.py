"""Five-stage ARDY -> Kimodo-style adaptation -> isolated SONIC -> validation."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _resolve_calibration(cli_value, cfg, config_path):
    """Return the calibration selected by the CLI or adapter config.

    A command-line value remains authoritative for backwards compatibility.
    When it is omitted, resolve the adapter's configured path relative to the
    config file so that copied/relocated pipeline invocations keep the same
    provenance.  ``None`` and an empty config value disable calibration.
    """
    if cli_value is not None:
        return Path(cli_value).resolve()
    configured = cfg.get('adapter', {}).get('calibration')
    if not configured:
        return None
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = config_path.resolve().parent / candidate
    return candidate.resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--case', choices=['center', 'near', 'far', 'left', 'right'])
    p.add_argument('--gpu', choices=['4', '5', '6', '7'])
    p.add_argument('--calibration', type=Path)
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--spatial-conditioned-prompt', action='store_true',
                   help='Append one deterministic case-derived spatial descriptor to each ARDY stage prompt')
    p.add_argument('--adapter', choices=['legacy', 'unified'], default='legacy')
    p.add_argument('--hand-profile', choices=['default','ardy','ardy_thumb_relief','ardy_thumb_relief_strong','slow_close','early_preshape',
                                              'timing_jitter','coordination_mild','coordination_tiny'],
                   default='ardy')
    p.add_argument('--preserve-generated-body', action='store_true')
    p.add_argument('--preserve-body-weight', type=float, default=1.0)
    p.add_argument('--body-preservation-policy', choices=['legacy', 'bounded'], default='legacy')
    p.add_argument('--opposition-centering', action='store_true',
                   help='Opt in to bounded geometry-derived distal-pair centering during closure')
    p.add_argument('--distal-alignment', action='store_true',
                   help='Opt in to bounded 3-D cube/distal-pair alignment during preclose and closure')
    p.add_argument('--distal-alignment-max-correction-m', type=float, default=0.04,
                   help='Maximum 3-D distal alignment correction (metres), at most 0.04')
    p.add_argument('--collision-aware-hand', action='store_true',
                   help='Opt in to bounded MuJoCo thumb-to-wrist clearance projection')
    p.add_argument('--collision-hand-min-clearance-m', type=float, default=0.001)
    p.add_argument('--collision-hand-max-relief-rad', type=float, default=0.30)
    p.add_argument('--eef-lift-offset-m', type=float, nargs=3, metavar=('DX','DY','DZ'),
                   default=(0.0, 0.0, 0.0),
                   help='Optional shared task-space EEF offset in metres; native gates remain authoritative')
    p.add_argument('--eef-lift-offset-window-s', type=float, nargs=2, metavar=('START','END'),
                   default=(10.0, 11.0),
                   help='Smoothstep onset window for the optional EEF offset')
    a = p.parse_args()
    qualification = Path(__file__).resolve().parent
    root = qualification.parent
    workspace_root = Path(os.environ.get('RESEARCH_ROOT', root.parent.parent)).resolve()
    code = root
    generation = root / 'generation'
    overlap = root / 'overlap'
    adaptation = root / 'adaptation'
    # ARDY qualification configs are JSON.  Keep a clear error for accidental
    # Hydra/Kimodo YAML configs instead of leaking JSONDecodeError from the
    # parser with no indication of the expected format.
    try:
        cfg = json.loads(a.config.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ARDY config must be JSON (received {a.config}); "
            "use research/ardy/config/default.json or a derived JSON config"
        ) from exc
    if a.seed is not None:
        cfg['seed'] = a.seed
    if a.case:
        cfg['case'] = a.case
    if a.gpu:
        cfg['physical_gpu'] = int(a.gpu)
    if cfg['physical_gpu'] not in (4, 5, 6, 7):
        p.error('Use GPU4--7')
    if a.spatial_conditioned_prompt:
        from research.shared_adapter.spatial_prompt import VERSION, condition_prompts, descriptor
        cfg['prompts'] = condition_prompts(cfg['prompts'], cfg['case'])
        cfg['spatial_prompt_mode'] = VERSION
        cfg['spatial_prompt_descriptor'] = descriptor(cfg['case'])
    else:
        cfg['spatial_prompt_mode'] = None
        cfg['spatial_prompt_descriptor'] = None
    calibration = _resolve_calibration(a.calibration, cfg, a.config)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(code, out/'code_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    config = out/'config.json'
    config.write_text(json.dumps(cfg, indent=2)+'\n')
    ardy = Path(os.environ.get('ARDY_REPO', root.parent.parent / 'workspace/fuyuhan/ardy')).resolve()
    gpu = str(cfg['physical_gpu'])
    env = os.environ.copy()
    env.update(ARDY_PHYSICAL_GPU=gpu, MUSA_VISIBLE_DEVICES=gpu, CUDA_VISIBLE_DEVICES=gpu,
               TEXT_ENCODER_MODE='local', TEXT_ENCODERS_DIR=str(ardy/'text_encoders'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               XDG_CACHE_HOME=str(root/'cache'), TORCH_HOME=str(root/'cache/torch'))
    trace = dict(config=cfg, calibration=str(calibration) if calibration else None,
                 stages=[], raw_motion_reused=False, other_workspaces_modified=False)

    def run(name, cmd, cwd=None):
        print('START', name, flush=True)
        start = time.monotonic()
        with (out/(name+'.log')).open('w') as f:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        trace['stages'].append(dict(name=name, command=cmd, exit_code=r.returncode, wall_s=time.monotonic()-start))
        (out/'pipeline.json').write_text(json.dumps(trace, indent=2)+'\n')
        print('END', name, r.returncode, flush=True)
        return r.returncode

    source = out/'source'
    if run('prepare', [sys.executable, str(generation/'prepare_conditions.py'), '--out', str(source),
                       '--case', cfg['case'], '--seed', str(cfg['seed']), '--prompt', ' | '.join(cfg['prompts'])]):
        return 2
    if run('generate', [sys.executable, str(overlap/'generate_segmented.py'),
                        '--source', str(source), '--config', str(config)], ardy):
        return 2
    command = [sys.executable, str(adaptation/'adapt_ardy.py'), '--source', str(source), '--out', str(out/'candidate'), '--config', str(config), '--adapter', a.adapter, '--hand-profile', a.hand_profile]
    if a.preserve_generated_body:
        command.append('--preserve-generated-body')
    command += ['--preserve-body-weight', str(a.preserve_body_weight)]
    command += ['--body-preservation-policy', a.body_preservation_policy]
    command += ['--eef-lift-offset-m', *map(str, a.eef_lift_offset_m),
                '--eef-lift-offset-window-s', *map(str, a.eef_lift_offset_window_s)]
    if a.opposition_centering:
        command.append('--opposition-centering')
    if a.distal_alignment:
        command += ['--distal-alignment', '--distal-alignment-max-correction-m', str(a.distal_alignment_max_correction_m)]
    if a.collision_aware_hand:
        command += ['--collision-aware-hand',
                    '--collision-hand-min-clearance-m', str(a.collision_hand_min_clearance_m),
                    '--collision-hand-max-relief-rad', str(a.collision_hand_max_relief_rad)]
    if calibration:
        command += ['--calibration', str(calibration)]
    if run('adapt', command):
        return 2
    if a.generate_only:
        return 0
    if run('replay', ['bash', str(root/'replay_isolated.sh'), str(workspace_root), str(out/'candidate'), str(out/'replay')]):
        return 2
    return run('verify', [sys.executable, str(qualification/'verify_run.py'), str(out/'replay')])


if __name__ == '__main__':
    raise SystemExit(main())

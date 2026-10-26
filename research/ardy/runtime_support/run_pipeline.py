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
    p.add_argument('--adapter', choices=['legacy', 'unified'], default='legacy')
    p.add_argument('--hand-profile', choices=['default','ardy','ardy_thumb_relief','ardy_thumb_relief_strong','slow_close','early_preshape',
                                              'timing_jitter','coordination_mild','coordination_tiny'], default='ardy')
    p.add_argument('--preserve-generated-body', action='store_true')
    p.add_argument('--preserve-body-weight', type=float, default=1.0)
    p.add_argument('--body-preservation-policy', choices=['legacy', 'bounded'], default='legacy')
    a = p.parse_args()
    code = Path(__file__).resolve().parent
    root = code.parent
    workspace_root = Path(os.environ.get('RESEARCH_ROOT', root.parent.parent)).resolve()
    cfg = json.loads(a.config.read_text())
    if a.seed is not None:
        cfg['seed'] = a.seed
    if a.case:
        cfg['case'] = a.case
    if a.gpu:
        cfg['physical_gpu'] = int(a.gpu)
    if cfg['physical_gpu'] not in (4, 5, 6, 7):
        p.error('Use GPU4--7')
    calibration = _resolve_calibration(a.calibration, cfg, a.config)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(code, out/'code_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    config = out/'config.json'
    config.write_text(json.dumps(cfg, indent=2)+'\n')
    ardy = Path(os.environ.get('ARDY_REPO', root / 'workspace/fuyuhan/ardy')).resolve()
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
    if run('prepare', [sys.executable, str(code/'prepare_conditions.py'), '--out', str(source),
                       '--case', cfg['case'], '--seed', str(cfg['seed']), '--prompt', ' | '.join(cfg['prompts'])]):
        return 2
    if run('generate', [sys.executable, str(code/'generate_segmented.py'),
                        '--source', str(source), '--config', str(config)], ardy):
        return 2
    command = [sys.executable, str(code/'adapt_ardy.py'), '--source', str(source), '--out', str(out/'candidate'), '--config', str(config), '--adapter', a.adapter, '--hand-profile', a.hand_profile]
    if a.preserve_generated_body:
        command.append('--preserve-generated-body')
    command += ['--preserve-body-weight', str(a.preserve_body_weight), '--body-preservation-policy', a.body_preservation_policy]
    if calibration:
        command += ['--calibration', str(calibration)]
    if run('adapt', command):
        return 2
    if a.generate_only:
        return 0
    if run('replay', ['bash', str(root/'replay_isolated.sh'), str(workspace_root), str(out/'candidate'), str(out/'replay')]):
        return 2
    return run('verify', [sys.executable, str(code/'verify_run.py'), str(out/'replay')])


if __name__ == '__main__':
    raise SystemExit(main())

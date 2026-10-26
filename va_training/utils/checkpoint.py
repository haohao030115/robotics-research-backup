from pathlib import Path
import torch
from models import build_policy


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, config, stats,
                    resume_progress=None, rng_states=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'format_version': 2, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'global_step': step, 'config': config,
                'normalization_stats': stats, 'resume_progress': resume_progress,
                'rng_states': rng_states}, temporary)
    temporary.replace(path)


def load_policy(path, device):
    # Checkpoints contain tensors and JSON-compatible config/statistics only.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    policy = build_policy(checkpoint['config'], checkpoint['normalization_stats'])
    policy.load_state_dict(checkpoint['model'])
    return policy.to(device).eval(), checkpoint

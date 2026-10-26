"""Separate V2 trainer/evaluator, reusing source splits, runtime and checkpoint IO."""
import copy
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from datasets.episodes import split_episodes
from datasets.tokens import TokenDataset, load_token_episodes, validation_report, STATE_WIDTHS
from models import build_policy
from losses.token import token_loss, TokenMetrics
from sonic.release import bundle_provenance, require_pass, digest
from utils.checkpoint import save_checkpoint, load_policy
from utils.config import dump_json
from utils.runtime import initialize, cleanup


def dataset_for(episodes, config):
    m = config['model']
    return TokenDataset(episodes, m['image_size'], m['history_length'], m['token_chunk_size'], config['data']['target_offset'])


def statistics(dataset, minimum):
    totals = {}
    for sample in dataset:
        values = {'image': sample['image'].permute(1, 2, 0).reshape(-1, 3),
                  'hand': sample['target_hand_ref'].reshape(-1, 14),
                  **{name: value[sample['history_mask']] for name, value in sample['state'].items()}}
        for name, value in values.items():
            value = value.double()
            total, square, count = totals.get(name, (torch.zeros(value.shape[-1], dtype=torch.float64),
                                                     torch.zeros(value.shape[-1], dtype=torch.float64), 0))
            totals[name] = total+value.sum(0), square+value.square().sum(0), count+len(value)
    result = {'state': {}, 'train_episode_ids': [e.id for e in dataset.episodes]}
    for name, (total, square, count) in totals.items():
        mean = total/count; std = (square/count-mean.square()).clamp_min(0).sqrt().clamp_min(minimum)
        entry = {'mean': mean.float().tolist(), 'std': std.float().tolist()}
        if name in STATE_WIDTHS:
            result['state'][name] = entry
        else:
            result[name] = entry
    return result


def move(batch, device):
    return (batch['image'].to(device), {k: v.to(device) for k, v in batch['state'].items()},
            batch['history_mask'].to(device), batch['target_sonic_token'].to(device), batch['target_hand_ref'].to(device))


@torch.inference_mode()
def evaluate_tokens(policy, loader, device):
    previous = policy.training; policy.eval(); metrics = TokenMetrics()
    try:
        for batch in loader:
            image, state, mask, token, hand = move(batch, device)
            metrics.add(policy(image, state, mask), token, hand)
        return metrics.report()
    finally:
        policy.train(previous)


def train_tokens(config, oracle_report, stage='overfit', overfit_report=None):
    config = copy.deepcopy(config)
    training = config['training']; runtime = initialize(training['device'], training['seed'])
    try:
        if runtime.distributed:
            raise ValueError('Initial V2 trainer supports single-device training; V1 DDP remains available')
        sonic = bundle_provenance(config['sonic'])
        oracle = require_pass(oracle_report, 'oracle_pass', sonic)
        if min(training['epochs'], training['max_steps'], training['batch_size_per_rank']) < 1:
            raise ValueError('Training limits must be positive')
        if stage == 'heldout':
            if overfit_report is None:
                raise ValueError('Stage B requires a passing Stage A report')
            require_pass(overfit_report, 'overfit_pass', sonic)
        if stage == 'overfit':
            if len(config['data']['episode_ids']) != 1:
                raise ValueError('Stage A requires exactly one episode ID')
            if oracle['episode'] != config['data']['episode_ids'][0]:
                raise ValueError('Stage A requires an oracle pass on the same episode')
            config['data']['val_fraction'] = 0
        episodes = load_token_episodes(config['data'])
        train, val = split_episodes(episodes, config['data'])
        if stage == 'heldout' and not val:
            raise ValueError('Stage B requires held-out source episodes')
        dataset, validation = dataset_for(train, config), dataset_for(val, config)
        out = Path(training['output_dir']); out.mkdir(parents=True, exist_ok=False)
        audit = validation_report(episodes, dataset_for(episodes, config))
        dump_json(out/'dataset_validation.json', audit)
        provenance = {'va_config': config, 'sonic': sonic, 'dataset_manifest': config['data']['manifests'],
            'episodes': [e.provenance for e in episodes], 'body_joint_order': episodes[0].body_order,
            'hand_joint_order': episodes[0].hand_order, 'token_label_source': 'selected replay native token_state.csv',
            'oracle_report': str(Path(oracle_report).resolve()), 'oracle_report_sha256': digest(oracle_report)}
        dump_json(out/'provenance.json', provenance); dump_json(out/'config.resolved.json', config)
        split = {'train': [e.id for e in train], 'val': [e.id for e in val]}
        dump_json(out/'split.json', split)
        stats = statistics(dataset, training['normalization_min_std']); dump_json(out/'normalization.json', stats)
        # Preprocess each selected RGB once. Random batches must not repeatedly
        # decompress a whole episode archive for every observation.
        samples = [dataset[i] for i in range(len(dataset))]
        v_samples = [validation[i] for i in range(len(validation))]
        batch_size = training['batch_size_per_rank']
        generator = torch.Generator().manual_seed(training['seed'])
        loader = DataLoader(samples, batch_size=batch_size, shuffle=True, generator=generator,
                            num_workers=training['num_workers'])
        eval_train = DataLoader(samples, batch_size=batch_size, shuffle=False)
        eval_val = DataLoader(v_samples, batch_size=batch_size, shuffle=False)
        policy = build_policy(config, stats).to(runtime.device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=training['lr'], weight_decay=training['weight_decay'])
        steps = min(training['max_steps'], training['epochs']*len(loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        initial = evaluate_tokens(policy, eval_train, runtime.device)
        dump_json(out/'initial_metrics.json', initial)
        step = 0; started = time.monotonic()
        (out/'logs').mkdir()
        with (out/'logs/metrics.jsonl').open('w') as log:
            for epoch in range(training['epochs']):
                policy.train()
                for batch in loader:
                    image, state, mask, token, hand = move(batch, runtime.device)
                    optimizer.zero_grad(set_to_none=True)
                    output = policy(image, state, mask)
                    loss, parts = token_loss(output, token, hand, training)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite V2 training loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), training['grad_clip'])
                    optimizer.step(); scheduler.step(); step += 1
                    if step == 1 or step % training['log_every'] == 0:
                        record = {'step': step, 'epoch': epoch, 'loss': float(loss.detach()),
                                  **{k: float(v.detach()) for k, v in parts.items()}}
                        log.write(json.dumps(record)+'\n'); log.flush()
                        if step == 1 or step % (training['log_every']*10) == 0:
                            print(record, flush=True)
                    if step >= steps:
                        break
                if step >= steps:
                    break
        checkpoint = out/'checkpoints/latest.pt'
        save_checkpoint(checkpoint, policy, optimizer, scheduler, epoch, step, config, stats)
        # Include provenance inside the checkpoint as well as adjacent outputs.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        saved['provenance'] = provenance
        temporary = checkpoint.with_suffix('.tmp'); torch.save(saved, temporary); temporary.replace(checkpoint)
        train_metrics = evaluate_tokens(policy, eval_train, runtime.device)
        val_metrics = evaluate_tokens(policy, eval_val, runtime.device)
        overfit = (train_metrics['token_mae'] <= training['overfit_token_mae_max'] and
                   train_metrics['fsq_level_accuracy'] >= training['overfit_fsq_accuracy_min'] and
                   train_metrics['hand_mae'] <= training['overfit_hand_mae_max'])
        heldout = (val_metrics.get('samples', 0) > 0 and
                   val_metrics['token_mae'] <= training['heldout_token_mae_max'] and
                   val_metrics['fsq_level_accuracy'] >= training['heldout_fsq_accuracy_min'] and
                   val_metrics['hand_mae'] <= training['heldout_hand_mae_max'])
        report = {'stage': stage, 'initial_train': initial, 'train': train_metrics, 'val': val_metrics,
            'overfit_pass': bool(stage == 'overfit' and overfit),
            'offline_evaluation_pass': bool(stage == 'heldout' and heldout),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint), 'steps': step,
            'seconds': time.monotonic()-started, 'split': split, 'provenance': provenance,
            'heldout_generalization_claim': False}
        dump_json(out/'report.json', report)
        print({k: report[k] for k in ('stage', 'steps', 'overfit_pass', 'offline_evaluation_pass')},
              {k: train_metrics[k] for k in ('token_mae', 'fsq_level_accuracy', 'hand_mae')}, flush=True)
        return report
    finally:
        cleanup(runtime)

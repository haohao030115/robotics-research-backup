import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_episodes, split_episodes, EpisodeDataset, compute_statistics
from models import build_policy
from losses.action import action_loss
from utils.runtime import initialize, cleanup
from utils.config import dump_json
from utils.checkpoint import save_checkpoint
from utils.distributed import wrap_ddp
from utils.rng import capture_rng, restore_rng
from .evaluation import Metrics, evaluate


def run_training(config, resume=None, stop_after_steps=None):
    training, model_config = config['training'], config['model']
    runtime = initialize(training['device'], training['seed'])
    try:
        episodes = load_episodes(config['data'])
        train_episodes, val_episodes = split_episodes(episodes, config['data'])
        train_data = EpisodeDataset(train_episodes, model_config['horizon'], model_config['image_size'])
        val_data = EpisodeDataset(val_episodes, model_config['horizon'], model_config['image_size'])
        directory = Path(training['output_dir'])
        latest = directory/'checkpoints/latest.pt'
        if latest.exists() and resume is None:
            raise ValueError('output already contains a checkpoint; choose a new output or --resume')
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
        if checkpoint is not None:
            for field in ('model_type', 'model', 'data'):
                if config[field] != checkpoint['config'][field]:
                    raise ValueError(f'resume {field} configuration differs from checkpoint')
            stats = checkpoint['normalization_stats']
        else:
            if runtime.rank == 0:
                stats = compute_statistics(train_data, training['normalization_min_std'])
                dump_json(directory/'normalization.json', stats)
            if runtime.distributed:
                dist.barrier()
            stats = json.loads((directory/'normalization.json').read_text())
        if runtime.rank == 0:
            dump_json(directory/'config.resolved.json', config)
            dump_json(directory/'split.json', {'train': [e.id for e in train_episodes],
                                             'val': [e.id for e in val_episodes]})
            if checkpoint:
                dump_json(directory/'normalization.json', stats)
        sampler = DistributedSampler(train_data, runtime.world_size, runtime.rank, shuffle=True,
                                     seed=training['seed'])
        loader_generator = torch.Generator()
        loader = DataLoader(train_data, batch_size=training['batch_size_per_rank'],
                            sampler=sampler, shuffle=False, num_workers=training['num_workers'],
                            generator=loader_generator,
                            pin_memory=False, drop_last=False)
        # No padding/duplication in validation; use the unwrapped model to avoid DDP
        # buffer-broadcast collectives on ranks with different validation lengths.
        validation = DataLoader(val_data, batch_size=training['batch_size_per_rank'],
            sampler=list(range(runtime.rank, len(val_data), runtime.world_size)),
            num_workers=training['num_workers'], pin_memory=False,
            generator=torch.Generator().manual_seed(training['seed']))
        policy = build_policy(config, stats).to(runtime.device)
        if checkpoint:
            policy.load_state_dict(checkpoint['model'])
        network = wrap_ddp(policy, runtime, training.get('disable_musa_ddp_runtime_timing', True))
        optimizer = torch.optim.AdamW(network.parameters(), lr=training['lr'],
                                     weight_decay=training['weight_decay'])
        planned = min(training['max_steps'], training['epochs']*len(loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(planned, 1))
        start, step, start_batch = 0, 0, 0
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start, step = checkpoint['epoch']+1, checkpoint['global_step']
            progress = checkpoint.get('resume_progress')
            if progress:
                if progress['world_size'] != runtime.world_size:
                    raise ValueError('exact resume requires the same world size')
                for field in ('seed', 'batch_size_per_rank', 'epochs', 'max_steps', 'lr',
                              'weight_decay', 'grad_clip', 'body_loss_weight', 'hand_loss_weight'):
                    if training[field] != checkpoint['config']['training'][field]:
                        raise ValueError(f'exact resume requires unchanged training.{field}')
                start, start_batch = progress['epoch'], progress['batch']
                restore_rng(checkpoint['rng_states'][runtime.rank], runtime)
        if step >= training['max_steps'] or start >= training['epochs']:
            raise ValueError('resume has no remaining epochs/steps; increase configured limits')
        log = directory/'logs/metrics.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(start, training['epochs']):
            sampler.set_epoch(epoch)
            loader_generator.manual_seed(training['seed']+epoch)
            network.train()
            epoch_metrics = Metrics()
            processed = 0
            stop_limit = min(training['max_steps'], stop_after_steps) if stop_after_steps else training['max_steps']
            if stop_limit <= step:
                raise ValueError('stop-after-steps must exceed the checkpoint global step')
            for batch_number, batch in enumerate(loader):
                processed = batch_number+1
                if epoch == start and batch_number < start_batch:
                    continue
                image, q, target = [batch[k].to(runtime.device) for k in ['image', 'q', 'target']]
                normalized = policy.normalization.action(target)
                optimizer.zero_grad(set_to_none=True)
                if config['model_type'] == 'flow':
                    xt, t, velocity = policy.scheduler.training_pair(normalized)
                    predicted = network(image, q, xt, t, batch['instruction'])
                    loss, parts = action_loss(predicted, velocity, 'mse',
                        training['body_loss_weight'], training['hand_loss_weight'])
                    # Diagnostic endpoint approximation; distinct from ODE-sampled validation L1.
                    output = xt + (1-t[:, None, None])*predicted
                    extra = {'flow_matching_loss': loss, 't_mean': t.mean(),
                             't_min': t.min(), 't_max': t.max(),
                             'predicted_velocity_norm': predicted.flatten(1).norm(dim=1).mean(),
                             'target_velocity_norm': velocity.flatten(1).norm(dim=1).mean()}
                    assert (t >= 0).all().item() and (t <= 1).all().item()
                else:
                    output = network(image, q, batch['instruction'])
                    loss, parts = action_loss(output, normalized, 'l1',
                        training['body_loss_weight'], training['hand_loss_weight'])
                    extra = {}
                # A shared failure flag prevents one rank leaving peers waiting in backward.
                finite = torch.isfinite(loss).to(dtype=torch.int32)
                if runtime.distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('non-finite loss on a training rank')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), training['grad_clip'])
                finite_grad = torch.isfinite(grad_norm).to(dtype=torch.int32)
                if runtime.distributed:
                    dist.all_reduce(finite_grad, op=dist.ReduceOp.MIN)
                if not finite_grad.item():
                    raise FloatingPointError('non-finite gradient norm')
                optimizer.step()
                scheduler.step()
                step += 1
                with torch.no_grad():
                    physical = policy.normalization.inverse_action(output)
                    _, errors = action_loss(physical, target)
                prefix = 'action_l1_proxy' if config['model_type'] == 'flow' else 'action_l1'
                values = {'loss': loss, prefix: errors['total'], 'body_l1': errors['body'],
                          'hand_l1': errors['hand'], 'grad_norm': grad_norm,
                          'lr': optimizer.param_groups[0]['lr'], **extra}
                epoch_metrics.add(values, len(q))
                if step == 1 or step % training['log_every'] == 0 or step == stop_limit:
                    current = Metrics()
                    current.add(values, len(q))
                    record = {'phase': 'train', 'epoch': epoch, 'global_step': step,
                              **current.result(runtime)}
                    if runtime.rank == 0:
                        with log.open('a') as handle:
                            handle.write(json.dumps(record)+'\n')
                        print(json.dumps(record), flush=True)
                if step >= stop_limit:
                    break
            summary = epoch_metrics.result(runtime)
            # Validation randomness must not perturb the training RNG stream.
            training_rng = capture_rng(runtime)
            val = evaluate(policy, validation, config, runtime)
            restore_rng(training_rng, runtime)
            progress = {'epoch': epoch+1 if processed == len(loader) else epoch,
                        'batch': 0 if processed == len(loader) else processed,
                        'world_size': runtime.world_size}
            rng_directory = directory/f'checkpoints/rng/step_{step:08d}'
            rng_directory.mkdir(parents=True, exist_ok=True)
            rng_path = rng_directory/f'rank_{runtime.rank}.pt'
            temporary = rng_path.with_suffix('.tmp')
            torch.save(training_rng, temporary)
            temporary.replace(rng_path)
            if runtime.distributed:
                dist.barrier()
            if runtime.rank == 0:
                record = {'phase': 'epoch', 'epoch': epoch, 'global_step': step,
                          'train': summary, 'val': val,
                          'val_status': 'available' if len(val_data) else 'no held-out episodes'}
                with log.open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
                states = [torch.load(rng_directory/f'rank_{r}.pt', map_location='cpu', weights_only=True)
                          for r in range(runtime.world_size)]
                save_checkpoint(latest, policy, optimizer, scheduler, epoch, step, config, stats, progress, states)
                if (epoch+1) % training['checkpoint_every_epochs'] == 0:
                    save_checkpoint(directory/f'checkpoints/epoch_{epoch:04d}.pt', policy,
                                    optimizer, scheduler, epoch, step, config, stats, progress, states)
            if runtime.distributed:
                dist.barrier()
            if step >= stop_limit:
                break
        return {'global_step': step, 'checkpoint': str(latest)}
    finally:
        cleanup(runtime)

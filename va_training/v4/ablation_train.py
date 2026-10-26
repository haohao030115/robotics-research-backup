"""Train controlled V4 auxiliary and architecture ablations on train_dataset_v0.

This entry point deliberately keeps the data split and preprocessing fixed.  The
baseline state is used only as initialization; each output is immutable and
records the exact variant, loss weights, split, and code hashes.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

from .dataset import dump, CanonicalV4
from .model import policy_for_variant
from .losses import objectives


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def metrics(p, d):
    b = p['body_tokens'] - d['body_tokens']; h = p['hand_ref'] - d['hand_ref']
    snapped = (p['body_tokens'] * 16).round().clamp(-16, 15) / 16
    exact = snapped == d['body_tokens']
    return dict(token_mae=float(b.abs().mean()), token_mse=float(b.square().mean()),
                fsq_scalar_accuracy=float(exact.float().mean()),
                whole_token_exact_accuracy=float(exact.all(-1).float().mean()),
                hand_mae=float(h.abs().mean()),
                moving_right_hand_mae=float(h[:, :, 7:14].abs().mean()),
                phase_accuracy=float((p['phase_logits'].argmax(-1) == d['phase']).float().mean()),
                progress_mae=float((p['closure_progress'] - d['progress']).abs().mean()))


@torch.no_grad()
def predict(m, d, batch=256):
    m.eval(); out = {}
    for i in range(0, len(d['rgb']), batch):
        p = m(d['rgb'][i:i+batch], d['body'][i:i+batch], d['hand'][i:i+batch])
        for k, v in p.items(): out.setdefault(k, []).append(v.detach())
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--prior', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--variant', choices=['cross_attention', 'mlp_fusion', 'proprio_only'], default='cross_attention')
    ap.add_argument('--aux', choices=['all', 'no_phase', 'no_progress', 'no_phase_progress'], default='all')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--canonical-manifest', required=True)
    ap.add_argument('--device', default='musa:0')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    dev = args.device; torch.set_num_threads(2); torch.manual_seed(4); np.random.seed(4)
    if dev.startswith('musa'):
        import torch_musa  # noqa: F401
        torch.musa.manual_seed_all(4)
    droot = Path(args.dataset)
    # Resolve the full canonical episode now so native execution cannot be
    # started with a compact tensor-only manifest by accident.
    canonical = CanonicalV4(args.canonical_manifest)
    with np.load(droot/'samples.npz', allow_pickle=False) as z: a = {k: z[k] for k in z.files}
    split = json.loads((droot/'split.json').read_text())
    rows = [json.loads(x) for x in (droot/'index.jsonl').read_text().splitlines() if x]
    tr_ids, va_ids = set(split['train_episodes']), set(split['val_episodes'])
    tr = np.array([i for i, r in enumerate(rows) if r['episode_id'] in tr_ids], dtype=np.int64)
    va = np.array([i for i, r in enumerate(rows) if r['episode_id'] in va_ids], dtype=np.int64)
    keys = ('rgb', 'body', 'hand', 'body_tokens', 'hand_ref', 'phase', 'progress')
    train = {k: torch.tensor(a[k][tr], device=dev) for k in keys}
    val = {k: torch.tensor(a[k][va], device=dev) for k in keys}
    model_variant = None if args.variant == 'cross_attention' else args.variant
    model = policy_for_variant(model_variant).to(dev)
    prior = torch.load(args.prior, map_location='cpu', weights_only=False)
    base = prior.get('model', prior)
    compatible = {k: v for k, v in base.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    with torch.no_grad():
        model.body_mean.copy_(train['body'].mean(0)); model.body_std.copy_(train['body'].std(0).clamp_min(.05))
        model.hand_mean.copy_(train['hand'].mean(0)); model.hand_std.copy_(train['hand'].std(0).clamp_min(.05))
    weights = [1., 1., 0.1, 0.1]
    if args.aux == 'no_phase': weights[2] = 0.
    elif args.aux == 'no_progress': weights[3] = 0.
    elif args.aux == 'no_phase_progress': weights[2] = weights[3] = 0.
    cfg = dict(variant=args.variant, auxiliary_ablation=args.aux,
               loss_weights=dict(body=weights[0], hand=weights[1], phase=weights[2], progress=weights[3]),
               epochs=args.epochs, batch_size=32, learning_rate=3e-4, seed=4,
               dataset=str(droot.resolve()), dataset_manifest_sha256=sha(droot/'manifest.json'),
               split_sha256=sha(droot/'split.json'), prior=str(Path(args.prior).resolve()),
               prior_sha256=sha(args.prior), canonical_manifest=str(Path(args.canonical_manifest).resolve()))
    (out/'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (out/'code_hashes.json').write_text(json.dumps({
        'model.py': sha(Path(__file__).with_name('model.py')), 'losses.py': sha(Path(__file__).with_name('losses.py')),
        'ablation_train.py': sha(Path(__file__))}, indent=2) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)
    history=[]; best=float('inf'); best_state=None; start=time.time()
    with (out/'training.jsonl').open('w') as log:
        for ep in range(1, args.epochs + 1):
            model.train(); order=torch.randperm(len(tr), device=dev); running={k:0. for k in ('body','hand','phase','progress')}
            for ix in order.split(32):
                opt.zero_grad(set_to_none=True); p=model(train['rgb'][ix], train['body'][ix], train['hand'][ix])
                loss, terms=objectives(p, {k:v[ix] for k,v in train.items()}, weights)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step()
                for k,v in terms.items(): running[k] += float(v.detach()) * len(ix) / len(order)
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                tp=predict(model, train); vp=predict(model, val); tm=metrics(tp, train); vm=metrics(vp, val)
                row=dict(epoch=ep, seconds=time.time()-start, optimization_losses=running, train_metrics=tm, val_metrics=vm)
                log.write(json.dumps(row) + '\n'); log.flush(); history.append(row); print(json.dumps(row), flush=True)
                score=tm['token_mae'] + tm['hand_mae'] + .1*tm['progress_mae']
                if score < best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_train=metrics(predict(model, train), train); final_val=metrics(predict(model, val), val)
    gate = (final_train['token_mae'] <= .01 and final_train['hand_mae'] <= .02 and
            final_train['moving_right_hand_mae'] <= .03 and final_train['progress_mae'] <= .03 and
            final_train['phase_accuracy'] >= .98 and final_train['fsq_scalar_accuracy'] >= .95)
    saved=dict(model=best_state, model_variant=model_variant, variant=args.variant, auxiliary_ablation=args.aux,
               config=cfg, epoch=history[-1]['epoch'], metrics=final_train, val_metrics=final_val,
               offline_gate=gate, manifest=str(Path(args.canonical_manifest).resolve()), provenance=canonical.provenance,
               manifest_purpose='Full canonical runtime episode used only for native trial provenance; tensors train from expert_datasets')
    torch.save(saved, out/'checkpoint.pt')
    # The native runner requires a sibling gate file; it is computed from actual predictions.
    dump(out/'offline_metrics.json', dict(**final_train, validation=final_val, offline_gate=gate,
        variant=args.variant, auxiliary_ablation=args.aux, source='expert_datasets/train_dataset_v0',
        split=dict(train_episodes=sorted(tr_ids), val_episodes=sorted(va_ids), train_samples=int(len(tr)), val_samples=int(len(va)))))
    (out/'training_status.json').write_text(json.dumps(dict(finished=True, epochs=args.epochs, elapsed_seconds=time.time()-start,
        checkpoint=str(out/'checkpoint.pt'), offline_gate=saved['offline_gate'], train_samples=int(len(tr)), val_samples=int(len(va))), indent=2) + '\n')
    np.savez_compressed(out/'offline_predictions.npz', **{k:v.detach().cpu().numpy() for k,v in predict(model, val).items()})


if __name__ == '__main__': main()

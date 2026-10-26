#!/usr/bin/env python3
"""Full manifest audit plus Dataset alignment, tail-padding and episode split checks."""
import argparse,json
from pathlib import Path
import numpy as np
from verify_expert_episode import verify
from episode_schema import validate_dataset_metadata,validate_images
from vision_action_dataset import VisionActionDataset
from expert_trajectory import write_json


def audit(manifest,horizon=50):
    manifest=Path(manifest).resolve();info=json.loads(manifest.read_text());rows=[];constants=None;pose=None;environment=None;fingerprints_checked=0
    if not info['episodes']:raise ValueError('No training-eligible episodes to audit')
    production_ledger_checked=False
    plan_path=manifest.parent/'sampling_plan.json';results_path=manifest.parent/'results.json'
    if plan_path.exists() and results_path.exists():
        plan=json.loads(plan_path.read_text());results=json.loads(results_path.read_text());cfg=plan['configuration']
        assert set(results)=={sample['case_id'] for sample in plan['samples']}
        for sample in plan['samples']:
            row=results[sample['case_id']]
            assert row['xy']==sample['xy'] and row['sample_index']==sample['sample_index']
            assert row['replay_attempts']<=cfg['max_replays'] and row['correction_iterations']<=cfg['max_corrections']
        assert {entry['episode_id'] for entry in info['episodes']}=={r['episode_id'] for r in results.values() if r.get('training_eligible')}
        if cfg['mode']=='random':
            expected=np.random.default_rng(cfg['seed']).uniform([cfg['x_min'],cfg['y_min']],[cfg['x_max'],cfg['y_max']],size=(cfg['num_episodes'],2))
            assert np.array_equal(expected,np.asarray([sample['xy'] for sample in plan['samples']]))
        production_ledger_checked=True

    for e in info['episodes']:
        ep=manifest.parent/e['episode_path'];meta=json.loads((ep/'metadata.json').read_text())
        validate_dataset_metadata(meta);checked=verify(ep);assert checked['expert_valid']
        assert meta['selected_validation_run']==e['selected_validation_rollout']
        assert meta['selected_candidate']==e['selected_candidate']
        task=meta['task_config'];assert np.array_equal(e['cube_initial_xy'],task['cube_initial_position'][:2])
        fixed={k:task[k] for k in ['object_name','cube_size','cube_mass','table_height']}
        if constants is None:constants=fixed
        assert constants==fixed
        if meta.get('environment_fingerprint') is not None:
            if environment is None:environment=meta['environment_fingerprint']
            assert environment==meta['environment_fingerprint']
            fingerprints_checked+=1
        initial=np.load(ep/'source/qpos.npy')[0]
        # Cube free-joint coordinates may vary; use named body/hand arrays for neutral equivalence.
        neutral=np.r_[np.load(ep/'source/body_q.npy')[0],np.load(ep/'source/hand_q.npy')[0]]
        if pose is None:pose=neutral
        assert np.array_equal(pose,neutral)
        ts=np.load(ep/'timestamps.npy');assert len(ts)==e['state_reference_length']
        assert np.allclose(np.diff(ts),.02,atol=1e-9,rtol=0)
        assert e['rgb_reference_frame_indices']==list(range(0,len(ts),5))
        image=validate_images(ep/'validation'/e['selected_validation_rollout'],ts)
        for key,width in [('body_q',29),('body_dq',29),('body_ref_q',29),('body_ref_dq',29),('hand_q',14),('hand_dq',14),('hand_ref_q',14),('hand_ref_dq',14)]:
            values=np.load(ep/f'{key}.npy');assert values.shape==(len(ts),width) and np.isfinite(values).all()
        rows.append(dict(episode_id=e['episode_id'],passed=True,images=image['frames']))
    full=VisionActionDataset(manifest,horizon=horizon);train=VisionActionDataset(manifest,horizon=horizon,split='train');val=VisionActionDataset(manifest,horizon=horizon,split='validation')
    assert {e['episode_id'] for e in train.episodes}.isdisjoint(e['episode_id'] for e in val.episodes)
    assert len(train)+len(val)==len(full)
    # Check each episode's first and last RGB-aligned samples against the original arrays.
    for i,e in enumerate(full.episodes):
        indexes=[j for j,(ei,_,_) in enumerate(full.samples) if ei==i]
        for j in [indexes[0],indexes[-1]]:
            sample=full[j];frame=sample['reference_frame'];ep=manifest.parent/e['episode_path'];length=e['state_reference_length']
            ix=np.minimum(np.arange(frame,frame+horizon),length-1)
            for key in ['body_ref_q','hand_ref_q']:assert np.array_equal(sample['target'][key],np.load(ep/f'{key}.npy')[ix].astype(np.float32))
            assert int(sample['target']['action_mask'].sum())==min(horizon,length-frame)
            for key in ['body_q','hand_q']:assert np.array_equal(sample['input'][key],np.load(ep/f'{key}.npy')[frame].astype(np.float32))
            assert set(sample['input'])=={'head_rgb','body_q','hand_q','instruction'}
    drop=VisionActionDataset(manifest,horizon=horizon,tail='drop')
    assert all(f+horizon<=drop.episodes[i]['state_reference_length'] for i,_,f in drop.samples)
    write_json(manifest.parent/'splits.json',full.splits)
    result=dict(passed=True,schema_version=3,production_ledger_and_seed_checked=production_ledger_checked,episodes=len(rows),samples=len(full),train_samples=len(train),validation_samples=len(val),horizon=horizon,
        common_neutral_pose=True,fixed_task_geometry=True,environment_fingerprints_checked=fingerprints_checked,
        full_environment_fingerprint_available=fingerprints_checked==len(rows),no_episode_split_leakage=True,tail_padding_and_mask_checked=True,rows=rows)
    write_json(manifest.parent/'dataset_audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--horizon',type=int,default=50);a=p.parse_args()
    print(json.dumps(audit(a.manifest,a.horizon),indent=2))

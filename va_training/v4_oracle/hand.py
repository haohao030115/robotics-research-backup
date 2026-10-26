"""Success-conditioned geometry envelope and exact one-shot expert hand replay."""
import json
from pathlib import Path
import numpy as np
from va_training.v4.dataset import read, dump, sha


def conjugate(q):
    q=np.asarray(q,float).copy();q[...,1:]*=-1;return q


def multiply(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.concatenate([a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True),
        a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])],axis=-1)


def relative_pose(wrist,cube):
    wrist,cube=np.asarray(wrist),np.asarray(cube)
    w=wrist[...,3:]/np.linalg.norm(wrist[...,3:],axis=-1,keepdims=True)
    c=cube[...,3:]/np.linalg.norm(cube[...,3:],axis=-1,keepdims=True)
    inv=conjugate(c)
    v=np.concatenate([np.zeros_like(wrist[...,:1]),wrist[...,:3]-cube[...,:3]],axis=-1)
    return multiply(multiply(inv,v),c)[...,1:],multiply(inv,w)


def angle(a,b):
    return 2*np.arccos(np.clip(np.abs(np.sum(np.asarray(a)*b,axis=-1)),0,1))


def closure_rows(h):
    initial=h[0];delta=h[np.argmax(np.linalg.norm(h-initial,axis=1))]-initial
    if not np.any(np.abs(delta)>.05):raise ValueError('No closing joints')
    progress=((h-initial)*delta).sum(-1)/(delta@delta)
    onset=int(np.flatnonzero(progress>1e-6)[0])
    hold=int(np.flatnonzero(progress>=1-1e-6)[0])
    # No later release/reset rows enter the calibrated onset or replay segment.
    early=np.flatnonzero((progress>1e-6)&(progress<=.05)&(np.arange(len(h))<=hold))
    return onset,hold,early,delta


def calibrate(dataset,template_manifest,output):
    """Fix all thresholds before reading learner data. No guessed fallback."""
    m=read(Path(dataset)/'manifest.json');template=Path(template_manifest).parent
    case=read(template/'canonical_replay/metadata.json')['runtime']['case']
    xyz=[];quats=[];sources=[]
    for e in m['episodes']:
        if not e['episode_id'].startswith('kimodo_'):continue
        root=Path(e['source_manifest']).parent
        if read(root/'canonical_replay/metadata.json')['runtime']['case']!=case:continue
        valid=read(root/'validation/dataset_validation.json')
        success=read(root/'validation/replay_validation.json')
        if not (valid['expert_valid'] and valid['physical_success'] and success['physical_success']):continue
        with np.load(root/'canonical_replay/references.npz') as z:refs={k:z[k] for k in z.files}
        with np.load(root/'canonical_replay/observations.npz') as z:obs={k:z[k] for k in ('observation_id','right_wrist_pose','cube_pose')}
        onset,hold,early,delta=closure_rows(refs['hand_ref_q'])
        lookup={str(k):i for i,k in enumerate(obs['observation_id'])}
        rows=[lookup[str(refs['observation_id'][i])] for i in early]
        pos,quat=relative_pose(obs['right_wrist_pose'][rows],obs['cube_pose'][rows]);xyz.extend(pos);quats.extend(quat)
        sources.append(dict(episode_id=e['episode_id'],manifest=str(root/'training_manifest.json'),
            hashes={str(p.relative_to(root)):sha(p) for p in [root/'canonical_replay/references.npz',root/'canonical_replay/observations.npz',root/'validation/dataset_validation.json',root/'validation/replay_validation.json']},
            onset_reference_row=onset,hold_reference_row=hold,early_reference_rows=early.tolist(),
            early_observation_ids=refs['observation_id'][early].tolist(),max_lift_m=success['cube_max_lift'],hold_s=success['qualifying_hold_duration']))
    if not sources:raise ValueError('No accepted successful same-task expert')
    xyz=np.array(xyz);quats=np.array(quats)
    # Markley quaternion mean. Angular bound and XYZ bounds are observed maxima/minima.
    _,v=np.linalg.eigh(quats.T@quats);center=v[:,-1];maxangle=float(angle(quats,center).max())
    with np.load(template/'canonical_replay/references.npz') as z:refs={k:z[k] for k in z.files}
    onset,hold,early,direction=closure_rows(refs['hand_ref_q'])
    trajectory=refs['hand_ref_q'][onset-1:hold+1].copy()
    signed=np.diff(trajectory,axis=0)*np.sign(direction)
    if np.any(signed[:,np.abs(direction)>.05]<-1e-6):raise ValueError('Expert closure contains reopening; cannot replay as one-shot closure')
    if not any(s['manifest']==str(Path(template_manifest)) for s in sources):raise ValueError('Template is not in accepted successful sources')
    cfg=dict(privileged_geometry_oracle=True,deployable=False,case=case,expert_sources=sources,
        calibration_rule='Cube-frame wrist XYZ componentwise observed min/max; quaternion geodesic distance to Markley mean <= observed maximum. Samples are exact current observations at expert progress (1e-6, .05] before first full closure. No learner tuning or clock gate.',
        envelope_limitation='Empirical successful-onset support, not an exhaustive characterization of all graspable poses.',
        relative_xyz_min=xyz.min(0).tolist(),relative_xyz_max=xyz.max(0).tolist(),relative_quaternion_center_wxyz=center.tolist(),orientation_limit_rad=maxangle,
        template_manifest=str(template_manifest),onset_reference_row=onset,closed_reference_row=hold,
        trajectory_reference_ids=refs['hand_ref_id'][onset-1:hold+1].tolist(),control_hz=50,
        closure_duration_s=(len(trajectory)-1)/50,peak_velocity_rad_s=(np.abs(np.diff(trajectory,axis=0))*50).max(0).tolist(),
        open_reference=trajectory[0].tolist(),hold_reference=trajectory[-1].tolist(),direction=direction.tolist(),
        terminal_release_excluded=True,network_hand_and_phase_ignored=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);dump(out/'oracle_calibration.json',cfg)
    np.savez_compressed(out/'expert_hand.npz',trajectory=trajectory,relative_xyz=xyz,relative_quaternion=quats)
    return cfg


class OracleHand:
    def __init__(self,calibration):
        self.path=Path(calibration);self.config=read(self.path/'oracle_calibration.json')
        with np.load(self.path/'expert_hand.npz') as z:self.trajectory=z['trajectory'].copy()
        self.trigger_tick=None;self.last_tick=None;self.log=[]
    def step(self,tick,wrist,cube):
        if self.last_tick is not None and tick!=self.last_tick+1:raise ValueError('Oracle control tick discontinuity')
        self.last_tick=tick;c=self.config;pos,q=relative_pose(wrist,cube)
        low=np.asarray(c['relative_xyz_min']);high=np.asarray(c['relative_xyz_max'])
        margin=np.minimum(pos-low,high-pos);err=float(angle(q,c['relative_quaternion_center_wxyz']))
        position_ok=bool((margin>=-1e-12).all());orientation_ok=bool(err<=c['orientation_limit_rad']+1e-12)
        ready=position_ok and orientation_ok
        event='observe'
        if self.trigger_tick is None and ready:self.trigger_tick=int(tick);event='trigger'
        slot=0 if self.trigger_tick is None else min(tick-self.trigger_tick,len(self.trajectory)-1)
        state='OPEN' if self.trigger_tick is None else 'HOLD' if slot==len(self.trajectory)-1 else 'CLOSING'
        target=self.trajectory[slot].copy()
        self.log.append(dict(tick=int(tick),event=event,state=state,ready=ready,position_ok=position_ok,orientation_ok=orientation_ok,
            relative_xyz=pos.tolist(),relative_quaternion_wxyz=q.tolist(),position_margin_xyz_m=margin.tolist(),
            orientation_error_rad=err,orientation_limit_rad=c['orientation_limit_rad'],trigger_tick=self.trigger_tick,trajectory_slot=int(slot),
            wrist_pose=np.asarray(wrist).tolist(),cube_pose=np.asarray(cube).tolist(),applied_hand_reference=target.tolist()))
        return target
    def report(self):
        return dict(privileged_geometry_oracle=True,deployable=False,trigger_tick=self.trigger_tick,
            readiness_not_reached=self.trigger_tick is None,closure_completed=any(r['state']=='HOLD' for r in self.log),
            position_ready_ticks=sum(r['position_ok'] for r in self.log),orientation_ready_ticks=sum(r['orientation_ok'] for r in self.log),
            jointly_ready_ticks=sum(r['ready'] for r in self.log))

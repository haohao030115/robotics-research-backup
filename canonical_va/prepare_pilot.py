"""One-time explicit migration of the selected successful pilot, not a producer replay path."""
from pathlib import Path
import numpy as np
from .common import ROOT, read, load_npz, dump, digest
from .schema import joint_orders, UNITS, validate


def main():
    src=ROOT/'kimodo-t6/g1-integration/task8_development_20261007_v3/accepted_touch_allowed_profile/qualification'
    success=read(src/'result.json')
    if success['expert_valid'] is not True: raise ValueError('Pilot source lacks success evidence')
    out=ROOT/'canonical_va/pilot_reference/pilot_0000'; out.mkdir(parents=True)
    old=load_npz(src/'input_reference.npz'); spec=read(src/'candidate/setting.json'); bn,hn=joint_orders()
    a={k:old[k] for k in UNITS if k!='time_s'}; a['time_s']=old['timestamps']
    np.savez_compressed(out/'reference.npz',**a)
    dump(out/'metadata.json',dict(episode_id='pilot_0000',producer_type='kimodo',reference_hz=50,reference_complete=True,
        body_joint_names=bn,hand_joint_names=hn,quaternion_convention='wxyz',root_pose_frame='world',units=UNITS,case=spec['case']))
    dump(out/'provenance.json',dict(migration='Rename timestamps to time_s; preserve required reference values exactly; no motion tuning',
        source=str(src.relative_to(ROOT)),source_hashes={str((src/p).relative_to(ROOT)):digest(src/p) for p in ('input_reference.npz','result.json','candidate/setting.json')},
        historical_physical_success=success['physical_success'],historical_max_lift=success['max_lift_m'],
        historical_hold_seconds=success['max_continuous_hold_seconds'],note='Historical success is selection evidence only; new replay must pass independently'))
    dump(out/'reference_validation.json',{'reference_schema_valid':True,'source_complete':success['completed']})
    _,_,report=validate(out); dump(out/'reference_validation.json',report)
    print(out)

if __name__=='__main__': main()

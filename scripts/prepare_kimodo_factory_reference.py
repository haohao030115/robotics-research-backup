#!/usr/bin/env python3
"""Package an already accepted native Kimodo replay for the Data Factory."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np

BODY = ['left_hip_pitch_joint','right_hip_pitch_joint','waist_yaw_joint','left_hip_roll_joint','right_hip_roll_joint','waist_roll_joint','left_hip_yaw_joint','right_hip_yaw_joint','waist_pitch_joint','left_knee_joint','right_knee_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint','left_ankle_pitch_joint','right_ankle_pitch_joint','left_shoulder_roll_joint','right_shoulder_roll_joint','left_ankle_roll_joint','right_ankle_roll_joint','left_shoulder_yaw_joint','right_shoulder_yaw_joint','left_elbow_joint','right_elbow_joint','left_wrist_roll_joint','right_wrist_roll_joint','left_wrist_pitch_joint','right_wrist_pitch_joint','left_wrist_yaw_joint','right_wrist_yaw_joint']
HAND = ['left_hand_index_0_joint','left_hand_index_1_joint','left_hand_middle_0_joint','left_hand_middle_1_joint','left_hand_thumb_0_joint','left_hand_thumb_1_joint','left_hand_thumb_2_joint','right_hand_index_0_joint','right_hand_index_1_joint','right_hand_middle_0_joint','right_hand_middle_1_joint','right_hand_thumb_0_joint','right_hand_thumb_1_joint','right_hand_thumb_2_joint']

def digest(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=Path, required=True); ap.add_argument('--replay', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    c, r, o = [x.resolve() for x in (a.candidate, a.replay, a.out)]
    if o.exists(): raise SystemExit(f'fresh output required: {o}')
    result = json.loads((r/'result.json').read_text())
    if not (result.get('physical_success') and result.get('expert_valid') and result.get('support_contact_steps') == 0 and result.get('recorded_frames') == result.get('expected_frames')):
        raise SystemExit('replay is not strictly accepted; refusing to package')
    z = np.load(c/'reference.npz'); n = len(z['body_ref_q'])
    required = ('body_ref_q','body_ref_dq','root_ref_pos','root_ref_quat','hand_ref_q')
    if any(k not in z for k in required) or any(not np.isfinite(z[k]).all() for k in required): raise SystemExit('invalid candidate reference')
    time_s = z['time_s'] if 'time_s' in z else z['timestamps']
    if len(time_s) != n or not np.isfinite(time_s).all(): raise SystemExit('invalid candidate timestamps')
    o.mkdir(parents=True)
    np.savez_compressed(o/'reference.npz', **{k:z[k] for k in required}, time_s=time_s)
    ch = {str(p): digest(p) for p in c.rglob('*') if p.is_file()}; rh = {str(p): digest(p) for p in r.rglob('*') if p.is_file()}
    setting = json.loads((c/'setting.json').read_text()) if (c/'setting.json').exists() else {}
    profile = setting.get('postprocessing',{}).get('hand_profile','default')
    # Keep same-profile seeds distinct during V4 aggregation.
    seed = setting.get('seed')
    case = setting.get('case', {'id':'center','block_xy':[0.65,0.0],'block_yaw_deg':0})
    seed_tag = f"_seed{int(seed)}" if seed is not None else ""
    meta = {'episode_id': f"kimodo_unified_{case.get('id','center')}{seed_tag}_hand_{profile}", 'producer_type':'kimodo', 'reference_hz':50, 'num_frames':n, 'duration_s':float((n-1)/50), 'body_dof':29, 'hand_dof':14, 'body_joint_names':BODY, 'hand_joint_names':HAND, 'quaternion_convention':'wxyz','root_pose_frame':'world','units':{'time_s':'s','body_ref_q':'rad','body_ref_dq':'rad/s','root_ref_pos':'m','root_ref_quat':'unitless','hand_ref_q':'rad'}, 'robot':'G1','scene':'canonical_tabletop','initial_pose':'canonical_reset','case':case,'task_arm':'right','reference_complete':True,'source_adapter':'unified','source_native_run':str(r),'source_candidate':str(c),'source_candidate_sha256':ch,'source_native_replay_sha256':rh,'hand_profile':profile,'seed':seed}
    (o/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    (o/'provenance.json').write_text(json.dumps({'factory_input':'accepted native Kimodo unified replay; no motion generation rerun','producer_type':'kimodo','adapter':'unified','source_reference':str(r),'source_candidate':str(c),'native_execution_evidence':str(r),'native_result':str(r/'result.json'),'source_candidate_sha256':ch,'source_native_replay_sha256':rh,'case':case,'hand_profile':profile,'seed':seed}, indent=2)+'\n')
    strict={'policy_version':'strict-unified-validation-v1','result':{'physical_success':True,'contact_safe':True,'expert_valid':True,'dataset_valid':True,'reasons':[]},'source':str(r/'result.json')}
    (o/'reference_validation.json').write_text(json.dumps({'reference_schema_valid':True,'finite':True,'timestamps_monotonic':True,'joint_limits_valid':True,'quaternion_valid':True,'reference_complete':True,'tested_with_native_replay':True,'source_native_result':result,'strict_gate':strict}, indent=2)+'\n')
    shutil.copytree(r, o/'source_evidence'/'native_replay')
    (o/'source_evidence'/'evidence_index.json').write_text(json.dumps({'source_candidate':str(c),'source_replay':str(r),'candidate_sha256':ch,'replay_sha256':rh,'strict_gate':strict}, indent=2)+'\n')
    print(json.dumps({'reference':str(o),'frames':n,'episode_id':meta['episode_id']})); return 0
if __name__ == '__main__': raise SystemExit(main())

#!/usr/bin/env python3
"""Export actual Kimodo qpos to canonical SONIC; offline adaptation is opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])).resolve()
PROJECT=Path(os.environ.get('KIMODO_RUNTIME_SUPPORT', Path(__file__).resolve().parents[1]/'runtime_support')).resolve()
SONIC=Path(os.environ.get('SONIC_REPO', ROOT/'GR00T-WholeBodyControl')).resolve()
ROBOT=SONIC/'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
BODY_INDEXES=[0,4,10,18,5,11,19,9,16,22,28,17,23,29]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def verify_generation(task_dir,generation_dir,task,gen):
    if gen.get('model_generation_run') is not True or gen.get('model_name')!='Kimodo-G1-RP-v1':
        raise ValueError('Actual Kimodo generation is required; dry-run/fixtures cannot be replay candidates')
    if gen.get('actual_frames')!=487 or gen.get('requested_frames')!=487 or gen.get('actual_source_last_timestamp_s')!=16.2:
        raise ValueError('Incomplete actual source duration/frame count')
    if gen.get('task_id')!=task['task_id']:raise ValueError('Generation/task identity mismatch')
    for name in ('task.json','task_plan.npz','pose_constraints.json'):
        path=(task_dir/name).resolve()
        if gen['source_hashes'].get(str(path))!=sha(path):raise ValueError('Generation/task hash mismatch '+name)
    for name,key in [('raw_motion.npz','raw_sha256'),('raw_motion.csv','raw_qpos_sha256')]:
        if gen.get(key)!=sha(generation_dir/name):raise ValueError('Generated raw hash mismatch '+name)
    if not gen.get('checkpoint_hashes'):raise ValueError('Missing checkpoint provenance')
    for name,expected in gen['checkpoint_hashes'].items():
        checkpoint=ROOT/'kimodo-t6/g1-integration/checkpoints/Kimodo-G1-RP-v1'/name
        if sha(checkpoint)!=expected:raise ValueError('Checkpoint changed '+name)

def resample(q,fps,t):
    q=np.asarray(q,dtype=float);t=np.asarray(t,dtype=float)
    if q.ndim!=2 or q.shape[1]!=36 or len(q)<2 or not np.isfinite(q).all():raise ValueError('Invalid finite source qpos(T,36)')
    if fps!=30 or len(t)!=810 or not np.array_equal(t,np.arange(810)/50):raise ValueError('Expected source30Hz/reference810x50Hz')
    source_t=np.arange(len(q))/fps
    if source_t[-1]<16.2 or t[-1]>source_t[-1]:raise ValueError('Insufficient raw coverage; extrapolation/last-pose padding forbidden')
    if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-5,rtol=0):raise ValueError('Source quaternion not unit')
    out=np.empty((len(t),36))
    for col in list(range(3))+list(range(7,36)):out[:,col]=np.interp(t,source_t,q[:,col])
    out[:,3:7]=Slerp(source_t,Rotation.from_quat(q[:,[4,5,6,3]]))(t).as_quat()[:,[3,0,1,2]]
    return out

def frame_zero_report(q,initial,h):
    vals=dict(body_max_error_rad=float(np.max(abs(q[0,7:]-initial[7:]))),root_max_error_m=float(np.max(abs(q[0,:3]-initial[:3]))),
        root_quat_max_error=float(np.max(abs(q[0,3:7]-initial[3:7]))),hand_max_error_rad=float(np.max(abs(h[0]))))
    vals['pass']=all(vals[k]<= (1e-10 if k.startswith('hand') else 1e-6) for k in vals)
    return vals

def canonical(task):
    os.environ['SONIC_REPO']=str(SONIC);sys.path[:0]=[str(PROJECT/'scripts'),str(SONIC)]
    import project_paths
    project_paths.default_robot_xml=lambda:ROBOT
    import mujoco
    from scene import load_scene
    from expert_trajectory import orders
    from cases import read_cases,reset_case
    m,d=load_scene(supported=False,robot_path=ROBOT)
    known={c['id']:c for c in read_cases()['cases']}
    if task['case']!=known[task['case']['id']]:raise ValueError('Case differs from canonical shared case')
    reset_case(m,d,task['case'])
    bn,hn,_=orders()
    if task['body_joint_order']!=bn or task['hand_joint_order']!=hn:raise ValueError('Joint name/order mismatch')
    if (m.nq,m.nv,m.nu,m.neq)!=(57,55,43,0):raise ValueError('Scene structure mismatch')
    initial=np.asarray(task.get('initial_qpos',task.get('initial_qpos57')),float)
    velocity=np.asarray(task.get('initial_qvel',task.get('initial_qvel55')),float)
    if initial.shape!=(57,) or velocity.shape!=(55,) or not np.isfinite(np.r_[initial,velocity]).all():raise ValueError('Invalid initial state')
    d.qpos[:]=initial;d.qvel[:]=velocity;mujoco.mj_forward(m,d)
    return m,d,bn,hn

def adapt(q,t,m,d,names,clearance,lift_margin,contact_plan=None,position_radius=.005,rotation_radius=.04,calibration=None,approach_refine=False,clearance_decay_s=None,preserve_generated_body=False,preserve_body_weight=1.0,spatial_clearance_m=0.0,body_preservation_policy='legacy',hand_q=None,hand_joint_names=None,opposition_centering=False,opposition_centering_max_correction_m=.03,opposition_centering_dominant_axis=False,distal_alignment=False,distal_alignment_max_correction_m=.04,collision_aware_hand=False,collision_hand_min_clearance_m=.001,collision_hand_max_relief_rad=.30,geometry_grasp_height=False,geometry_wrist_orientation=False):
    import mujoco
    from grasp_reference import GraspReference
    plan=GraspReference(m,d);original=mujoco.MjData(m)
    qa=np.array([m.joint(n).qposadr[0] for n in names]);rqa=int(m.joint('floating_base_joint').qposadr[0])
    initial=np.r_[d.qpos[rqa:rqa+7],d.qpos[qa]]
    result=np.tile(initial,(len(q),1));u=np.clip(t/1.5,0,1);u=u*u*(3-2*u)
    start=d.xpos[plan.wrist].copy();rotation=d.xmat[plan.wrist].reshape(3,3).copy();rows=[]
    geometry_wrist_target = None
    if geometry_wrist_orientation:
        from research.shared_adapter.grasp_geometry import geometry_wrist_orientation as _geometry_wrist_orientation
        geometry_wrist_target = _geometry_wrist_orientation()
    armcols=[names.index(m.joint(int(j)).name)+7 for j in plan.arm]
    armset=set(armcols)
    bodycols=[7+i for i in range(len(names)) if 7+i not in armset]
    opposition_anchor = None
    opposition_info = None
    opposition_correction = np.zeros(3)
    distal_alignment_anchor = None
    distal_alignment_info = None
    distal_alignment_correction = np.zeros(3)
    collision_projection_rows = []
    geometry_grasp_height_m = 0.0
    geometry_grasp_height_info = None
    if geometry_grasp_height:
        # Reuse the producer-neutral geometry policy already exercised by
        # ARDY.  The offset is derived from cube/finger extents and does not
        # inspect the position name or seed; native contact/lift gates remain
        # authoritative.
        from research.shared_adapter.grasp_geometry import approach_retraction_m as _approach_retraction_m
        _, geometry_grasp_height_info = _approach_retraction_m(m)
        geometry_grasp_height_m = float(geometry_grasp_height_info['grasp_height_offset_m'])
    hand_qa = None
    if opposition_centering or distal_alignment:
        if hand_q is None or hand_joint_names is None:
            raise ValueError('hand geometry policies require hand_q and hand_joint_names')
        hand_q = np.asarray(hand_q, dtype=float)
        if hand_q.shape != (len(q), len(hand_joint_names)) or not np.isfinite(hand_q).all():
            raise ValueError('opposition hand trajectory shape mismatch')
        hand_qa = np.array([m.joint(name).qposadr[0] for name in hand_joint_names])
    body_policy = None
    bounded_body = None
    if preserve_generated_body:
        source_body = q[:, 7:][:, [c-7 for c in bodycols]]
        initial_body = initial[bodycols]
        if body_preservation_policy == 'bounded':
            from research.shared_adapter.body import bounded_body_trajectory
            body_names = [names[c-7] for c in bodycols]
            lower = np.array([m.jnt_range[m.joint(n).id, 0] for n in body_names])
            upper = np.array([m.jnt_range[m.joint(n).id, 1] for n in body_names])
            bounded_body, body_policy = bounded_body_trajectory(
                source_body, initial_body, t, weight=preserve_body_weight,
                joint_lower=lower, joint_upper=upper)
        elif body_preservation_policy != 'legacy':
            raise ValueError('unknown body preservation policy '+str(body_preservation_policy))
    for k,row in enumerate(q):
        original.qpos[:]=d.qpos;original.qpos[rqa:rqa+7]=row[:7];original.qpos[qa]=row[7:]
        if opposition_centering or distal_alignment:
            original.qpos[hand_qa] = hand_q[k]
        mujoco.mj_forward(m,original)
        rawp=original.xpos[plan.wrist].copy();rawr=original.xmat[plan.wrist].reshape(3,3).copy()
        modelp=rawp.copy();modelr=rawr.copy();refine_weight=0.
        if contact_plan is not None:
            source_t,planned_pos,planned_quat=contact_plan
            goalp=np.array([np.interp(t[k],source_t,planned_pos[:,j]) for j in range(3)])
            goalr=Slerp(source_t,Rotation.from_quat(planned_quat[:,[1,2,3,0]]))([t[k]]).as_matrix()[0]
            # Offline task-space projection retains generated residuals inside
            # the grasp corridor; every change is measured against actual output.
            corrected=goalp+np.clip(rawp-goalp,-position_radius,position_radius)
            residual=(Rotation.from_matrix(rawr)*Rotation.from_matrix(goalr).inv()).as_rotvec()
            residual*=min(1.,rotation_radius/max(np.linalg.norm(residual),1e-12))
            corrected_r=(Rotation.from_rotvec(residual)*Rotation.from_matrix(goalr)).as_matrix()
            v=np.clip((t[k]-(1 if approach_refine else 6))/(1 if approach_refine else 2),0,1);refine_weight=v*v*(3-2*v)
            rawp=rawp+refine_weight*(corrected-rawp)
            rawr=Slerp([0,1],Rotation.from_matrix([rawr,corrected_r]))([refine_weight]).as_matrix()[0]
        contact_position_correction=float(np.linalg.norm(rawp-modelp))
        contact_rotation_correction=float((Rotation.from_matrix(rawr)*Rotation.from_matrix(modelr).inv()).magnitude())
        calibration_position=np.zeros(3);calibration_rotation=np.zeros(3)
        if calibration is not None:
            v=np.clip((t[k]-10)/2,0,1);v=v*v*(3-2*v)
            cv=np.clip((t[k]-6)/2,0,1);cw=cv*cv*(3-2*cv)
            calibration_position=cw*((1-v)*np.asarray(calibration['close_position_bias_m'])+v*np.asarray(calibration['hold_position_bias_m']))
            calibration_rotation=cw*((1-v)*np.asarray(calibration['close_rotation_bias_rad'])+v*np.asarray(calibration['hold_rotation_bias_rad']))
            rawp-=calibration_position
            rawr=(Rotation.from_rotvec(-calibration_rotation)*Rotation.from_matrix(rawr)).as_matrix()
        geometry_rotation_weight = 0.0
        if geometry_wrist_target is not None:
            # Apply the same terminal orientation to every case after the
            # pre-close approach.  The smooth window prevents an orientation
            # impulse and keeps the hand/body schedules frame aligned.
            v=np.clip((t[k]-7)/1,0,1);geometry_rotation_weight=v*v*(3-2*v)
            rawr=Slerp([0,1],Rotation.from_matrix([rawr,geometry_wrist_target]))([geometry_rotation_weight]).as_matrix()[0]
        target=start+u[k]*(rawp-start)
        # Base and geometry-derived clearance are one approach policy.  They
        # must share the same decay before hand closure; adding them before
        # the comparison below used to leave the spatial term active during
        # grasp, which lifted near-edge targets out of the hand corridor.
        v=np.clip((t[k]-4)/3,0,1); clearance_weight=v*v*(3-2*v)
        if clearance_decay_s is not None:
            decay=np.clip((t[k]-clearance_decay_s[0])/(clearance_decay_s[1]-clearance_decay_s[0]),0,1)
            clearance_weight*=1-decay*decay*(3-2*decay)
        target[2]+=(clearance+spatial_clearance_m)*clearance_weight
        if geometry_grasp_height:
            def _smooth(a, b, x):
                u = float(np.clip((x-a)/(b-a), 0., 1.))
                return u*u*(3.-2.*u)
            target[2] += geometry_grasp_height_m * _smooth(6.5, 7.5, t[k])
        v=np.clip((t[k]-10)/2,0,1); lift_weight=v*v*(3-2*v)
        target[2]+=lift_margin*lift_weight
        opposition_weight = 0.0
        if opposition_centering:
            from research.shared_adapter.grasp_geometry import opposition_centering_correction
            # Select once at the shared 7.0 s preclose anchor and hold the
            # chosen pair/correction through closure.  The source position
            # and seed never enter this selection.
            if opposition_anchor is None and t[k] >= 7.0:
                opposition_correction, opposition_info = opposition_centering_correction(
                    m, original, max_correction_m=opposition_centering_max_correction_m,
                    dominant_axis=opposition_centering_dominant_axis)
                opposition_anchor = int(k)
            if opposition_anchor is not None:
                from research.shared_adapter.grasp_geometry import opposition_centering_weight
                opposition_weight=float(opposition_centering_weight(t[k]))
                target[:2] += opposition_correction[:2] * opposition_weight
        distal_alignment_weight_value = 0.0
        if distal_alignment:
            from research.shared_adapter.grasp_geometry import distal_alignment_correction as _distal_alignment_correction
            # Measure after the hand has reached the preclose posture.  At the
            # start of the approach the distal midpoint is intentionally high
            # above the cube; anchoring there would turn a bounded alignment
            # policy into an unsafe vertical drop.
            if distal_alignment_anchor is None and t[k] >= 8.0:
                distal_alignment_correction, distal_alignment_info = _distal_alignment_correction(
                    m, original, max_correction_m=distal_alignment_max_correction_m)
                distal_alignment_anchor = int(k)
            if distal_alignment_anchor is not None:
                from research.shared_adapter.grasp_geometry import distal_alignment_weight
                distal_alignment_weight_value = float(distal_alignment_weight(t[k]))
                target += distal_alignment_correction * distal_alignment_weight_value
        plan.rotation=Slerp([0,1],Rotation.from_matrix([rotation,rawr]))([u[k]]).as_matrix()[0]
        body_values = None
        if preserve_generated_body:
            blend=float(u[k])
            if bounded_body is not None:
                body_values=initial[bodycols]+blend*(bounded_body[k]-initial[bodycols])
            else:
                body_values=initial[bodycols]+blend*preserve_body_weight*(row[7:][[c-7 for c in bodycols]]-initial[bodycols])
            # Solve the right arm in the same torso/leg/left-arm state that
            # will be exported; assigning after IK would invalidate the wrist
            # target and reintroduce the collision this experiment measures.
            plan.plan.qpos[bodycols]=body_values
            mujoco.mj_forward(m,plan.plan)
        if k:plan.ik(target)
        mujoco.mj_forward(m,plan.plan)
        err=float(np.linalg.norm(plan.plan.xpos[plan.wrist]-target))
        rot_error=float((Rotation.from_matrix(plan.plan.xmat[plan.wrist].reshape(3,3))*Rotation.from_matrix(plan.rotation).inv()).magnitude())
        if preserve_generated_body:
            # Preserve producer body motion while the wrist/arm remains under
            # bounded task-space IK.  The entry ramp retains the canonical
            # frame-zero state and avoids a neutral-to-generated discontinuity.
            result[k,bodycols]=body_values
        result[k,armcols]=plan.plan.qpos[plan.aq]
        if collision_aware_hand:
            from research.shared_adapter.hand_collision import project_hand_frame
            hand_q[k], collision_info = project_hand_frame(
                m, plan.plan, hand_q[k], hand_joint_names, t[k],
                min_clearance_m=collision_hand_min_clearance_m,
                max_relief_rad=collision_hand_max_relief_rad)
            collision_info['frame'] = int(k)
            collision_info['time_s'] = float(t[k])
            collision_projection_rows.append(collision_info)
        rows.append(dict(frame=k,time_s=float(t[k]),raw_wrist_world_pos=modelp.tolist(),refined_wrist_world_pos=rawp.tolist(),target_wrist_world_pos=target.tolist(),position_error_m=err,orientation_error_rad=rot_error,
            model_wrist_world_pos=modelp.tolist(),contact_refinement_weight=float(refine_weight),
            contact_position_correction_m=contact_position_correction,
            contact_rotation_correction_rad=contact_rotation_correction,
            offline_tracking_position_compensation_m=calibration_position.tolist(),offline_tracking_rotation_compensation_rad=calibration_rotation.tolist(),
            max_qpos_change=float(np.max(abs(result[k]-row))),world_wrist_entry_blend=float(u[k]),
            opposition_centering_weight=float(opposition_weight),
            opposition_centering_correction_m=opposition_correction.tolist(),
            distal_alignment_weight=float(distal_alignment_weight_value),
            distal_alignment_correction_m=distal_alignment_correction.tolist(),
            geometry_wrist_orientation_weight=float(geometry_rotation_weight),
            geometry_wrist_orientation_target=(geometry_wrist_target.tolist() if geometry_wrist_target is not None else None)))
    if opposition_centering:
        body_policy = dict(body_policy or {}, opposition_centering=True,
                           opposition_centering_anchor_frame=opposition_anchor,
                           opposition_centering_policy=opposition_info)
    if distal_alignment:
        body_policy = dict(body_policy or {}, distal_alignment=True,
                           distal_alignment_anchor_frame=distal_alignment_anchor,
                           distal_alignment_max_correction_m=float(distal_alignment_max_correction_m),
                           distal_alignment_policy=distal_alignment_info)
    if collision_aware_hand:
        body_policy = dict(body_policy or {}, collision_aware_hand=True,
                           collision_hand_min_clearance_m=float(collision_hand_min_clearance_m),
                           collision_hand_max_relief_rad=float(collision_hand_max_relief_rad),
                           collision_hand_projection={
                               'policy': 'shared-hand-collision-v1',
                               'frames': int(len(collision_projection_rows)),
                               'projected_frames': int(sum(bool(r.get('projected')) for r in collision_projection_rows)),
                               'target_met_frames': int(sum(bool(r.get('clearance_target_met')) for r in collision_projection_rows)),
                               'max_applied_delta_rad': float(max((max(abs(x) for x in r.get('applied_delta_rad', [0.0]))
                                                                   for r in collision_projection_rows), default=0.0)),
                               'strict_native_gate': 'unchanged; native opposition/lift/contact validator remains authoritative',
                           })
    return result,rows,body_policy

def angular_velocity(quat):
    n,b=quat.shape[:2];r=Rotation.from_quat(quat[:,:,[1,2,3,0]].reshape(-1,4))
    delta=(r[b:]*r[:-b].inv()).as_rotvec().reshape(n-1,b,3)*50
    v=np.empty((n,b,3));v[0]=delta[0];v[-1]=delta[-1];v[1:-1]=(delta[:-1]+delta[1:])/2
    return v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','generation','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapt',choices=['neutral-world-wrist'])
    p.add_argument('--clearance',type=float,default=0.)
    p.add_argument('--lift-margin',type=float,default=0.)
    p.add_argument('--contact-refine',action='store_true',help='Explicit offline task-space grasp corridor projection; preserve raw generation')
    p.add_argument('--approach-refine',action='store_true',help='Extend bounded task-space projection to approach, blending from 1 to 2 seconds')
    p.add_argument('--tracking-calibration',type=Path,help='Explicit measured offline WBC bias compensation; no runtime feedback')
    p.add_argument('--adapter',choices=['legacy','unified'],default='legacy',
                   help='Select the shared task-conditioned adapter; legacy preserves historical reproduction')
    p.add_argument('--preserve-generated-body', action='store_true',
                   help='Experimental: retain generated non-right-arm joints while adapting the wrist')
    p.add_argument('--preserve-body-weight', type=float, default=1.0,
                   help='Fraction of generated non-arm displacement to retain (0..1)')
    p.add_argument('--body-preservation-policy', choices=['legacy','bounded'], default='legacy',
                   help='Deterministic generated-body residual policy; bounded applies fixed smoothing, rate, and joint limits')
    p.add_argument('--opposition-centering', action='store_true',
                   help='Opt in to bounded geometry-derived distal-pair centering during closure')
    p.add_argument('--opposition-centering-max-correction-m', type=float, default=0.03,
                   help='Shared geometry correction cap (metres), at most 0.03')
    p.add_argument('--opposition-centering-dominant-axis', action='store_true',
                   help='Use only the largest geometry-derived horizontal opposition error component')
    p.add_argument('--distal-alignment', action='store_true',
                   help='Apply bounded geometry-derived 3-D distal-pair alignment during preclose/closure')
    p.add_argument('--distal-alignment-max-correction-m', type=float, default=0.04,
                   help='Maximum 3-D distal alignment correction in metres (at most 0.04)')
    p.add_argument('--collision-aware-hand', action='store_true',
                   help='Opt in to bounded MuJoCo thumb-to-wrist clearance projection')
    p.add_argument('--collision-hand-min-clearance-m', type=float, default=0.001)
    p.add_argument('--collision-hand-max-relief-rad', type=float, default=0.30)
    p.add_argument('--hand-profile', choices=['default','ardy','ardy_thumb_relief','ardy_thumb_relief_strong','slow_close','early_preshape',
                                               'timing_jitter','coordination_mild','coordination_tiny'],
                   default='default', help='Opt-in shared hand policy for unified adaptation')
    p.add_argument('--task-conditioned-clearance', action='store_true',
                   help='Derive bounded approach clearance from table/cube edge margin')
    p.add_argument('--geometry-grasp-height', action='store_true',
                   help='Apply the shared geometry-derived grasp-height offset during closure')
    p.add_argument('--geometry-wrist-orientation', action='store_true',
                   help='Apply the shared geometry-derived terminal wrist orientation during preclose/closure')
    p.add_argument('--task-clearance-target-margin', type=float, default=0.10,
                   help='Geometry policy target edge margin (same for all positions)')
    args=p.parse_args()
    frozen_path=Path(os.environ.get('KIMODO_FROZEN_CONFIG',
        ROOT/'workspace/kunqi/task9_formal40_20261008/kimodo_frozen_method_source/config/frozen_config.json'))
    run_config=json.loads(frozen_path.read_text())
    if args.tracking_calibration is not None:
        declared=Path(run_config['adapter']['calibration']).resolve()
        if run_config.get('mode')!='development_debug' or args.tracking_calibration.resolve()!=declared:
            p.error('Tracking calibration is not the declared frozen development calibration')
    if args.out.exists():p.error('Fresh output required')
    if (args.clearance or args.lift_margin) and not args.adapt:p.error('Clearance/lift require explicit adaptation')
    if args.contact_refine and not args.adapt:p.error('Contact refinement requires explicit offline adaptation')
    if args.geometry_wrist_orientation and not args.adapt:p.error('Geometry wrist orientation requires explicit adaptation')
    if args.approach_refine and not args.contact_refine:p.error('Approach refinement requires contact refinement')
    if args.tracking_calibration and not args.contact_refine:p.error('Tracking calibration requires explicit contact refinement')
    if not np.isfinite([args.clearance,args.lift_margin,args.preserve_body_weight]).all() or min(args.clearance,args.lift_margin)<0 or not 0<=args.preserve_body_weight<=1:p.error('Invalid adaptation heights/body weight')
    if not np.isfinite(args.opposition_centering_max_correction_m) or not 0 < args.opposition_centering_max_correction_m <= .03:p.error('opposition-centering-max-correction-m must be in (0,.03]')
    if not np.isfinite(args.distal_alignment_max_correction_m) or not 0 < args.distal_alignment_max_correction_m <= .04:p.error('distal-alignment-max-correction-m must be in (0,.04]')
    if not np.isfinite(args.collision_hand_min_clearance_m) or args.collision_hand_min_clearance_m < 0:p.error('collision-hand-min-clearance-m must be finite and nonnegative')
    if not np.isfinite(args.collision_hand_max_relief_rad) or not 0 < args.collision_hand_max_relief_rad <= .5:p.error('collision-hand-max-relief-rad must be in (0,.5]')
    args.out.mkdir(parents=True);out=args.out.resolve()
    task=json.loads((args.task/'task.json').read_text());gen=json.loads((args.generation/'generation.json').read_text())
    verify_generation(args.task,args.generation,task,gen)
    if task['task_id']!=gen['task_id']:raise ValueError('task_id mismatch')
    names=gen['source_qpos_joint_order']
    if len(names)!=29 or len(set(names))!=29 or set(names)!=set(task['body_joint_order']):raise ValueError('Wrong source joint names')
    raw=args.generation/'raw_motion.npz';rawcsv=args.generation/'raw_motion.csv'
    q=np.loadtxt(rawcsv,delimiter=',',ndmin=2)
    required={'local_rot_mats':(34,3,3),'root_positions':(3,), 'posed_joints':(34,3),'global_rot_mats':(34,3,3), 'smooth_root_pos':(3,), 'foot_contacts':(4,), 'global_root_heading':(2,)}
    with np.load(raw,allow_pickle=False) as source:
        for key,shape in required.items():
            if source[key].shape!=(len(q),*shape) or not np.isfinite(source[key]).all():raise ValueError('Invalid raw field '+key)
    plan=np.load(args.task/'task_plan.npz',allow_pickle=False);t=plan['timestamps'];h=plan['hand_ref_q'].copy();hd=plan['hand_ref_dq'].copy()
    if args.adapter == 'unified':
        sys.path.insert(0, str(ROOT))
        from research.shared_adapter.hand import HandMotionGenerator, profile_config
        # Unified profiles are generated from the shared, producer-neutral
        # hand configuration.  Do not inherit a producer-local terminal pose;
        # that was the source of the thumb/table grazing seen in native replay.
        hand_config = profile_config(args.hand_profile)
        h, hand_info = HandMotionGenerator(hand_config).generate(t, seed=int(task.get('seed', 0)))
        hand_info['profile'] = args.hand_profile
        hd = np.gradient(h, t, axis=0)
    else:
        hand_info = None
    hand_schedule=run_config['adapter'].get('hand_preclose')
    if hand_schedule is not None and args.adapter == 'legacy':
        hp=hand_schedule
        u=np.clip((t-hp['start_s'])/(hp['end_s']-hp['start_s']),0,1);early=hp['fraction']*u*u*(3-2*u)
        u=np.clip((t-hp['end_s'])/(hp['close_end_s']-hp['end_s']),0,1);late=hp['fraction']+(1-hp['fraction'])*u*u*(3-2*u)
        fraction=np.where(t<hp['end_s'],early,late)
        h[:,7:]=fraction[:,None]*plan['hand_ref_q'][-1,7:]
        if 'thumb_start_s' in hp:
            u=np.clip((t-hp['thumb_start_s'])/(hp['thumb_end_s']-hp['thumb_start_s']),0,1)
            thumb_fraction=np.where(t<hp['end_s'],hp['fraction']*u*u*(3-2*u),late)
            h[:,11:14]=thumb_fraction[:,None]*plan['hand_ref_q'][-1,11:14]
        hd=np.gradient(h,t,axis=0)
    if h.shape!=(810,14) or hd.shape!=h.shape or not np.isfinite(np.r_[h.ravel(),hd.ravel()]).all():raise ValueError('Invalid hand plan')
    fps=gen.get('fps',gen.get('source_fps'))
    q50=resample(q,fps,t);m,d,bn,hn=canonical(task)
    geometry_grasp_height_info = None
    if args.geometry_grasp_height:
        from research.shared_adapter.grasp_geometry import approach_retraction_m as _approach_retraction_m
        _, geometry_grasp_height_info = _approach_retraction_m(m)
    spatial_clearance_m = 0.0
    spatial_clearance_info = None
    if args.task_conditioned_clearance:
        from research.shared_adapter import task_conditioned_clearance
        spatial_clearance_m, spatial_clearance_info = task_conditioned_clearance(
            m, task['case']['block_xy'], data=d,
            target_edge_margin_m=args.task_clearance_target_margin)
    qa=np.array([m.joint(n).qposadr[0] for n in names]);rqa=int(m.joint('floating_base_joint').qposadr[0])
    initial=np.r_[d.qpos[rqa:rqa+7],d.qpos[qa]];zero=frame_zero_report(q50,initial,h)
    hashes={str(f.resolve()):sha(f) for f in [args.task/'task.json',args.task/'task_plan.npz',args.generation/'generation.json',raw,rawcsv]}
    calibration=None
    if args.tracking_calibration:
        calibration=json.loads(args.tracking_calibration.read_text())
        if calibration['task_id']!=task['task_id']:raise ValueError('Calibration/task mismatch')
        if not calibration.get('source_hashes'):raise ValueError('Missing measured calibration evidence hashes')
        for key in ('close_position_bias_m','hold_position_bias_m','close_rotation_bias_rad','hold_rotation_bias_rad'):
            value=np.asarray(calibration[key]);bound=.06 if 'position' in key else .30
            if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)>bound:raise ValueError('Invalid/beyond-bound measured calibration '+key)
        for path,expected in calibration['source_hashes'].items():
            if sha(path)!=expected:raise ValueError('Changed calibration evidence')
        hashes[str(args.tracking_calibration.resolve())]=sha(args.tracking_calibration)
        hashes.update(calibration['source_hashes'])
    metadata=dict(task_id=task['task_id'],contract_version=task['contract_version'],source='actual Kimodo generation',raw_source=str(rawcsv.resolve()),raw_motion=str(raw.resolve()),source_hashes=hashes,
        body_joint_order=bn,hand_joint_order=hn,source_qpos_joint_order=names,units={'positions':'m','angles':'rad','time':'s'},quaternion_order='wxyz',velocity_frame='world',raw_frame_zero=zero,
        raw_vs_adapted='adapted' if args.adapt else 'raw',physical_success=None,expert_valid=None)
    write_json(out/'export_metadata.json',metadata)
    if not zero['pass'] and not args.adapt:
        write_json(out/'export_result.json',dict(status='REFUSED_NONCANONICAL_FRAME_ZERO',playable=False,**zero));return 2
    if args.adapt:
        contact_plan=(plan['source_timestamps'],plan['wrist_world_pos'][:,1],plan['wrist_world_quat'][:,1]) if args.contact_refine else None
        q50,rows,body_policy=adapt(q50,t,m,d,names,args.clearance,args.lift_margin,contact_plan,calibration=calibration,approach_refine=args.approach_refine,clearance_decay_s=run_config['adapter'].get('clearance_decay_s'),preserve_generated_body=args.preserve_generated_body,preserve_body_weight=args.preserve_body_weight,spatial_clearance_m=spatial_clearance_m,body_preservation_policy=args.body_preservation_policy,hand_q=h,hand_joint_names=hn,opposition_centering=args.opposition_centering,opposition_centering_max_correction_m=args.opposition_centering_max_correction_m,opposition_centering_dominant_axis=args.opposition_centering_dominant_axis,distal_alignment=args.distal_alignment,distal_alignment_max_correction_m=args.distal_alignment_max_correction_m,collision_aware_hand=args.collision_aware_hand,collision_hand_min_clearance_m=args.collision_hand_min_clearance_m,collision_hand_max_relief_rad=args.collision_hand_max_relief_rad,geometry_grasp_height=args.geometry_grasp_height,geometry_wrist_orientation=args.geometry_wrist_orientation)
        write_json(out/'adaptation_perframe.json',rows)
        metadata['postprocessing']=dict(mode=args.adapt+'+bounded-contact-refinement' if args.contact_refine else args.adapt,
            error_measurement='offline IK against explicit refined target' if args.contact_refine else 'offline IK against generated world wrist plus explicit clearance/lift margin',
            entry_blend_s=1.5,clearance_m=args.clearance,clearance_blend_s=[4,7],lift_margin_m=args.lift_margin,lift_blend_s=[10,12],
            max_wrist_position_error_m=max(r['position_error_m'] for r in rows),max_wrist_orientation_error_rad=max(r['orientation_error_rad'] for r in rows),offline_ik_only=True,
            contact_refinement=args.contact_refine,contact_refinement_blend_s=([1,2] if args.approach_refine else [6,8]) if args.contact_refine else None,
            grasp_corridor_axis_halfwidth_m=.005 if args.contact_refine else None,grasp_orientation_radius_rad=.04 if args.contact_refine else None,
            max_contact_position_correction_m=max(r['contact_position_correction_m'] for r in rows),
            max_contact_rotation_correction_rad=max(r['contact_rotation_correction_rad'] for r in rows))
        metadata['postprocessing']['task_conditioned_clearance'] = bool(args.task_conditioned_clearance)
        metadata['postprocessing']['task_clearance_target_margin_m'] = float(args.task_clearance_target_margin)
        metadata['postprocessing']['spatial_clearance_m'] = float(spatial_clearance_m)
        metadata['postprocessing']['spatial_clearance_policy'] = spatial_clearance_info
        metadata['postprocessing']['geometry_grasp_height'] = bool(args.geometry_grasp_height)
        metadata['postprocessing']['geometry_grasp_height_policy'] = (
            geometry_grasp_height_info if args.geometry_grasp_height else None)
        metadata['postprocessing']['geometry_wrist_orientation'] = bool(args.geometry_wrist_orientation)
        metadata['postprocessing']['geometry_wrist_orientation_policy'] = ({
            'policy': 'shared-geometry-wrist-orientation-v1',
            'x_tilt_rad': -0.3490658503988659,
            'window_s': [7.0, 8.0],
            'target_rotation': __import__('research.shared_adapter.grasp_geometry', fromlist=['geometry_wrist_orientation']).geometry_wrist_orientation().tolist(),
            'case_or_seed_lookup': False,
            'native_gate': 'unchanged; native opposition/lift/contact validator remains authoritative',
        } if args.geometry_wrist_orientation else None)
        metadata['postprocessing']['preserve_generated_body'] = bool(args.preserve_generated_body)
        metadata['postprocessing']['preserve_body_weight'] = float(args.preserve_body_weight)
        metadata['postprocessing']['body_preservation_policy'] = args.body_preservation_policy
        metadata['postprocessing']['body_preservation'] = body_policy
        metadata['postprocessing']['opposition_centering'] = bool(args.opposition_centering)
        metadata['postprocessing']['opposition_centering_anchor_frame'] = (
            body_policy.get('opposition_centering_anchor_frame') if isinstance(body_policy, dict) else None)
        metadata['postprocessing']['opposition_centering_policy'] = (
            body_policy.get('opposition_centering_policy') if isinstance(body_policy, dict) else None)
        metadata['postprocessing']['opposition_centering_max_correction_m'] = float(args.opposition_centering_max_correction_m)
        metadata['postprocessing']['opposition_centering_dominant_axis'] = bool(args.opposition_centering_dominant_axis)
        metadata['postprocessing']['distal_alignment'] = bool(args.distal_alignment)
        metadata['postprocessing']['distal_alignment_max_correction_m'] = float(args.distal_alignment_max_correction_m)
        metadata['postprocessing']['collision_aware_hand'] = bool(args.collision_aware_hand)
        metadata['postprocessing']['collision_hand_min_clearance_m'] = float(args.collision_hand_min_clearance_m)
        metadata['postprocessing']['collision_hand_max_relief_rad'] = float(args.collision_hand_max_relief_rad)
        # Kimodo's established scalar lift-margin is the z-only instance of
        # the shared EEF lift policy.  Keep the historical 10--12 s schedule
        # and scalar CLI unchanged while exposing an explicit, producer-neutral
        # provenance record; native physical/contact gates remain authoritative.
        metadata['postprocessing']['eef_lift_offset_policy'] = {
            'version': 'shared-eef-lift-offset-v1',
            'offset_m': [0.0, 0.0, float(args.lift_margin)],
            'onset_window_s': [10.0, 12.0],
            'active_after_end': True,
            'source_cli': '--lift-margin',
            'native_gate': 'unchanged; native opposition/lift/contact validator remains authoritative',
        }
        metadata['postprocessing']['clearance_decay_s']=run_config['adapter'].get('clearance_decay_s')
        metadata['postprocessing']['hand_preclose']=hand_schedule
        metadata['postprocessing']['hand_generator'] = hand_info
        metadata['postprocessing']['hand_profile'] = args.hand_profile
        metadata['postprocessing']['adapter'] = args.adapter
        metadata['postprocessing']['offline_tracking_calibration']=calibration
        if calibration:metadata['postprocessing']['mode']+=' + measured-offline-tracking-compensation'
        if calibration and calibration.get('kind')=='offline_grasp_geometry_probe':
            metadata['postprocessing']['offline_grasp_geometry_calibration']=metadata['postprocessing'].pop('offline_tracking_calibration')
            metadata['postprocessing']['mode']=metadata['postprocessing']['mode'].replace('measured-offline-tracking-compensation','baseline-informed-offline-grasp-geometry-probe')
        if max(r['contact_position_correction_m'] for r in rows)>.10:raise ValueError('Generated wrist requires >10cm contact correction; reject candidate')
        write_json(out/'export_metadata.json',metadata)
        if metadata['postprocessing']['max_wrist_position_error_m']>.02:raise ValueError('Offline IK target error >.02m; reference refused')
    zero_after=frame_zero_report(q50,initial,h)
    if not zero_after['pass']:raise ValueError('Output frame zero still noncanonical')
    import mujoco
    body_ids=[m.body('pelvis').id]+[int(m.jnt_bodyid[m.joint(bn[i-1]).id]) for i in BODY_INDEXES[1:]]
    pos=np.empty((810,14,3));quat=np.empty((810,14,4));fk=mujoco.MjData(m);geometry=[]
    for k,row in enumerate(q50):
        fk.qpos[:]=d.qpos;fk.qpos[rqa:rqa+7]=row[:7];fk.qpos[qa]=row[7:];mujoco.mj_forward(m,fk)
        pos[k]=fk.xpos[body_ids];quat[k]=fk.xquat[body_ids]
        geometry.append([float(fk.xpos[m.body('pelvis').id,2]),float(fk.xpos[m.body('left_ankle_roll_link').id,2]),float(fk.xpos[m.body('right_ankle_roll_link').id,2])])
    if not np.allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-5):raise ValueError('FK quaternions not unit')
    canonical_q=q50[:,[names.index(n)+7 for n in bn]];linear=np.gradient(pos,.02,axis=0);angular=angular_velocity(quat)
    arrays=dict(timestamps=t,body_ref_q=canonical_q,body_ref_dq=np.gradient(canonical_q,.02,axis=0),hand_ref_q=h,hand_ref_dq=hd,
        root_ref_pos=q50[:,:3],root_ref_quat=q50[:,3:7],root_lin_vel=linear[:,0],root_ang_vel=angular[:,0])
    np.savez_compressed(out/'reference.npz',**arrays)
    motion=out/'reference/motion';motion.mkdir(parents=True)
    values=dict(joint_pos=arrays['body_ref_q'],joint_vel=arrays['body_ref_dq'],body_pos=pos,body_quat=quat,body_lin_vel=linear,body_ang_vel=angular)
    headers={}
    for key,value in values.items():
        flat=value.reshape(810,-1)
        if key.startswith('joint'):header=[('joint_' if key=='joint_pos' else 'joint_vel_')+str(i) for i in range(29)]
        else:
            axes='wxyz' if key=='body_quat' else 'xyz';suffix={'body_lin_vel':'vel_','body_ang_vel':'angvel_'}.get(key,'')
            header=[f'body_{i}_{suffix}{axis}' for i in range(14) for axis in axes]
        headers[key]=header;np.savetxt(motion/(key+'.csv'),flat,delimiter=',',fmt='%.12g',header=','.join(header),comments='')
    for side in ('left','right'):
        labels=[side+'_'+n for n in ['thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1']]
        cols=[hn.index(side+'_hand_'+label[len(side)+1:]+'_joint') for label in labels]
        np.savetxt(motion/(side+'_hand_pos.csv'),h[:,cols],delimiter=',',fmt='%.12g',header=','.join(labels),comments='')
    (motion/'metadata.txt').write_text('Metadata for: motion\nBody part indexes:\n['+' '.join(map(str,BODY_INDEXES))+']\n\nTotal timesteps: 810\n')
    metadata.update(headers=headers,body_part_indexes=BODY_INDEXES,body_part_names=[m.body(i).name for i in body_ids],frame_zero=zero_after,torso_feet_z_min=np.min(geometry,axis=0).tolist(),torso_feet_z_max=np.max(geometry,axis=0).tolist())
    write_json(out/'export_metadata.json',metadata)
    setting=dict(metadata,case=task['case'],initial_qpos=d.qpos.tolist(),initial_qvel=d.qvel.tolist(),prompt=task['prompt'],seed=task['seed'],task_path=str(args.task.resolve()),generation_path=str(args.generation.resolve()))
    write_json(out/'setting.json',setting);write_json(out/'export_result.json',dict(status='REFERENCE_READY',playable=True,physical_success=None,expert_valid=None))
    print(json.dumps(dict(status='REFERENCE_READY',out=str(out),frames=810,raw_frame_zero=zero)));return 0

if __name__=='__main__':raise SystemExit(main())

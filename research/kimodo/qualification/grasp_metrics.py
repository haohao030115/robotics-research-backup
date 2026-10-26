"""Per-physics-step evidence for the user's grasp criterion (version 2026-10-07).

Uses canonical thumb/index-or-middle contact grouping. Contact positions and
normals are retained for inspection; no new angular threshold is invented.
"""
import numpy as np


class HoldTimer:
    def __init__(self):
        self.current = self.maximum = 0.

    def update(self, valid, dt):
        self.current = self.current + dt if valid else 0.
        self.maximum = max(self.maximum, self.current)
        return self.current


class GraspEvidence:
    def __init__(self, model, initial_z):
        self.model = model
        self.initial_z = float(initial_z)
        self.cube = model.body('task_red_cube').id
        self.cube_geom = model.geom('task_cube_geom').id
        self.timer = HoldTimer()
        self.rows = []
        self.support_steps = self.fall_steps = 0

    def step(self, data, frame, playing):
        import mujoco
        contacts = []
        support = []
        for i, c in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            normal_force = float(abs(force[0]))
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = self.model.geom_bodyid[[g1,g2]]
            n1, n2 = self.model.body(b1).name, self.model.body(b2).name
            other = g2 if g1 == self.cube_geom else g1 if g2 == self.cube_geom else None
            if other is not None:
                contacts.append(dict(body=self.model.body(self.model.geom_bodyid[other]).name,
                    normal_force_n=normal_force,position=c.pos.tolist(),
                    cube_outward_normal=(c.frame[:3]*(1 if g1==self.cube_geom else -1)).tolist(),
                    distance_m=float(c.dist)))
            if normal_force > 1e-4:
                for rb, env, rb_id, env_id in [(n1,n2,b1,b2),(n2,n1,b2,b1)]:
                    if rb_id <= 0 or rb.startswith('task_'):
                        continue
                    if env == 'task_table' or (env_id == 0 and rb not in
                            ('left_ankle_roll_link','right_ankle_roll_link')):
                        support.append([rb,env or 'world',normal_force])
        thumb = max((c['normal_force_n'] for c in contacts if 'right_hand_thumb' in c['body']),default=0.)
        finger = max((c['normal_force_n'] for c in contacts if
                      'right_hand_index' in c['body'] or 'right_hand_middle' in c['body']),default=0.)
        cube_table = any(c['body']=='task_table' for c in contacts)
        lift = float(data.xpos[self.cube,2]-self.initial_z)
        qualified = bool(playing and lift>=.10 and thumb>1e-4 and finger>1e-4 and not cube_table)
        hold = self.timer.update(qualified,self.model.opt.timestep)
        fall = bool(data.xpos[self.model.body('pelvis').id,2]<.2)
        self.support_steps += bool(support)
        self.fall_steps += fall
        row = dict(sim_time=float(data.time),frame=int(frame),playing=bool(playing),lift_m=lift,
            thumb_force_n=thumb,opposing_finger_force_n=finger,cube_table_contact=cube_table,
            qualified=qualified,hold_seconds=hold,max_hold_seconds=self.timer.maximum,
            fall=fall,support_contacts=support,contacts=contacts,
            cube_world_pos=data.xpos[self.cube].tolist(),cube_world_quat=data.xquat[self.cube].tolist(),
            qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),actuator_control=data.ctrl.tolist(),
            external_force_max=float(max(np.max(abs(data.xfrc_applied)),np.max(abs(data.qfrc_applied)))),equality_count=int(self.model.neq))
        self.rows.append(row)
        return row

    def report(self):
        active = [r for r in self.rows if r['playing']]
        return dict(criterion_version='user_physical_success_20261007_v1',
            contact_grouping='canonical right thumb AND right index/middle with positive normal force >1e-4N',
            physical_success=bool(self.timer.maximum>=2.-1e-9),
            max_continuous_hold_seconds=self.timer.maximum,
            final_continuous_hold_seconds=self.timer.current,
            max_lift_m=max((r['lift_m'] for r in active),default=0.),
            final_lift_m=active[-1]['lift_m'] if active else None,
            support_contact_steps=self.support_steps,fall_steps=self.fall_steps,
            recorded_physics_steps=len(self.rows),recorded_playback_physics_steps=len(active),
            initial_cube_z_m=self.initial_z)

#!/usr/bin/env python3
"""SONIC DDS body + independent hands in a real tabletop MuJoCo scene.

The logged SONIC playback ticks are the shared trajectory clock. Physics runs
in real time; wall time is used only for pacing/timeouts, never for hand playback.
No body reference is written directly to qpos or to body actuator targets.
"""
import argparse
import json
import os
from pathlib import Path
import pty
import select
import shlex
import signal
import shutil
import subprocess
import tempfile
import time

import mujoco
import numpy as np

from cases import reset_case
from expert_trajectory import (JointMap, PhysicalGraspEvaluator, load_reference, orders,
                               robot_state, save_arrays, sha256, write_json)
from hand_trajectory_controller import HandTrajectoryController
from project_paths import ROOT, SONIC_REPO, DEFAULT_EPISODE
from scene import load_scene


class CSVFollower:
    """Consume complete flushed rows, including every row in a delayed batch."""
    def __init__(self, path, header=True):
        self.path, self.header = Path(path), header
        self.file = None
        self.partial = ""

    def read(self):
        if self.file is None:
            if not self.path.exists(): return []
            self.file = self.path.open()
        text = self.partial + self.file.read()
        parts = text.split("\n")
        self.partial = parts.pop()
        rows = []
        for line in parts:
            if not line.strip(): continue
            if self.header:
                self.header = False
                continue
            rows.append([float(v) for v in line.strip().split(",") if v])
        return rows

    def close(self):
        if self.file: self.file.close()


class PlaybackCursor:
    """One no-planner motion: frame advances once per active 50 Hz control tick.

    motion_playing.csv is flushed by LogPostState at the same control tick whose
    target is used by SONIC. record-input-file provides the explicit 100 Hz
    current_frame/play cursor for an independent index agreement check. No
    keyboard timestamp or wall-clock estimate drives either reference array.
    """
    def __init__(self, out, total):
        self.play = CSVFollower(out / "policy_logs/motion_playing.csv")
        self.explicit = CSVFollower(out / "playback_cursor.csv", header=False)
        self.first_index = None
        self.last_tick = -1
        self.frame = 0
        self.started = self.completed = False
        self.total = total
        self.explicit_frame = 0
        self.tick_time = 0.
        self.missing_ticks = 0
        self.standby_ticks = 0

    def update(self):
        for row in self.explicit.read():
            if len(row) >= 3 and row[2] > .5:
                self.explicit_frame = int(row[1])
        new = []
        for row in self.play.read():
            index, playing = int(row[0]), row[-1] > .5
            if playing:
                if self.first_index is None: self.first_index = index
                frame = index - self.first_index
                if frame < 0 or frame >= self.total:
                    raise RuntimeError("Unexpected SONIC restart/motion switch")
                if self.last_tick >= 0 and index != self.last_tick + 1:
                    self.missing_ticks += index - self.last_tick - 1
                self.last_tick = index
                self.frame = frame
                self.started = True
                self.tick_time = row[2] / 1000
                new.append((frame, self.tick_time))
            elif self.started:
                self.completed = True
            else:
                self.standby_ticks += 1
        return new

    def close(self):
        self.play.close(); self.explicit.close()


class SonicProcess:
    def __init__(self, motion_parent, out, kp_scale, kd_scale=1.):
        deploy = SONIC_REPO / "gear_sonic_deploy"
        self.master, slave = pty.openpty()
        self.log = (out / "sonic.log").open("wb")
        self.captured = bytearray()
        command = [str(deploy / "target/release/g1_deploy_onnx_ref"), "lo", "policy/release/model_decoder.onnx",
                   str(motion_parent), "--obs-config", "policy/release/observation_config.yaml",
                   "--encoder-file", "policy/release/model_encoder.onnx", "--input-type", "keyboard",
                   "--output-type", "zmq", "--zmq-out-port", "5567", "--disable-crc-check",
                   "--logs-dir", str(out / "policy_logs"), "--enable-csv-logs",
                   "--target-motion-logfile", str(out / "target_motion.csv"),
                   "--record-input-file", str(out / "playback_cursor.csv")]
        if kp_scale != 1:
            command += ["--motor-kp-scale", f"0-28={kp_scale}"]
        if kd_scale != 1:
            command += ["--motor-kd-scale", f"0-28={kd_scale}"]
        self.command = command
        self.process = subprocess.Popen(["bash", "--noprofile", "--norc", "-c",
                                         "source scripts/setup_env.sh; exec " + shlex.join(command)],
                                        cwd=deploy, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)

    def pump(self):
        while select.select([self.master], [], [], 0)[0]:
            try: chunk = os.read(self.master, 65536)
            except OSError: break
            if not chunk: break
            self.captured.extend(chunk)
            self.log.write(chunk); self.log.flush()

    def key(self, value):
        os.write(self.master, value.encode())

    def close(self):
        self.pump()
        if self.process.poll() is None:
            self.key("O")
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
        self.pump(); self.log.close(); os.close(self.master)


class TabletopDDS:
    def __init__(self, arrays, metadata, args):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
        config = SimLoopConfig(enable_onscreen=False).load_wbc_yaml()
        ChannelFactoryInitialize(0, "lo")
        self.config = config
        robot_xml = None if args.robot == "baseline" else SONIC_REPO / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
        self.model, self.data = load_scene(supported=args.pelvis_support == "weld", robot_path=robot_xml,
                                          root_height=metadata.get("source_root_height_m"))
        m, d = self.model, self.data
        m.opt.timestep = 1. / args.physics_hz
        reset_case(m, d, metadata["case"])
        names, hand_names, motors = orders()
        self.body = JointMap(m, names, 29)
        self.motor = JointMap(m, motors, 29)
        self.hand = JointMap(m, hand_names, 14)
        # Initialization only: source robot state, cube remains at reset_case's pose.
        d.qpos[self.body.qa] = arrays["body_ref_q"][0]
        d.qpos[self.hand.qa] = arrays["hand_ref_q"][0]
        root = m.joint("floating_base_joint").id
        self.root_qa, self.root_va = m.jnt_qposadr[root], m.jnt_dofadr[root]
        d.qpos[self.root_qa:self.root_qa+3] = arrays["root_ref_pos"][0]
        d.qpos[self.root_qa+3:self.root_qa+7] = arrays["root_ref_quat"][0]
        mujoco.mj_forward(m, d)
        self.home = d.qpos[self.motor.qa].copy()
        self.startup_wbc = None
        if args.startup_body == "freebase-wbc":
            from generate_freebase_expert import WholeBody
            self.startup_wbc = WholeBody(m, d, self.body, self.hand, neutral=metadata.get("initial_pose", {}).get("name") == "neutral_standing_v1")
        self.bridge = UnitreeSdk2Bridge(config)
        self.band = ElasticBand()
        self.band.kd_ang = args.band_angular_damping
        robot_mass = sum(m.body_mass[i] for i in range(m.nbody)
                         if not (m.body(i).name or "").startswith("task_"))
        if args.band_anchor == "source-equilibrium":
            self.band.point = arrays["root_ref_pos"][0].copy()
            self.band.point[2] += robot_mass * abs(m.opt.gravity[2]) / self.band.kp_pos
        self.final_band_enabled = args.band == "on"
        self.startup_support = args.startup_support == "elastic-band"
        self.hand_controller = HandTrajectoryController(m, arrays["timestamps"], arrays["hand_ref_q"], arrays["hand_ref_dq"])
        self.evaluator = PhysicalGraspEvaluator(m,d,task_arm=metadata.get("task_arm","right"))
        self.support_mode = args.pelvis_support
        self.jp, self.jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        self.initial_cube_qpos = d.qpos[m.joint("task_cube_free").qposadr[0]:m.joint("task_cube_free").qposadr[0]+7].copy()
        self.command_received_steps = 0

    def publish(self):
        m,d = self.model,self.data
        torso = m.body("torso_link").id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,torso,velocity,1)
        rqa,rva = self.root_qa,self.root_va
        obs = {"floating_base_pose": d.qpos[rqa:rqa+7], "floating_base_vel": d.qvel[rva:rva+6],
               "floating_base_acc": d.qacc[rva:rva+6], "secondary_imu_quat": d.xquat[torso],
               "secondary_imu_vel": np.r_[velocity[3:], velocity[:3]],
               "body_q": d.qpos[self.motor.qa], "body_dq": d.qvel[self.motor.va],
               "body_ddq": d.qacc[self.motor.va], "body_tau_est": d.actuator_force[self.motor.aids],
               "left_hand_q": d.qpos[self.hand.qa[:7]], "left_hand_dq": d.qvel[self.hand.va[:7]],
               "right_hand_q": d.qpos[self.hand.qa[7:]], "right_hand_dq": d.qvel[self.hand.va[7:]], "time": d.time}
        self.bridge.PublishLowState(obs)

    def step(self, reference_time, sonic_control, playing):
        m,d = self.model,self.data
        self.publish()
        if sonic_control:
            with self.bridge.low_cmd_lock:
                received = self.bridge.low_cmd_received
                cmd = self.bridge.low_cmd
                values = np.array([[motor.q,motor.dq,motor.kp,motor.kd,motor.tau]
                                   for motor in cmd.motor_cmd[:29]],float)
            if not received or not np.isfinite(values).all():
                raise RuntimeError("SONIC motor command missing/non-finite")
            q,dq,kp,kd,ff = values.T
            torque = ff + kp*(q-d.qpos[self.motor.qa]) + kd*(dq-d.qvel[self.motor.va])
            self.command_received_steps += int(playing)
        else:
            # Startup only; no scripted/IK body command during SONIC playback.
            torque = 180*(self.home-d.qpos[self.motor.qa])-5*d.qvel[self.motor.va]+d.qfrc_bias[self.motor.va]
        for i,jid in enumerate(self.motor.jids):
            if m.jnt_actfrclimited[jid]: torque[i] = np.clip(torque[i], *m.jnt_actfrcrange[jid])
        d.ctrl[self.motor.aids] = torque
        if self.startup_wbc is not None and not playing:
            self.startup_wbc.apply()
        hand_target, hand_command = self.hand_controller.apply(d,reference_time)
        # Support is opt-in for diagnostics and never eligible for certification.
        enabled = self.final_band_enabled if playing else self.startup_support
        pelvis = m.body("pelvis").id
        if self.support_mode == "weld":
            enabled = False
        if enabled:
            mujoco.mj_jacBody(m,d,self.jp,self.jr,pelvis)
            pose = np.r_[d.xpos[pelvis],d.xquat[pelvis],self.jp@d.qvel,self.jr@d.qvel]
            d.xfrc_applied[pelvis] = self.band.Advance(pose)
        else:
            d.xfrc_applied[pelvis] = 0
        mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("NaN/Inf physics state")
        if any(w.number for w in d.warning):
            raise RuntimeError("MuJoCo physics warning")
        physical = self.evaluator.step(d) if playing else {}
        return hand_target, hand_command, enabled, physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--candidate", help="Candidate folder; defaults to the episode selected_candidate, or actual for a new source")
    parser.add_argument("--band", choices=["on","off"], default="off")
    parser.add_argument("--startup-support", choices=["none", "elastic-band"], default="none",
                        help="Debug band support during startup; prevents expert certification")
    parser.add_argument("--band-anchor", choices=["official","source-equilibrium"], default="source-equilibrium")
    parser.add_argument("--band-angular-damping", type=float, default=10., help="Explicit runtime band rotational damping; official default is 10")
    parser.add_argument("--robot", choices=["baseline","sonic"], default="baseline")
    parser.add_argument("--pelvis-support", choices=["none","weld"], default="none")
    parser.add_argument("--kp-scale", type=float, default=1.)
    parser.add_argument("--kd-scale", type=float, default=1.)
    parser.add_argument("--physics-hz", type=int, default=200,
                        help="Real-time physics rate; official sim uses 200 Hz, source baseline uses 500 Hz")
    parser.add_argument("--warmup-frames", type=int, default=0,
                        help="Actual SONIC control ticks holding reference frame 0 before playback")
    parser.add_argument("--startup-body", choices=["sonic-init","source-pd","freebase-wbc"], default="freebase-wbc",
                        help="Unassisted whole-body standing before playback; other modes are diagnostic")
    parser.add_argument("--encoder-mode",type=int,choices=[0,1],help="Defaults to candidate metadata; mode 1 requires three-point body poses")
    parser.add_argument("--run-name")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    if args.candidate is None:
        episode_metadata = args.episode / "metadata.json"
        if not episode_metadata.is_file(): parser.error(f"Missing episode metadata: {episode_metadata}")
        args.candidate = json.loads(episode_metadata.read_text()).get("selected_candidate", "actual")
    if Path(args.candidate).name != args.candidate: parser.error("Candidate must be one directory name")
    if args.kp_scale <= 0 or not np.isfinite(args.kp_scale): parser.error("kp-scale must be finite/positive")
    if args.kd_scale <= 0 or not np.isfinite(args.kd_scale): parser.error("kd-scale must be finite/positive")
    if not np.isfinite(args.band_angular_damping) or args.band_angular_damping < 0: parser.error("Invalid band damping")
    if args.physics_hz < 50 or args.physics_hz % 50: parser.error("physics-hz must be a multiple of 50")
    if args.warmup_frames < 0: parser.error("warmup-frames must be nonnegative")
    if args.pelvis_support == "weld" and args.band == "on":
        parser.error("Weld diagnostic must use --band off; support is labeled separately")
    folder = args.episode.resolve() / "candidates" / args.candidate
    arrays, metadata = load_reference(folder)
    encoder_mode = args.encoder_mode if args.encoder_mode is not None else metadata["sonic_config"].get("encoder_mode",0)
    if encoder_mode == 1 and not metadata.get("cartesian_body_part_indexes"):
        parser.error("Mode 1 requires an augmented Cartesian candidate, not pelvis-only body CSV")
    name = args.run_name or f"{args.candidate}_band_{args.band}"
    if Path(name).name != name: parser.error("Run name must be one directory name")
    out = args.episode.resolve() / "validation" / name
    if out.exists(): parser.error(f"Run already exists: {out}; choose --run-name")
    out.mkdir(parents=True)
    staging = out / "reference"
    staging.mkdir()
    (staging / "grasp_episode").symlink_to(folder, target_is_directory=True)
    runtime = TabletopDDS(arrays,metadata,args)
    assert runtime.model.neq == (1 if args.pelvis_support == "weld" else 0)
    # Per-tick std::endl/CSV reads and video writes must not stall on shared NFS.
    # Preserve every artifact by copying the node-local live directory after stop.
    live_temp = tempfile.TemporaryDirectory(prefix="sonic_grasp_live_")
    live = Path(live_temp.name)
    policy = SonicProcess(staging,live,args.kp_scale,args.kd_scale)
    cursor = PlaybackCursor(live,len(arrays["timestamps"]))
    records = {}; frame_ids = []; wall_times = []; sim_times = []; contact_rows = []
    hand_commands = []; applied_targets = []; band_values = []; sync_ages = []; direct_offsets = []
    skipped_frames = []; last_recorded = -1
    initialized = control_sent = play_sent = False
    failure = None; completed = False
    images = {}; image_frames = []; image_sim_times = []
    renderers = {}; writers = {}
    if not args.no_images:
        import cv2
        for camera in ("head_rgb","wrist_rgb","task_overview"):
            renderers[camera] = mujoco.Renderer(runtime.model,height=240,width=320)
            writers[camera] = cv2.VideoWriter(str(live / f"{camera}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),10,(320,240))
            if not writers[camera].isOpened(): raise RuntimeError("Video writer failed")
            # Compile EGL rendering before playback, so shader warmup cannot skip frames.
            renderers[camera].update_scene(runtime.data,camera=camera)
            renderers[camera].render()
    start = time.monotonic(); next_step = start
    try:
        while time.monotonic()-start < 200:
            policy.pump()
            if policy.process.poll() is not None: raise RuntimeError("SONIC process exited unexpectedly")
            if not initialized and b"Init Done" in policy.captured:
                initialized = True
                if encoder_mode == 1: policy.key("Z")
                policy.key("]")
                control_sent = True
                print(f"{name}: SONIC initialized; starting body control",flush=True)
            # Initialization/start signal only; source frame state remains the clock.
            if initialized and not play_sent and runtime.bridge.low_cmd_received and cursor.play.path.exists() and cursor.standby_ticks >= args.warmup_frames:
                policy.key("T"); play_sent = True
                print(f"{name}: requested playback; hands wait for SONIC playing tick",flush=True)
            was_playing = cursor.started
            new = cursor.update()
            if cursor.started and not was_playing:
                next_step = time.monotonic()
            if cursor.completed:
                completed = True
                break  # Freeze physics immediately, before SONIC's frame-0 reset acts.
            frame = cursor.frame if cursor.started else 0
            ref_time = arrays["timestamps"][frame]
            body_ready = runtime.bridge.low_cmd_received and (args.startup_body == "sonic-init" or control_sent)
            hq,hcmd,band_enabled,physical = runtime.step(ref_time, body_ready, cursor.started)
            if new:
                # Never invent intermediate states if IO stalls: retain missed frames and fail expert gating.
                for missed,_ in new[:-1]: skipped_frames.append(missed)
                if frame != last_recorded + 1: skipped_frames.extend(range(last_recorded+1,frame))
                state = robot_state(runtime.model,runtime.data,runtime.body,runtime.hand)
                for key,value in state.items(): records.setdefault(key,[]).append(value)
                frame_ids.append(frame); wall_times.append(time.time()); sim_times.append(runtime.data.time)
                applied_targets.append(hq); hand_commands.append(hcmd); band_values.append(int(band_enabled))
                sync_ages.append(max(0,time.time()-cursor.tick_time))
                direct_offsets.append(cursor.explicit_frame-frame)
                contact_rows.append({"frame":frame,**physical,"contacts":runtime.evaluator.last_contacts})
                last_recorded = frame
                if frame % 50 == 0: print(f"{name}: frame {frame}, lift={physical.get('lift_m',0):.3f}m",flush=True)
                if renderers and frame % 5 == 0:
                    for camera,renderer in renderers.items():
                        renderer.update_scene(runtime.data,camera=camera)
                        image = renderer.render().copy()
                        images.setdefault(camera,[]).append(image)
                        writers[camera].write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    image_frames.append(frame); image_sim_times.append(runtime.data.time)
            next_step += runtime.model.opt.timestep
            delay = next_step-time.monotonic()
            if delay > 0: time.sleep(delay)
            # Catch up after a short delay rather than discarding physical time.
        if not completed: failure = "timeout_before_episode_completed"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        print(f"{name}: {failure}",flush=True)
    finally:
        policy.close(); cursor.close()
        for writer in writers.values(): writer.release()
        for renderer in renderers.values(): renderer.close()
        for path in live.iterdir():
            if path.is_dir(): shutil.copytree(path,out/path.name)
            else: shutil.copy2(path,out/path.name)
        live_temp.cleanup()
    actual = {key:np.asarray(value) for key,value in records.items()}
    widths = {"body_q":29,"body_dq":29,"hand_q":14,"hand_dq":14,
              "root_pos":3,"root_quat":4,"cube_pos":3,"cube_quat":4,
              "eef_pos":3,"eef_quat":4,"qpos":runtime.model.nq,"qvel":runtime.model.nv,"ctrl":runtime.model.nu}
    for key,width in widths.items(): actual.setdefault(key,np.empty((0,width)))
    actual.update(timestamps=arrays["timestamps"][np.asarray(frame_ids,dtype=int)],
                  reference_frame=np.asarray(frame_ids,dtype=np.int64),wall_timestamps=np.asarray(wall_times),sim_timestamps=np.asarray(sim_times),
                  hand_applied_ref_q=np.asarray(applied_targets).reshape(-1,14),hand_actuator_command=np.asarray(hand_commands).reshape(-1,14),
                  elastic_band_enabled=np.asarray(band_values),sync_age_seconds=np.asarray(sync_ages),
                  explicit_cursor_offset_frames=np.asarray(direct_offsets))
    save_arrays(out,actual)
    if images:
        np.savez_compressed(out / "images.npz",**{k:np.asarray(v) for k,v in images.items()},
                            reference_frame=np.asarray(image_frames),timestamps=arrays["timestamps"][np.asarray(image_frames,dtype=int)],sim_timestamps=np.asarray(image_sim_times))
    write_json(out / "contacts.json",contact_rows)
    report = runtime.evaluator.report(runtime.data)
    coverage = frame_ids == list(range(len(arrays["timestamps"])))
    synced = coverage and not skipped_frames and cursor.missing_ticks == 0 and bool(sync_ages) and max(sync_ages) <= .04
    hand_error = float(np.max(np.abs(np.asarray(applied_targets)-arrays["hand_ref_q"][frame_ids]))) if frame_ids else None
    synced = synced and hand_error is not None and hand_error < 1e-8
    physical_duration = float(np.ptp(sim_times)) if sim_times else 0.
    reference_duration = float(np.ptp(arrays["timestamps"]))
    # Check the policy's own per-tick target stream against the candidate, by name.
    target_agreement = None
    mode_agreement = False
    try:
        play = np.loadtxt(out / "policy_logs/motion_playing.csv",delimiter=",",skiprows=1,ndmin=2)
        target = np.loadtxt(out / "target_motion.csv",delimiter=",",usecols=range(36),ndmin=2)
        ix = np.flatnonzero(play[:,-1]>.5)
        modes = np.loadtxt(out/"policy_logs/encoder_mode.csv",delimiter=",",skiprows=1,ndmin=2)
        mode_agreement = len(ix)>0 and len(modes)>ix[-1] and bool(np.all(modes[ix,-1]==encoder_mode))
        names,_,motors = orders()
        perm = [motors.index(j) for j in names]
        if len(ix) == len(arrays["timestamps"]) and len(target) >= ix[-1]+1:
            target_agreement = float(np.max(np.abs(target[ix,7:][:,perm]-arrays["body_ref_q"])))
    except (OSError,ValueError,IndexError): pass
    reasons = []
    if args.band != "off" or args.startup_support != "none": reasons.append("virtual_support_enabled")
    if args.pelvis_support != "none": reasons.append("pelvis_welded")
    if metadata.get("pelvis_supported_during_source_generation", True): reasons.append("source_pelvis_supported")
    if failure: reasons.append(failure)
    if not completed: reasons.append("episode_not_completed")
    if not synced: reasons.append("hand_body_sync_or_frame_coverage_failed")
    if abs(physical_duration-reference_duration) > .10: reasons.append("physics_not_real_time_with_sonic_reference")
    if target_agreement is None or target_agreement > 1e-5: reasons.append("sonic_target_agreement_unverified")
    if not mode_agreement: reasons.append("sonic_encoder_mode_unverified")
    if not report["physical_grasp_success"]: reasons.append("cube_lift_opposing_contact_and_2s_hold_not_met")
    if report["fall_steps"]: reasons.append("robot_fell")
    if not metadata["source_success"]: reasons.append("source_grasp_not_successful")
    if metadata.get('initial_pose', {}).get('name') == 'neutral_standing_v1':
        if report['robot_self_contact_steps']:reasons.append('neutral_rollout_robot_self_contact')
        if report['inactive_arm_environment_contact_steps']:reasons.append('neutral_inactive_arm_environment_contact')
    expert_valid = not reasons
    report.update(expert_valid=expert_valid,failure_reason=reasons,episode_completed=completed,
                  recorded_frames=len(frame_ids),expected_frames=len(arrays["timestamps"]),frame_coverage_complete=coverage,
                  hand_synchronized=synced,hand_applied_reference_max_error=hand_error,
                  physics_hz=args.physics_hz,physical_duration_seconds=physical_duration,
                  reference_duration_seconds=reference_duration,
                  sync_age_max_seconds=max(sync_ages) if sync_ages else None,
                  sonic_target_max_error_rad=target_agreement,skipped_frames=sorted(set(skipped_frames)),
                  encoder_mode=encoder_mode,encoder_mode_matches=mode_agreement,
                  sonic_body_command_steps=runtime.command_received_steps,
                  body_reference_rmse_rad=float(np.sqrt(np.mean((actual["body_q"]-arrays["body_ref_q"][frame_ids])**2))) if frame_ids else None)
    run_metadata = dict(metadata,controller_type="SONIC C++ DDS body + independent name-mapped hand PD",
                        candidate_type=args.candidate,elastic_band=args.band,band_anchor=args.band_anchor,
                        band_point=runtime.band.point.tolist(),pelvis_supported_during_final_rollout=args.pelvis_support,
                        band_parameters={k:getattr(runtime.band,k) for k in ("kp_pos","kd_pos","kp_ang","kd_ang")},
                        runtime_robot=args.robot,initialization_support=args.startup_support,
                        hand_actuator_types=runtime.hand_controller.types,hand_pd={"kp":12.,"kd":.3},
                        sonic_kp_scale=args.kp_scale,expert_valid=expert_valid,
                        sonic_kd_scale=args.kd_scale,
                        sonic_config=dict(metadata["sonic_config"],encoder_mode=encoder_mode),
                        warmup_control_ticks=args.warmup_frames,
                        body_startup_controller=args.startup_body,
                        live_logging_storage="node-local temporary directory; all logs and videos archived after stop",
                        physics_hz=args.physics_hz,
                        final_sonic_hand_physical_success=expert_valid,failure_reason=reasons,
                        runtime_command=policy.command,
                        sync_clock="SONIC motion_playing.csv active control index; validated against target stream and explicit current_frame log",
                        sonic_binary_sha256=sha256(SONIC_REPO/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'))
    write_json(out / "report.json",report); write_json(out / "metadata.json",run_metadata)
    parent = json.loads((args.episode / "metadata.json").read_text())
    parent.setdefault("validation_runs",{})[name] = {"path":str(out),"candidate":args.candidate,"band":args.band,
                                                     "expert_valid":expert_valid,"failure_reason":reasons}
    # Expose measured arrays from the preferred trial, or promote only a validated success.
    selected = parent.get("selected_validation_run")
    previous_complete = False
    if selected:
        previous_report = args.episode / "validation" / selected / "report.json"
        if previous_report.exists():
            previous_complete = json.loads(previous_report.read_text()).get("frame_coverage_complete", False)
    if expert_valid or (args.candidate == "actual" and not parent.get("expert_valid") and
                        (selected is None or (coverage and not previous_complete))):
        for path in folder.iterdir():
            if path.is_file() and path.name != "metadata.json": shutil.copy2(path,args.episode/path.name)
        for key,value in actual.items():
            # Preserve the reference clock even when a failed run has partial coverage.
            filename = "actual_timestamps" if key == "timestamps" else key
            np.save(args.episode/f"{filename}.npy",value)
        parent.update(selected_candidate=args.candidate,selected_validation_run=name,
                      candidate_type=args.candidate,
                      expert_valid=expert_valid,final_sonic_hand_physical_success=expert_valid,
                      failure_reason=reasons,controller_type=run_metadata["controller_type"],elastic_band=args.band,
                      sonic_config=dict(metadata["sonic_config"],encoder_mode=encoder_mode,kp_scale=args.kp_scale,kd_scale=args.kd_scale),
                      pelvis_supported_during_final_rollout=args.pelvis_support,
                      band_point=runtime.band.point.tolist(),physics_hz=args.physics_hz,
                      shape={k:list(v.shape) for k,v in arrays.items()},
                      reference_sample_count=len(arrays["timestamps"]),
                      reference_duration_seconds=reference_duration,
                      reference_origin=metadata.get("reference_origin",{"candidate":args.candidate}),
                      source_trajectory=metadata["source_trajectory"],
                      validation_report=str(out/'report.json'))
        (args.episode/'info.txt').write_text((args.episode/'info.txt').read_text()+f"\nfinal_expert_valid: {expert_valid}\nselected_validation: {name}\n")
    write_json(args.episode / "metadata.json",parent)
    print(json.dumps(report,indent=2),flush=True)
    if not expert_valid: raise SystemExit(2)


if __name__ == "__main__":
    main()

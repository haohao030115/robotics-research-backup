"""Passive DDS observation; matches native output values, not just nearest time."""
from pathlib import Path
import numpy as np
from diagnostics.control import cpp_parameters


def command_alignment(output, handoffs, dds_events, config):
    if not dds_events:
        return [],dict(matched=0,unmatched=len(handoffs),reason='No DDS commands observed')
    received=np.array([e[0] for e in dds_events]);q=np.array([e[1] for e in dds_events])
    np.savez_compressed(Path(output)/'dds_commands.npz',received_monotonic=received,motor_q=q)
    file=Path(output)/'policy_logs/action.csv'
    if not file.exists():return [],dict(matched=0,unmatched=len(handoffs),reason='No native action log')
    action=np.loadtxt(file,delimiter=',',skiprows=1,ndmin=2)
    p=cpp_parameters(config['sonic']['repository']);rows=[];missing=0
    for h in handoffs:
        f=h['applied_index'];next_action=action[action[:,0]==f+1]
        if len(next_action)!=1 or 'consumed_monotonic' not in h:
            missing+=1;continue
        # Native action.csv[f+1] stores the decoder output from tick f.
        target=next_action[0,5:][p['isaaclab_to_mujoco']]*p['g1_action_scale']+p['default_angles']
        consumed=h['consumed_monotonic']
        candidates=np.flatnonzero((received>=consumed)&(received<consumed+.02))
        matches=[i for i in candidates if np.max(abs(q[i]-target))<2e-5]
        if not matches:missing+=1;continue
        i=matches[0];command_time=float(received[i]);hand_time=h.get('hand_applied_monotonic')
        row=dict(observation_index=h['anchor'],slot=h['chunk_index'],target_index=h['execution_index'],
            actual_execution_index=f,observation_monotonic=h['requested'],rgb_sim_timestamp=h['rgb_sim_timestamp'],
            proprio_sim_timestamp=h['proprio_sim_timestamp'],native_consumption_monotonic=consumed,
            body_command_received_monotonic=command_time,hand_applied_monotonic=hand_time,
            body_command_latency_ms=(command_time-h['requested'])*1000,
            token_consumption_to_body_command_ms=(command_time-consumed)*1000,
            body_hand_command_skew_ms=None if hand_time is None else (hand_time-command_time)*1000)
        if hand_time is not None:row['observation_to_command_ms']=(max(command_time,hand_time)-h['requested'])*1000
        rows.append(row)
    return rows,dict(matched=len(rows),unmatched=missing,match_tolerance_radians=2e-5,
        method='Match full 29 motor targets from next-row native previous_action; first matching DDS receipt in same 20ms tick')

"""Summarize recorded V3 experiments without rewriting source measurements."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[1]
    root=project.parent/'outputs/va_v3'
    get=lambda path:json.loads((root/path).read_text())
    val=get('offline_val.json');strict=get('strict_lowlatency_0000/report.json')
    lat=get('strict_lowlatency_0000/latency.json');full=get('diagnostic_0000/report.json')
    video=get('diagnostic_videos/report.json');partial=get('process_videos/report.json')
    sources=[*sorted((project/'v3').glob('*.py')),*sorted((project/'scripts').glob('*v3*.py')),project/'train_token_v3.py',project/'configs/token_v3.json']
    summary=dict(status='implemented_and_trained_but_strict_timing_acceptance_failed',
        checkpoint=str(root/'train/best.pt'),offline_validation=val,
        strict=dict(run='strict_lowlatency_0000',recorded_frames=strict['recorded_frames'],failure=strict['failure'],timing_contract_pass=False),
        full_video=dict(path=str(root/'diagnostic_videos/expert_vs_v3.mp4'),worker='thread',render_shape=[240,320],diagnostic_only=True,
            recorded_ticks=800,predictions=400,stale_ticks=full['buffer_stale_control_ticks'],task_success=False,verification=video['videos']),
        process_video=dict(path=str(root/'process_videos/expert_vs_v3.mp4'),recorded_ticks=189,truncated=True,verification=partial['videos']),
        tests=(root/'tests.log').read_text().splitlines()[-5:],
        source_sha256={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# VA V3 results — 2026-10-06','',
        '**Code, trained checkpoint, offline evaluation, latency reports and physical videos are available. Strict 25/50-Hz execution acceptance has NOT passed.**','',
        'V2 and SONIC files are unchanged. Current RGB + 96-D proprio produces two 64-D body tokens and two absolute 14-D hand references. No policy history or recurrent memory.','',
        '## Deliverables','',
        '| Deliverable | Path |','|---|---|',
        '| Model, loader, training, evaluation, runtime | `/home/group2/va_training/v3/` and V3 entry scripts |',
        '| Architecture, alignment, reproduction | `/home/group2/va_training/VA_V3.md` |',
        '| Configuration | `/home/group2/va_training/configs/token_v3.json` |',
        '| Trained checkpoint | `train/best.pt` |',
        '| All 5,120 sample alignments | `train/alignment.json` |',
        '| Episode split and provenance | `train/split.json`, `train/provenance.json` |',
        '| Independent held-out metrics | `offline_val.json` |',
        '| CPU / MUSA benchmark | `benchmark_cpu.json`, `benchmark_musa.json` |',
        '| Latest strict timing result | `strict_lowlatency_0000/report.json` |',
        '| Actual command/observation clocks | `strict_lowlatency_0000/execution_alignment.json`, `control_timeline.json`, `dds_commands.npz` |',
        '| Full 16-second diagnostic comparison | `diagnostic_videos/expert_vs_v3.mp4` |',
        '| Aborted process-worker prefix | `process_videos/expert_vs_v3.mp4` (3.8 seconds) |',
        '| Tests | `tests.log` |','',
        'Artifact paths are relative to `/home/group2/outputs/va_v3`.','',
        '## Offline evaluation','',
        f"26 train / 6 held-out episodes, 4,160 / 960 samples. 2,000 MUSA updates. Token MAE {val['token_mae']:.8f}, hand MAE {val['hand_mae']:.8f} rad.",
        f"Slot token MAE: {val['token_mae_per_slot']}; slot hand MAE: {val['hand_mae_per_slot']}.",'',
        'Post-step observation r maps to labels [r+1,r+2], not K=5 offset=2. Training RGB is 10 Hz and consists of independent current observations; no synthetic 25-Hz image stream is invented. Live replanning requests current observations every two ticks.','',
        '## Strict runtime','',
        f"Latest strict test: {strict['recorded_frames']} valid ticks, then `{strict['failure']}`. A full compliant 800-tick run has not been achieved.",'',
        'Independent vision process, MUSA inference, 120x160 render preserving 4:3 aspect, then 128x128 preprocessing. Saved training images are 240x320; lower live raster resolution is an explicit latency tradeoff. The short trial limits percentile interpretation.','',
        '| Stage (ms) | N | Mean | Median | p95 | Max |','|---|---:|---:|---:|---:|---:|']
    for key in ['rgb_acquisition_render','preprocessing','visual_encoder','state_encoder','cross_attention_fusion','body_hand_heads','total_va_forward','sonic_dispatch','observation_to_prediction','observation_to_command']:
        v=lat['stages'][key]
        lines.append(f"| {key} | {v['count']} | {v['mean']:.3f} | {v['median']:.3f} | {v['p95']:.3f} | {v['max']:.3f} |")
    lines+=['',
        'Observation-to-command ends at the later of value-matched body DDS receipt and paired hand application, excluding startup. It includes the intentional slot-1 wait; it is not inference time or physical motor acknowledgement. Native pre-action consumption is logged separately. The unchanged native v4 interface does not schedule by frame_index. Mean inference speed does not establish deadline compliance.','',
        '## Actual physical videos','',
        f"The full 16-second video is the earlier **thread-worker, 240x320, continue-on-miss diagnostic**, explicitly labeled TIMING FAILED. It records 800 ticks, 400 predictions and {full['buffer_stale_control_ticks']} stale ticks. Lift {full['max_lift_m']:.5f} m, hold 0 seconds, no fall, grasp failure. This is not the later aborted strict run.",'',
        'The expert is the recorded successful native GT-token/GT-hand rollout from `outputs/va_v2_video/expert`: lift 0.15050 m, hold 4.375 s. Decoder/runtime hashes, bootstrap hashes, episode and physical criterion match. All three MP4 files were decoded completely: 400 frames at 25 FPS. Rendering uses actual qpos/qvel and never steps physics or substitutes expert joint trajectories.','',
        'The process-worker diagnostic aborts at 189 observed ticks after a missed native snapshot. Its separate 95-frame clip shows only the actual prefix, without padding; its expert outcome is the full expert rollout outcome, not a success within the cropped prefix. Episode 0000 is a training episode; neither video establishes held-out grasp success.','',
        '## Validation and outstanding acceptance','',
        '47 tests: 45 passed; 2 existing optional real-data integration tests skipped. The native loader separately validates all 32 real episodes during training/evaluation. New tests cover alignment and tails, current-only inputs, absence of recurrent memory, gradients through fusion and both heads, checkpoint reload, slot expiry and value-based DDS timing matching.','',
        '**Remaining:** robust end-to-end deadline compliance with the unchanged native consumer, followed by a complete strict closed-loop evaluation. The offline result and failure-diagnostic videos do not resolve this acceptance failure.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for p in sources:
        if p.suffix=='.py':ast.parse(p.read_text())
    print('Reports written; source syntax valid.')
    print('\n'.join(summary['tests']))


if __name__=='__main__':main()

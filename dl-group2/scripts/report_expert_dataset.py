#!/usr/bin/env python3
"""Create a portable Markdown production report and XY coverage SVG from saved results."""
import argparse,html,json
from pathlib import Path


def report(base):
    summary=json.loads((base/'summary.json').read_text());cfg=summary['configuration'];rows=summary['cases']
    audit_path=base/'dataset_audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {'passed':False}
    manifest=json.loads((base/'training_manifest.json').read_text())
    lines=['# XY-only expert dataset', '',
        f"Requested candidates: {summary['sampled_cases']}; processed: {summary['processed_cases']}; training eligible: {summary['valid_experts']}; failed: {summary['failed_cases']}.",
        f"Direct experts: {summary['direct_success']}; corrected experts: {summary['corrected_success']}.",
        f"Mean replay attempts: {summary['average_replay_attempts']:.3f}; mean corrections: {summary['average_correction_iterations']:.3f}.",
        f"Uniform XY bounds (metres): x=[{cfg['x_min']}, {cfg['x_max']}], y=[{cfg['y_min']}, {cfg['y_max']}]; seed={cfg['seed']}.",
        f"Per-position budget: {cfg['max_replays']} replays, {cfg['max_corrections']} corrections. Failed positions are retained; no resampling until success.",
        f"Dataset consistency audit passed: {audit['passed']}.", '',
        '[Training manifest](training_manifest.json) · [Failure manifest](failure_manifest.json) · [Detailed summary](summary.json) · [Dataset audit](dataset_audit.json) · [Episode splits](splits.json)', '',
        '[Region selection evidence](region_selection.json) (when present; chosen before random sampling).', '', '![XY coverage](xy_coverage.svg)', '',
        'Green: direct expert; amber: corrected expert; red: failed/incomplete sample. Hover points in the SVG for case details.', '',
        '| Episode | XY (m) | Class | Replays | Corrections | Lift (cm) | Hold (s) | Failure |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        failure='; '.join(r.get('failure_reason',[])+r.get('packaging_failures',[])).replace('|','/')
        lines.append(f"| {r['episode_id']} | {r['xy'][0]:.5f}, {r['xy'][1]:.5f} | {r['generation_method']} | {r['replay_attempts']} | {r['correction_iterations']} | {100*r.get('final_lift_m',0):.2f} | {r.get('hold_seconds',0):.3f} | {failure} |")
    total=sum(e['state_reference_length'] for e in manifest['episodes']);rgb=sum(len(e['rgb_reference_frame_indices']) for e in manifest['episodes'])
    lines+=['',f'Training pool: {total} state/action frames at 50 Hz; {rgb} RGB frames per camera at 10 Hz.',
        'Default input: head_rgb + body_q + hand_q + instruction. Targets: future body_ref_q[H,29] and hand_ref_q[H,14]. Cube GT and task_overview are excluded from default policy observations.',
        'Root/velocity/cube/contact/SONIC reference and failed replay artifacts remain in each episode. Train/validation separation is by complete episode.',
        '', 'Success rates describe these sampled positions and bounded attempts; they do not establish reliability outside the tested region.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="580" viewBox="0 0 760 580">',
         '<rect width="760" height="580" fill="white"/><g font-family="sans-serif" fill="#222">',
         '<text x="70" y="28" font-size="20">Continuous XY expert production coverage</text>']
    for i in range(6):
        px=80+i*120;py=480-i*84;x=cfg['x_min']+(cfg['x_max']-cfg['x_min'])*i/5;y=cfg['y_min']+(cfg['y_max']-cfg['y_min'])*i/5
        svg += [f'<path d="M {px} 60 V 480 M 80 {py} H 680" stroke="#ddd" fill="none"/>',f'<text x="{px}" y="506" text-anchor="middle">{x:.3f}</text>',f'<text x="65" y="{py+5}" text-anchor="end">{y:.3f}</text>']
    svg += ['<text x="380" y="550" text-anchor="middle">cube x (m)</text>','<text x="15" y="270" transform="rotate(-90 15 270)" text-anchor="middle">cube y (m)</text>']
    for r in rows:
        px=80+(r['xy'][0]-cfg['x_min'])/(cfg['x_max']-cfg['x_min'])*600;py=480-(r['xy'][1]-cfg['y_min'])/(cfg['y_max']-cfg['y_min'])*420
        color='#218c45' if r.get('training_eligible') and r['generation_method']=='direct_expert' else '#c18710' if r.get('training_eligible') else '#ce3535'
        label=html.escape(f"{r['episode_id']} | XY={r['xy']} | {r['generation_method']} | replays={r['replay_attempts']} | corrections={r['correction_iterations']}")
        svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="6" fill="{color}" stroke="white"><title>{label}</title></circle>')
    svg.append('</g></svg>');(base/'xy_coverage.svg').write_text('\n'.join(svg)+'\n')
    return base/'REPORT.md'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(report(a.dataset))

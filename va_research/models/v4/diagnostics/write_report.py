"""Summarize already-computed zero-training diagnostics; no model/runtime changes."""
import json
from pathlib import Path


def main():
    out=Path('/workspace/group2/outputs/va_v4_overfit/kimodo/diagnostics_plan_consistency')
    def read(name):return json.loads((out/name).read_text())
    overlap=read('overlap_metrics.json')['learner_saved'];stability=read('expert_vs_learner_stability.json')
    ensemble=read('temporal_ensemble_offline.json');early=read('early_rollout.json');tracking=read('hand_tracking.json')
    events=read('boundary_jumps.json')['largest'];parity=read('observation_replay_validation.json')
    lines=['# Kimodo V4 zero-training plan-consistency diagnostics',
        '', '**Dominant evidence: 3 — both learner-state/OOD sensitivity and execution-time handoff discontinuity.** The learner stream has much larger same-tick overlap disagreement than the expert stream, and its individual plans are themselves jerky. This is not solely a smooth-plan stitching problem and is not evidence that hand PD is the primary cause.',
        '', '## Controls and replay fidelity',
        '', '- Frozen existing checkpoint; no training, architecture changes, history inputs, consistency loss, corrective collection, SONIC execution/modification, PD changes, or new physical rollout.',
        '- Diagnostic A uses all 162 saved learner plans (161 adjacent pairs). Moving-right means six canonical joints whose peak reference displacement exceeds 0.05 rad: index 0/1, middle 0/1, thumb 1/2. The fixed thumb-0 joint and seven left joints remain included in all14.',
        '- B1 re-infers all 161 canonical expert decision observations, including nine samples excluded from training only because their target horizons end early. No targets were padded. The 152 training-horizon subset is also reported and gives essentially the same stability conclusion.',
        '- B2 reconstructs RGB from saved qpos/qvel at 161 learner capture anchors, using the unchanged scene assets, head_rgb at 320×240, the validated EGL script, and Moore Threads / MTT S4000. Exactly the original V4 preprocessing and single-observation inference are reused.',
        '- **All four re-inferred outputs match the saved learner outputs exactly: max absolute difference 0.0 for body, hand, phase logits, and progress.** Original RGB was not saved; this output parity validates the reconstruction for this checkpoint, not a pixel-equality claim.',
        '- Startup observation anchor −2 was not saved and cannot be reconstructed. It remains in A through its saved prediction, but is excluded from B2 re-inference. Expert anchors are 0,5,…; learner anchors are 3,8,… after startup. No artificial stream shift or new timing convention is introduced.',
        '- Two unrelated controller/evaluator source files differ from the archived runtime contract. They were not executed or modified. All rendering assets and scene code matched their hashes; the direct scene-only reconstruction yielded exact prediction parity. See render_asset_validation.json.',
        '', '## A. Same-tick overlap and plan-internal changes',
        '', 'Boundary compares old h[5] to new h[0] for the same intended tick. Overlap compares old h[5+j] to new h[j], j=0…44. Each value below is mean absolute difference across the named joint set (rad), then a percentile across pairs/slots. Internal compares consecutive slots inside a single 50-step plan; it is a different-time quantity and is not called a same-tick jump. Full predicted overlaps include unexecuted future horizon slots.',
        '', '| Joint set | Quantity | Median rad | P95 rad | Max rad |', '|---|---|---:|---:|---:|']
    for group in ('all14','moving_right'):
        for key in ('boundary','overlap','internal'):
            v=overlap[group][key]['mae'];lines.append(f'| {group} | {key} | {v["median"]:.6f} | {v["p95"]:.6f} | {v["max"]:.6f} |')
    lines+=['', 'RMS and unnormalized L2 distributions are also saved in overlap_metrics.json. **Jitter classification: D (multiple effects)** — large cross-plan revisions plus nontrivial within-plan roughness. For moving-right joints, the median internal change is 0.0494 rad per 20 ms slot; the same-tick boundary P95 is 0.6628 rad. Individual learner plans cannot be described as uniformly smooth.',
        '', 'Largest moving-right boundary events:', '', '| Old → new plan | Intended tick / s | MAE rad | New phase | New auxiliary progress |', '|---|---|---:|---|---:|']
    for e in events[:5]:lines.append(f'| {e["old_plan_id"]} → {e["new_plan_id"]} | {e["intended_tick"]} / {e["time_seconds"]:.2f} | {e["boundary_mae_moving_right"]:.6f} | {["OPEN","CLOSING","HOLD"][e["phase"]]} | {e["progress"]:.6f} |')
    lines+=['', 'boundary_jumps.json stores every event and the top ten, including both full 14-D targets, current phase/progress, capture anchor, and recorded learner body-q/wrist pose. The largest event is plan 62→63 at 6.30 s; selected_plan_handoffs.png overlays both predictions, actual applied target, and measured q on the absolute tick axis.',
        '', '**Applied versus measured:** selecting the saved newest plan reconstructs all 810 applied hand targets with max absolute error 0.0. Thus the revisions reach the applied reference. The measured hand initially begins closure at tick 7 after the applied onset at tick 5, but does not track all rapidly varying targets accurately: moving-right tracking MAE is 0.1622 rad; same-tick joint correlations range from −0.028 to 0.253. Moving-right measured per-tick change median/P95 is 0.02854/0.12579 rad, versus applied 0.05841/0.27999 rad. This is response with substantial tracking error, not proof of faithful tracking or proof of an independent PD defect.',
        '', 'Measured nominal-50-Hz finite-difference jerk median/P95 is 2037.6/5355.1 rad/s³, versus target 27132.0/94769.9 rad/s³. These are control-tick finite-difference diagnostics, not estimates from uniformly sampled simulation-time derivatives. Because the reference is already rough, condition C (smooth targets but independently oscillatory measured q) is not established.',
        '', '## B. Expert versus learner observation streams',
        '', 'The table compares 161 expert observations with 161 reproducibly re-inferred learner observations (160 adjacent pairs each):',
        '', '| Joint set / statistic | Expert median | Expert P95 | Learner median | Learner P95 | Median ratio |', '|---|---:|---:|---:|---:|---:|']
    for group in ('all14','moving_right'):
        for key in ('boundary','overlap','internal'):
            e=stability['expert_all'][group][key]['mae'];l=stability['learner_reinferred'][group][key]['mae']
            lines.append(f'| {group} / {key}, rad | {e["median"]:.6f} | {e["p95"]:.6f} | {l["median"]:.6f} | {l["p95"]:.6f} | {l["median"]/e["median"]:.2f}× |')
    lines+=['', '- Expert phase changes **2** times, learner phase changes **32** times.',
        '- Absolute auxiliary-progress changes: expert median/P95/max **0.000359 / 0.05339 / 0.10008**; learner **0.007563 / 0.92834 / 0.98397**.',
        '- Body same-tick five-slot overlap MAE: expert median/P95 **0.005447 / 0.008689**; learner **0.031542 / 0.058505**. Same-local-slot replan changes are separately recorded; those describe different intended times.',
        '- Expert outputs are relatively stable, not mathematically smooth/exact. Learner boundary disagreement is about **10.79×** larger at the moving-right median, overlap **12.86×**, and internal changes **4.64×**. The expert training-only subset has nearly identical statistics, excluding tail samples as the explanation.',
        '', '**Interpretation:** strong evidence of learner-observation distribution dependence, consistent with covariate shift/OOD sensitivity. A new plan is not intrinsically this discontinuous on the expert stream. The direct handoff exposes the inconsistent learner predictions without damping. This experiment does not separately attribute the input-distribution effect to RGB versus body versus measured hand, and does not prove a universal mathematical OOD boundary.',
        '', 'Fixed-RGB local sensitivity uses four deterministic small Gaussian state perturbations per early observation (q σ=0.001 rad; dq σ=0.01 rad/s; base angular velocity σ=0.001 rad/s; quaternion/gravity unchanged). Expert hand-output MAE median/P95/max: **0.00133 / 0.00177 / 0.00206 rad**. Learner: **0.00090 / 0.00981 / 0.03415 rad**. No phase flips in either stream. Learner tails are more sensitive, but its median is lower and body perturbation sensitivity is not higher: do not call the model uniformly locally unstable. Fixed RGB makes this a function-sensitivity probe, not a physically consistent sensor perturbation.',
        '', '## First second and ordering',
        '', 'Body token error below compares only targets obtained from existing canonical future_execution_indices. Startup compares eight available slots; all other listed plans compare ten. Boundary values are all14 same-tick MAE. Capture time and intended execution start are shown separately.',
        '', '| Learner capture s | Intended start tick | Body token MAE | Hand slot-0 progress | Phase | Auxiliary progress | Boundary MAE rad |', '|---:|---:|---:|---:|---|---:|---:|']
    for r in early['learner']:
        boundary='—' if r['boundary_mae'] is None else f'{r["boundary_mae"]:.5f}'
        lines.append(f'| {r["time_seconds"]:.2f} | {r["start_tick"]} | {r["body_token_mae"]:.5f} | {r["hand_slot0_progress"]:.5f} | {["OPEN","CLOSING","HOLD"][r["phase"]]} | {r["auxiliary_progress"]:.5f} | {boundary} |')
    lines+=['', 'Expert observations at 0.0…1.0 s remain OPEN, with hand slot-0 progress 0…0.00644 and body token MAE 0.01009…0.01400; all per-anchor rows are in early_rollout.json.',
        '', '| Physical tick / s | Body-q RMS rad | Wrist position m | Wrist orientation rad |', '|---|---:|---:|---:|']
    fine=read('early_physics_exact_ids.json')['rows'];fine_by={r['tick']:r for r in fine}
    phy=early['physical_errors'][:1]+[fine_by[t] for t in [2,3,4,5,10,15,20,25,30,35,40,45,50]]
    for r in phy:lines.append(f'| {r["tick"]} / {r["time_seconds"]:.2f} | {r["body_q_rms"]:.5f} | {r["wrist_position_error"]:.5f} | {r["wrist_orientation_error"]:.5f} |')
    lines+=['', 'Fine rows use authored future_execution_indices → body_token_ids → encoder observation_id → recorded post-state IDs, and agree exactly with every available direct sample anchor. This is an identity join, not an offset search or changed convention.',
        '', '**Ordering supported by the records:** state differences are already present at tick 2 (0.04 s). At the first learner replan capture, tick 3 (0.06 s), body-q RMS is 0.0721 rad and wrist position error 0.0299 m; that observation produces the first large boundary revision. Wrist error crosses the descriptive 5 cm threshold at tick 4 (0.08 s), before the revised hand target is applied at tick 5 (0.10 s). Therefore this first executed handoff cannot explain all earlier body/wrist divergence. Conversely, large prediction changes appear before that chosen 5 cm threshold; the threshold is not an expert-manifold membership test. The data support already-changing learner state preceding amplification, not proof of the initial causal disturbance or an exact manifold-exit time.',
        '', '## A2. Hand-only temporal ensemble (offline)',
        '', 'For tick T, average only available plans with start≤T<start+50, weighting by exp(−λ·age_in_5_tick_plan_intervals). Settings are newest-only, weak λ=2.0, and medium λ=0.7; the two older/newer weight ratios are exp(−2)=0.135 and exp(−0.7)=0.497. Body predictions are unchanged. No expert hand target enters the average.',
        '', 'Same-tick revision compares the ensemble with and without the new plan at its intended start T. Actual consecutive-tick target steps/derivatives are separate metrics. Moving-right results:',
        '', '| Setting | Boundary median / P95 rad | Target velocity P95 rad/s | Acceleration P95 rad/s² | Jerk P95 rad/s³ | Closure onset s | Peak closure |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in ensemble['settings'].items():
        m=s['metrics']['moving_right'];b=m['same_tick_boundary_revision']['mae']
        lines.append(f'| {name} | {b["median"]:.5f} / {b["p95"]:.5f} | {m["target_velocity"]["mae"]["p95"]:.3f} | {m["target_acceleration"]["mae"]["p95"]:.1f} | {m["target_jerk"]["mae"]["p95"]:.1f} | {s["closure_onset_tick"]/50:.2f} | {s["peak_closure_progress"]:.5f} |')
    lines+=['', 'Medium weighting reduces moving-right boundary median by **45.8%** and P95 by **56.0%**, and target jerk P95 by **58.6%**. However, all three settings still begin closure at **0.10 s**, versus the recorded expert onset of **7.24 s**, and peak normalized closure falls from **0.9870 to 0.5343**. Smoothing suppresses both jumps and intended amplitude; it does not correct task readiness. Both joint groups, complete derivative statistics and per-joint amplitudes are in temporal_ensemble_offline.json; replayable targets are in ensemble_targets.npz.',
        '', '**Live temporal ensemble: not run.** It was optional. Given unchanged erroneous onset and large amplitude attenuation, this diagnostic stops at offline evidence rather than claiming a physical benefit. No live grasp/lift/hold/body-trajectory change is available; the counterfactual saved body predictions remain identical by construction, which would not guarantee identical body physics in a live hand ablation.',
        '', '## Final decision table',
        '', '| Test | Result | Interpretation |', '|---|---|---|',
        '| Plan-internal hand trajectory smooth? | Relatively smooth on expert; substantially rougher on learner (moving-right median 0.0107 vs 0.0497 rad/slot) | Not just stitching smooth chunks |',
        '| Cross-plan overlap disagreement large? | Learner moving-right median 0.1576 rad across all saved plans | Strong same-future-tick inconsistency |',
        '| Boundary jump large? | Learner moving-right P95 0.6628 rad; max 1.1382 rad | Direct newest-plan handoff exposes large revisions |',
        '| Applied target follows predicted jump? | Yes, newest-plan reconstruction error exactly zero | Not an incorrect-slot scheduling artifact |',
        '| Measured hand follows applied target? | Initially responds; overall tracking is poor, MAE 0.1622 rad | Cannot isolate PD fault while targets are rough |',
        '| Expert-observation plans stable? | Relatively; moving-right boundary median 0.01143 rad, 2 phase transitions | Policy can produce coherent plans on expert observations |',
        '| Learner-observation plans stable? | No; boundary median 0.12332 rad, 32 phase transitions | Supports learner-state covariate shift / OOD dependence |',
        '| Temporal ensemble reduces target jumps? | Yes offline, but onset unchanged and amplitude attenuated | Useful bounded handoff control, not proven task fix |',
        '| Live temporal ensemble improves grasp? | Not run | No physical improvement claim |',
        '', '## Recommended next change — not implemented',
        '', 'Prioritize **body-policy/body-state coverage and more diverse or corrective learner-visited data**: body/wrist drift is measurable before the first revised hand target executes, and predictions are much more coherent on expert observations. This is the strongest source-specific evidence, not a generic “dataset too small” claim.',
        '', 'Treat **controlled handoff/temporal ensemble** as a separate secondary direction, with explicit readiness and closure-amplitude checks. It demonstrably reduces target revisions but cannot recover the correct intent from these unstable learner predictions. Previous-plan context/history is not the first conclusion from this test because the expert stream is already coherent without it. Do not prioritize PD tuning based on this dataset: the smooth-reference prerequisite for that diagnosis is absent.',
        '', 'No recommendation above has been applied. This separates the observed causes: input-distribution dependence generates inconsistent/internal-rough plans, and newest-plan replacement delivers their revisions directly to the hand reference.',
        '', '## Files and plots',
        '', '- Required JSON: [overlap_metrics.json](overlap_metrics.json), [boundary_jumps.json](boundary_jumps.json), [expert_vs_learner_stability.json](expert_vs_learner_stability.json), [temporal_ensemble_offline.json](temporal_ensemble_offline.json).',
        '- Plots: [boundary_jump.png](boundary_jump.png), [overlap_heatmap.png](overlap_heatmap.png), [selected_plan_handoffs.png](selected_plan_handoffs.png), [expert_vs_learner_overlap.png](expert_vs_learner_overlap.png), [phase_progress_body.png](phase_progress_body.png).',
        '- Evidence: [replay validation](observation_replay_validation.json), [early predictions](early_rollout.json), [fine early state joins](early_physics_exact_ids.json), [tracking](hand_tracking.json), [input/checkpoint provenance](provenance.json).',
        '- Source: `va_training/v4/diagnostics/plan_consistency.py`; configurable offline decay flags `--weak-decay` and `--medium-decay`. Three focused indexing/availability tests cover same-absolute-tick overlap, non-mixing of times, and exclusion of unavailable/future plans.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out/'REPORT.md')


if __name__=='__main__':main()

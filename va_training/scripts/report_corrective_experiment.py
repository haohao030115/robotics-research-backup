"""Concise evidence report for the bounded branch-controlled corrective experiment."""
import argparse,json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    audit=json.loads((root/'round1/audit.json').read_text());analysis=json.loads((root/'analysis/report.json').read_text())
    rgb_audit=json.loads((root/'round1/rgb_audit.json').read_text());assert rgb_audit['all_rgb_exact']
    models=['original','nominal_ft','body25','body50','hand25','hand50','combined25','combined50','combined25_scratch']
    models=[m for m in models if (root/f'models/{m}/report.json').exists()]
    reports={m:json.loads((root/f'models/{m}/report.json').read_text()) for m in models}
    log=(root/'regression_tests.log').read_text();tests=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    assert tests and log.strip().endswith('OK')
    round2=(root/'round2/audit.json').exists();best=analysis['round2_model']
    successes=sum(r['task']['closed_loop_pass'] for r in analysis['rollouts'].values())
    conclusion='首轮纠正数据达到预设的离线与闭环漂移改善标准，但抓取是否解决以任务表为准。' if best else '本轮没有达到同时改善纠正验证、闭环漂移且保住名义精度的预设标准；不能宣称 DAgger 成功。'
    lines=['# VA V2 有界纠正数据实验','',conclusion,'',f'实际闭环任务通过 {successes}/{len(analysis["rollouts"])}；改善漂移不等于解决抓取。','',
        '原架构、冻结 policy/release SONIC、专家数据、物理、50 Hz 控制、hand controller 与 K=5 async buffer 保持一致。每个模型/episode 均核对 bootstrap 字节哈希及运行参数。未采用 RL。','',
        '## 数据审计','',
        f"原型 {audit['original_prototype_samples']} 条来自 `{audit['source_rollout']}`，source episode `{audit['source_episode_id']}`。全部输入状态和单步标签逐行核验；encoder 精确重建 {audit['native_encoder_exact_rows']} 条原生 token。",'',
        f"K=5 可用 **{audit['samples']} 条**；末尾 anchor 795 缺完整未来五步，剔除。实际训练 **{audit['train']} 条**、时间块验证 {audit['val']} 条、边界隔离 {audit['guard']} 条。不是 160 条全部训练，也不是独立 learner rollout 验证。",'',
        '|张量|原型|K=5|','|---|---|---|']
    for name in ['rgb','history_mask','body_q','body_dq','hand_q','hand_dq','base_quat','base_ang_vel','previous_action','projected_gravity','target_sonic_token','target_hand_ref']:
        lines.append(f"|{name}|{audit['prototype_shapes'][name]}|{audit['shapes'][name]}|")
    lines+=['', 'RGB 为实际 learner qpos/qvel 的同场景 head_rgb 重渲染，独立重建逐像素核验 160 帧完全一致；模型输入 resize 至 128²。state 为 post-step t 的实际测量及 t−2..t 历史，初始行带 mask。目标是 t+2..t+6。body 标签使用 learner t 的 base quaternion、记录的 initial heading 和 delta_heading=0，以及 intended future q/dq/root orientation；不是盲贴 nominal token，也不是完整纠正控制 action。G1 encoder 该分支不直接使用 learner joint q/dq，低层反馈由 frozen decoder 承担。', '',
        f"五步标签中 {100*audit['nominal_tokens_not_equivalent_fraction']:.2f}% 的完整 token 与名义 token 不同，MAE {audit['token_vs_nominal_mae']:.6f}。hand 标签仍是 intended future nominal schedule；本轮不改变相位标签方案。逐样本来源、tick、对齐和 split 见 `round1/samples.json`，原型与原专家数据未覆盖。",'',
        '## 受控训练','',
        '起点为已有 K=5 offset=2 checkpoint。original 是原权重；nominal_ft 是 0% corrective 的同预算微调对照。六个纠正变体固定 25%/50% 比例：32 样本批次分别为 24+8 / 16+16；500 steps，AdamW lr=1e−4、cosine decay、seed 38，原 normalization 冻结。未做超参搜索。', '',
        'body-only 在 learner 样本只计算 token loss；hand-only 只计算 hand loss；combined 两者都计算。全部变体的 nominal 样本两分支都训练。共享 backbone 可间接影响另一输出头，这与把 learner 标签加入该头的直接监督不同。样本按固定比例有放回采样，未简单拼接。', '',
        '补充 combined25_scratch：从原始 seed=38 初始化训练 2500 steps、lr=1e−3、25% corrective。初始 nominal train metrics 与原始 K=5 初始化核验一致；原 normalization 相同。总步数对应原始 2000 + 微调 500，但混合曝光和 LR schedule 不同，因此是探索性起点对比，不并入六个微调变体的严格因果比较；若实际改善且名义精度保留，可据工程收益准备 round2，机制归因仍有 schedule 混杂。', '',
        '## 离线结果','',
        'N=原名义专家 validation split；C=39 条时间块留出的 learner 输入。每输入含 5 个 token，完整 token accuracy 按 64 维全对统计，未与 scalar FSQ 混淆。', '',
        '|模型|N token MAE|C token MAE|N hand MAE|C hand MAE|N scalar FSQ %|C scalar FSQ %|N whole token %|C whole token %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,r in reports.items():
        n=r['metrics']['nominal_validation'];c=r['metrics']['corrective_validation']
        lines.append(f"|{model}|{n['token_mae']:.6f}|{c['token_mae']:.6f}|{n['hand_mae']:.6f}|{c['hand_mae']:.6f}|{100*n['fsq_level_accuracy']:.2f}|{100*c['fsq_level_accuracy']:.2f}|{100*n['whole_token_exact_accuracy']:.2f}|{100*c['whole_token_exact_accuracy']:.2f}|")
    lines+=['','## 闭环结果','', '所有模型均实际运行相同 0000（known/train）和 0001（held-out）两条 episode。成功条件与前次诊断一致；contact 是有正力的右 thumb 与 index/middle 同帧接触。', '',
        '|模型|episode|成功|contact frames|最大 cube z cm|最大 lift cm|hold s|fall ticks|reach wrist mean cm|closure hand RMS rad|closure hand ref MAE rad|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for key,r in analysis['rollouts'].items():
            if r['model']!=model:continue
            t=r['task'];lines.append(f"|{model}|{r['episode'][-4:]}|{'成功' if t['closed_loop_pass'] else '失败'}|{r['contact_frames']}|{100*t['maximum_cube_height_m']:.3f}|{100*t['max_lift_m']:.3f}|{t['max_continuous_hold_seconds']:.3f}|{t['fall_steps']}|{r['mean_wrist_before_closure_cm']:.3f}|{r['hand_q_rms_closure_rad']:.4f}|{r['hand_reference_mae_closure_rad']:.4f}|")
    lines+=['','## 漂移与假设检验','',
        'body q/dq 为相对 intended nominal trajectory 的逐 tick RMS；base/right-wrist orientation 为 quaternion geodesic degrees；right-wrist position 为欧氏距离。积分 body-state = ∑ sqrt((qRMS/0.1rad)²+(dqRMS/1rad/s)²)/50；closure 是 8–10 s。首次阈值越界未作平滑，不将晚期偶然返回解释成恢复。', '',
        '|模型|episode|wrist 首次 >5 cm s|body-q 首次 >0.1 rad s|peak wrist cm|body-state integral|stale ticks|deadline misses|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for key,r in analysis['rollouts'].items():
            if r['model']==model:
                lines.append(f"|{model}|{r['episode'][-4:]}|{r['wrist_over_5cm_seconds']}|{r['body_q_over_0_1rad_seconds']}|{r['peak_wrist_cm']:.3f}|{r['integrated_body_state_error']:.3f}|{r['buffer_stale_control_ticks']}|{r['execution_deadline_misses']}|")
    lines+=['', '预设工程改善门槛：受监督 corrective validation error 至少下降 10%；两个 episode 平均 reach wrist 或 closure hand error 至少下降 10%，另一状态指标不恶化超过 10%；名义 token/hand MAE 均不恶化超过 20%。这是固定预算下的 effect-size 筛选，不是统计显著性。', '',
        '|模型|reach wrist 相对 0%|body integral 相对 0%|closure hand 相对 0%|纠正离线改善|名义保留|漂移改善|',
        '|---|---:|---:|---:|---|---|---|']
    for model,c in analysis['comparisons'].items():
        lines.append(f"|{model}|{c['mean_wrist_ratio']:.3f}|{c['mean_body_integral_ratio']:.3f}|{c['mean_closure_hand_ratio']:.3f}|{c['corrective_offline_improved']}|{c['nominal_not_materially_degraded']}|{c['drift_improved']}|")
    lines+=['','![0000 distribution shift](analysis/xy_random_v1_random_0000_shift.png)','',
        '![0001 distribution shift](analysis/xy_random_v1_random_0001_shift.png)','',
        '## Phase diagnostic','',
        f"距离配置（每字段 RMS 的平方除以 scale² 再求和）：`{audit['phase_scales']}`。包含 body q/dq、right wrist position、hand q，完整专家时间轴搜索。符号为 matched_expert_index − nominal_index；负值表示更接近早期专家状态。matcher 仅诊断，没有用于 hand relabel。",'',
        '|原纠正 rollout 区间|P05 ticks|median ticks|P95 ticks|mean ticks|','|---|---:|---:|---:|---:|']
    for name,q in audit['phase_lag_ticks'].items():lines.append(f"|{name}|{q['p05']:.2f}|{q['p50']:.2f}|{q['p95']:.2f}|{q['mean']:.2f}|")
    lines+=['','|模型|episode|reach median ticks|closure median ticks|lift median ticks|closure hand target disagreement rad|',
        '|---|---|---:|---:|---:|---:|']
    for model in models:
        for r in analysis['rollouts'].values():
            if r['model']==model:
                q=r['phase_lag_ticks'];lines.append(f"|{model}|{r['episode'][-4:]}|{q['reach']['p50']:.1f}|{q['closure']['p50']:.1f}|{q['lift']['p50']:.1f}|{r['phase_hand_target_disagreement_closure_rad']:.4f}|")
    lines+=['', '完整 phase 分布保存于 `analysis/report.json`。状态已离开专家轨迹时最近点仍可能很远，不能把 nearest index 当成经过验证的 task-phase estimator。手部字段也会影响匹配，存在关联而非独立相位真值。', '',
        '## 纠正状态覆盖','',
        '纠正数据来自原 K=1 learner rollout，主评估沿用 K=5；不能默认两条 visitor distribution 一样。以下是实际闭环状态到 112 个纠正训练快照的最近物理距离，同 phase scales，≤1 表示处于指定尺度邻域；不代表 RGB/任务记忆覆盖。', '',
        '|模型|episode|nearest distance median|全程 ≤1 %|closure ≤1 %|','|---|---|---:|---:|---:|']
    for model in models:
        for r in analysis['rollouts'].values():
            if r['model']==model:
                c=r['corrective_state_coverage'];lines.append(f"|{model}|{r['episode'][-4:]}|{c['nearest_scaled_distance']['p50']:.3f}|{100*c['within_unit_distance_fraction']:.2f}|{100*c['closure_within_unit_distance_fraction']:.2f}|")
    lines+=['',
        '## 结论与唯一下一步','',conclusion]
    for prefix,label in [('body','body 纠正监督'),('hand','hand 纠正监督'),('combined','联合纠正监督')]:
        cs=[c for m,c in analysis['comparisons'].items() if m.startswith(prefix) and not m.endswith('_scratch')]
        lines+=['',f"- {label}：离线改善={any(c['corrective_offline_improved'] for c in cs)}；闭环漂移改善={any(c['drift_improved'] for c in cs)}；同时保留名义精度={any(c['corrective_offline_improved'] and c['drift_improved'] and c['nominal_not_materially_degraded'] for c in cs)}。"]
    nominal_degraded=[m for m,c in analysis['comparisons'].items() if not c['nominal_not_materially_degraded']]
    h1=analysis['h1_supporting_models']
    lines+=['',f"H1（纠正离线误差下降且真实闭环改善）的支持模型：{h1}。" if h1 else '本轮未得到 H1 所要求的“纠正离线改善 + 闭环改善”联合证据；结果支持本批纠正样本未产生预期闭环收益的 H0 方向。它不推翻先前 covariate-shift 诊断，也不能证明增加任意 learner data 必然无效。']
    lines+=['',f"名义 MAE 超过预设 20% 相对退化阈值的模型：{nominal_degraded}。该阈值只用于 token/hand MAE；body25 的 MAE 尚在阈值内，但 nominal whole-token accuracy 已从 14.65% 降至 6.65%，不能称所有名义指标保持不变。仅降低 learner-state 离线误差不能证明 H1；需结合上表实际漂移和任务指标。单 rollout 时间块验证仍不足以证明跨轨迹泛化。"]
    if best:
        assert round2,'Round2 was justified but not prepared'
        lines+=['',f"另一个有界 DAgger round 已获工程证据支持，选择 `{best}`；已运行新的 learner rollout 并另存 `round2/`，不覆盖 expert_round0/corrective_round1。round2 尚未训练，未启动无限循环。",'',
            f'**唯一下一步：使用已准备的 round2 数据，沿用 `{best}` 的监督分支及混合比例做一次同预算纠正微调，并复测相同两个 episode。**']
    else:
        lines+=['','未准备 round2：本轮未满足预设条件，不追加大量相似纠正样本。尚不能在 phase、视觉记忆或覆盖不足之间作唯一归因；最近状态诊断与仍然 nominal 的 hand schedule 是明确未解决的标签限制。', '',
            '**唯一下一步：进行一组保持模型和数据来源固定的 hand-label alignment 对照，检验 nominal 墙钟手部标签的相位错配。**']
    lines+=['','## 复现','',
        '现有结果：`/workspace/group2/outputs/va_v2_corrective/`（host 为 `/home/group2/outputs/va_v2_corrective/`）。逐模型 checkpoint、训练日志、两分布 offline report、每次 physics/hand/wrist/cube/contact 轨迹及 shift NPZ 均保留。', '',
        '完整 fresh-output 复现（host bash，需要现有 va-train 容器）：','',
        '```bash','bash /home/group2/va_training/diagnostics/reproduce_corrective.sh /workspace/group2/outputs/va_v2_corrective_repro','```','',
        f"测试：{tests[1]} 项通过，耗时 {tests[2]} s；见 `regression_tests.log`。本轮全部结果仅一个 seed、每模型每 episode 一次 rollout；无成功率统计或机制唯一性声明。"]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump_json(root/'REPORT.json',{'audit':audit,'analysis':analysis,'training':reports,'round2_prepared':round2,'tests':int(tests[1])})
    print('Report saved',root/'REPORT.md',flush=True)


if __name__=='__main__':main()

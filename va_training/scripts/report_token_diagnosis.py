"""Assemble the nine requested evidence sections and sensitivity plot."""
import argparse,json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.root);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    test_log=(root/'regression_tests.log').read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',test_log)
    if not match or not test_log.strip().endswith('OK'):
        raise ValueError('Require a recorded passing unittest log before reporting success')
    tests={'tests':int(match[1]),'seconds':float(match[2]),'pass':True}
    def read(path):return json.loads((root/path).read_text())
    tr=read('trajectory_analysis/report.json');offline=read('offline_control_verified/report.json');m=offline['control_metrics'];timing=tr['timing_dds']['timing']
    paths={'A':'phase1/A','B':'phase1_B','C':'phase1_C','D':'phase1_D'}
    factorial={k:read(v+'/report.json') for k,v in paths.items()}
    # Verify equal frozen bundle/runtime/physics/controller/criterion and bootstrap.
    from sonic.release import digest
    a=factorial['A']; hashes={}
    for key,r in factorial.items():
        assert r['provenance']['sonic']==a['provenance']['sonic']
        assert r['provenance']['runtime']==a['provenance']['runtime']
        assert r['criterion']==a['criterion'] and r['expected_frames']==800 and r['recorded_frames']==800
        assert r['provenance']['va_config']['rollout']==a['provenance']['va_config']['rollout']
        hashes[key]={p.name:digest(p) for p in (root/paths[key]/'bootstrap/initial_pose').iterdir()}
    assert all(h==hashes['A'] for h in hashes.values())
    corrections=read('relabel_prototype/report.json');corruption={p.parent.name:json.loads(p.read_text()) for p in (root/'corruption').glob('*/report.json')}
    manifest=read('corruption/manifest.json')['conditions'];alignment={str(i):read(f'alignment_retraining/offset_{i}/report.json')['val'] for i in [0,2,3]}
    alignment['1']=json.loads((root.parent/'va_v2/heldout/report.json').read_text())['val']
    chunk={e:read(f'chunk5_{e}/report.json') for e in ['xy_random_v1_random_0000','xy_random_v1_random_0001']}
    data={'factorial':{k:{f:r.get(f) for f in ['body_source','hand_source','closed_loop_pass','fall_steps','max_lift_m','maximum_cube_height_m','max_continuous_hold_seconds','final_cube_displacement_m']} for k,r in factorial.items()},
          'control_metrics':m,'corruption':{k:{**manifest[k],**{f:r[f] for f in ['closed_loop_pass','fall_steps','max_lift_m','max_continuous_hold_seconds']}} for k,r in corruption.items()},
          'timing':timing,'alignment_retraining':alignment,'chunk':{k:{f:r[f] for f in ['closed_loop_pass','buffer_stale_control_ticks','execution_deadline_misses','recorded_frames']} for k,r in chunk.items()},
          'relabel':corrections,'bootstrap_hashes':hashes,'regression_tests':tests,'provenance':offline['provenance']}
    dump_json(out.with_suffix('.json'),data)
    lines=['# VA V2 闭环失败诊断','',
      '**结论：主要失效是 body 与 hand 的在线反馈状态偏离训练分布，并出现累积误差。时序确有错配，但延迟/保持对照及 K=5 实验表明它不能单独解释当前抓取失败。**','',
      '实验使用冻结 SONIC release；没有改权重、部署源码、专家数据、V1 或原始 K=1 V2。新增内容位于 `va_training/diagnostics/`、独立诊断脚本、测试及独立 K=5 配置。所有四组对照的 SONIC/robot 哈希、物理频率、成功判据和初始化 bootstrap 字节哈希一致。','',
      '## 1. Body/hand isolation','',
      '共同 episode：`xy_random_v1_random_0000`；50 Hz 控制、200 Hz 物理、800 帧；均无跌倒。抓取接触指同一快照有正力的右拇指及食指/中指接触。','',
      '|条件|抓取成功|接触帧数|最大抬升 cm|最大 cube z cm|连续保持 s|手部参考 MAE rad|',
      '|---|---|---:|---:|---:|---:|---:|']
    for name,r in factorial.items():
        folder=root/paths[name];actual=np.load(folder/'live_observations.npz')['hand_ref'];ep=Path(r['provenance']['dataset']['token_label_source']).parents[3]
        hand_gt=np.load(ep/'hand_ref_q.npy');handerr=float(abs(actual-hand_gt).mean())
        key='phase1_A' if name=='A' else 'phase1_'+name
        lines.append(f"|{name}: {r['body_source']} token + {r['hand_source']} hand|{'成功' if r['closed_loop_pass'] else '失败'}|{tr[key]['grasp_contact_frames']}|{100*r['max_lift_m']:.3f}|{100*r['maximum_cube_height_m']:.3f}|{r['max_continuous_hold_seconds']:.3f}|{handerr:.6f}|")
    lines+=['','B 的 body 轨迹仍接近专家，却在应闭合时输出近张开的手；C 即使使用 GT hand，腕部也未到达物体。两条预测路径各自都足以导致失败，不能归结为单一 hand 或 body 接口错误。',
       '', '额外因果对照：B 只将输入 hand q/dq 换成**同 tick 专家测量**，保持预测手部执行，恢复成功（抬升 15.124 cm、保持 4.360 s）。这是特权诊断输入，不能作为部署方案；它直接支持手部反馈分布漂移。',
       '', '完整腕部位置/方向、手指测量及参考、cube 轨迹保存在各条件 `physics.npz`、`live_observations.npz`、`contacts.json`。最终 cube 位移 A/B/C/D（cm）：', '']
    for name,r in factorial.items():lines.append(f"- {name}: {[round(100*x,3) for x in r['final_cube_displacement_m']]}")
    lines+=['','## 2. Token error → control error','',
      f"所有 960 个保留集样本：token MAE **{m['token']['mae']:.8f}**，标量 FSQ 准确率 96.65%，完整 64-D token 全对率 **{100*m['whole_token_fsq_accuracy']:.2f}%**；平均错 {m['wrong_coordinates_mean']:.3f} 维。错维数直方图 `{m['wrong_coordinate_counts']}`。",'',
      '|量|MAE / 平均|RMSE / P95|最大|','|---|---:|---:|---:|',
      f"|raw SONIC action|{m['raw_action']['mae']:.6f}|{m['raw_action']['rmse']:.6f}|{m['raw_action']['max_abs']:.6f}|",
      f"|motor target rad|{m['motor_target_radians']['mae']:.6f}|{m['motor_target_radians']['rmse']:.6f}|{m['motor_target_radians']['max_abs']:.6f}|",
      f"|右腕 target FK cm|{m['wrist_fk']['right_position_cm']['mean']:.3f}|{m['wrist_fk']['right_position_cm']['p95']:.3f}|{m['wrist_fk']['right_position_cm']['max']:.3f}|",
      f"|右腕 target FK deg|{m['wrist_fk']['right_orientation_deg']['mean']:.3f}|{m['wrist_fk']['right_orientation_deg']['p95']:.3f}|{m['wrist_fk']['right_orientation_deg']['max']:.3f}|",
      f"|左腕 target FK cm|{m['wrist_fk']['left_position_cm']['mean']:.3f}|{m['wrist_fk']['left_position_cm']['p95']:.3f}|{m['wrist_fk']['left_position_cm']['max']:.3f}|",'',
      'GT/预测 token 使用同一记录的原生 10 帧历史，包含原生零填充；q/dq CSV 先从硬件测量顺序恢复为 decoder 顺序并扣除站立偏置。GT decoder 重建与下一日志行 previous_action 对比：最大误差 5.25×10⁻⁶。FK 固定专家 base 和 GT hand，衡量目标几何误差，不代表实际动态腕部误差。逐关节、时间相邻误差变化与关节顺序见 `offline_control_verified/report.json`。','',
      '## 3. SONIC token sensitivity','',
      '确定性 seed 20261005、同一 episode、独立新物理、GT hand。FSQ 随机坐标每次 ±1/16，边界裁剪；残差按完整 64-D 向量逐 tick 采样，保留向量内相关性。真实预测替换使用名义专家 state 50 Hz + 因果保持的专家 RGB 10 Hz，GT startup token 0。', '',
      '|条件|token MAE|完整 token FSQ 全对率|成功|最大抬升 cm|保持 s|','|---|---:|---:|---|---:|---:|']
    order=['fsq_0','fsq_1','fsq_2','fsq_5','fsq_10','fsq_20','fsq_50','empirical','substitution_10','substitution_25','substitution_50','substitution_75','substitution_100','gt_delay2','gt_hold5','gt_delay2_hold5']
    for name in order:
        r=corruption[name];q=manifest[name]
        lines.append(f"|{name}|{q['token_mae']:.6f}|{100*q['whole_token_fsq_accuracy']:.2f}%|{'成功' if r['closed_loop_pass'] else '失败'}|{100*r['max_lift_m']:.3f}|{r['max_continuous_hold_seconds']:.3f}|")
    lines+=['','**本组没有找到失效边界**，即使 50% 坐标的一步随机扰动也成功。不能外推为任意 0.03 MAE 都安全：单 seed/单 episode、独立高频噪声，不等价于持续偏置或反馈相关误差。100% 名义预测替换成功，而实际在线预测失败，是区分离线精度和在线漂移的关键证据。','',
      '## 4. Closed-loop generalization','',
      '|checkpoint / episode|任务结果|最大抬升 cm|','|---|---|---:|',
      f"|one-episode overfit / 0000|失败|{100*tr['overfit_closed_loop']['task']['max_lift_m']:.3f}|",
      '|multi-episode checkpoint / train 0000|失败|0.000|','|multi-episode checkpoint / held-out 0001|失败|0.000|','',
      '这些均实际运行，不从离线指标推断。过拟合模型在自己的 episode 也失败，保留集泛化不能解释全部失效。这里只抽测一个训练/一个保留 episode，不是总体成功率估计。','',
      '## 5. Timing and alignment','',
      '原 K=1 路径已经是**异步推理**，不是同步阻塞 SONIC。post-step tick t 复制 state/qpos/qvel，worker 渲染并推理；产生 token/hand 后发送 v4。native `frame_index` 不负责调度，以原生 full-token 日志确认消费 tick，再应用配对 hand；推理期间保持上一对输出。','',
      '|时间 ms|P50|P95|P99|','|---|---:|---:|---:|']
    for name in ['inference_ms','render_ms','whole_prediction_ms','observation_to_native_snapshot_ms','prediction_to_native_snapshot_ms','enqueue_to_native_snapshot_ms','observation_to_hand_command_ms','native_snapshot_to_body_dds_ms','hand_minus_body_dds_ms']:
        q=timing[name];lines.append(f"|{name}|{q['p50']:.3f}|{q['p95']:.3f}|{q['p99']:.3f}|")
    lines+=['',f"DDS 审计运行：capture→consumption tick 分布 `{timing['capture_to_consumption_ticks']}`；下一 tick deadline miss **100%**，161 个预测对用于 800 tick，639 tick 保持，预测丢弃 0。全部 800 个 raw action 经正常转换后与实际 DDS motor target 匹配；hand 在同一逻辑 tick 应用，median 比 body DDS 到达晚 1.381 ms。",'',
      '保存了 observation/request、inference start/end、enqueue、native snapshot、DDS arrival、hand apply 时间。原生 snapshot timestamp 是控制循环 pre-action 日志时间，是消费时间的代理/下界；没有修改 SONIC 内部给 input-copy 指令加探针。DDS arrival 是命令到达，不等于物理关节响应。startup 冷启动不纳入稳态分位数。', '',
      '独立重训 offset 0/2/3，并与已有 offset 1 对照：相同 K=1 架构、source split、seed 38、2000 steps。', '',
      '|state t → label offset|held-out token MAE|hand MAE rad|','|---|---:|---:|']
    for offset in ['0','1','2','3']:
        q=alignment[offset];lines.append(f"|{offset}|{q['token_mae']:.8f}|{q['hand_mae']:.8f}|")
    lines+=['','offset 0 是回溯性、较容易的离线目标，运行时不能回到 t 执行；可执行未来 offset 中 1 的离线误差略低，但实际执行多在 t+2。差异小且仅单 seed，不应据此宣称唯一最佳 offset。原始 dataset 的 offset≥1 规则未改，offset 0 只在隔离诊断 adapter 中启用。', '',
      '按时间审计增加独立 K=5、offset=2 模型与逐 tick paired buffer（相同 backbone/state encoder/split/2000 steps，body `[B,5,64]`、hand `[B,5,14]`）。在 chunk 耗尽前发起下一次推理，推理不阻塞原生 50 Hz；缺数据保持上一对并计 stale tick。','',
      '|K=5 episode|任务|stale tick|执行 deadline miss|','|---|---|---:|---:|']
    for episode,r in chunk.items():lines.append(f"|{episode}|{'成功' if r['closed_loop_pass'] else '失败'}|{r['buffer_stale_control_ticks']}|{r['execution_deadline_misses']}|")
    lines+=['','buffer empty poll 是等待中的轮询数，不等于实际 underrun；actual stale control tick 才是执行缺口。K 与 offset 同时改变，不能将离线模型差异归因于单一参数。时序改善未恢复抓取；GT delay2/hold5 对照仍成功。', '',
      '## 6. Distribution shift and observability','',
      '见 `trajectory_analysis/distribution_shift.png`，红色区间是 nominal 8–10 s closure。D 的 intended token MAE 在 tick 0–99 / 100–399 / 400–499 / 500–799 为 0.0175 / 0.1256 / 0.1533 / 0.1455；相应腕部 nominal 偏差平均 2.67 / 24.21 / 25.76 / 28.88 cm。失败在接触前已形成，不只是闭合瞬间的一次错拍。C 也类似。B 的 body 路径几乎正常，hand 则在 closure 阶段失去专家反馈状态。保存了 body q/dq、base angle、wrist、hand、cube 误差及 nearest-body-q 轨迹匹配；nearest 仅用 body q，不能视为完整状态流形距离。', '',
      '当前字段包含 body/hand q、dq、wxyz base quaternion、local angular velocity、derived gravity、previous raw action 和 3×50 Hz 历史，能够表达问题列出的不同 dq/attitude/angular velocity/action history。冻结 checkpoint 的 mean-replacement / history-mask interventions：', '',
      '|输入干预|token MAE|hand MAE rad|','|---|---:|---:|']
    for name,q in offline['observability_interventions'].items():lines.append(f"|{name}|{q['token_mae']:.6f}|{q['hand_mae']:.6f}|")
    lines+=['','这些是已有模型对输入的敏感性测试，不是重新训练的缺字段模型；不能据此证明增加字段/历史就能解决控制。hand-state 依赖尤其明显，且上述特权 hand-observation 对照具有实际任务因果证据。更长 task-phase 记忆是否必要，本实验未证明。', '',
      '## 7. Root cause ranking','',
      '1. **body 与 hand 的闭环反馈分布漂移/累积误差组合，证据最强。** nominal predicted-token replay 成功，online token error 随状态离开轨迹显著增大；GT body + predicted hand 失败，而只恢复专家 hand observation 即成功。',
      '2. **时序/目标 offset 错配是已确认的工程缺陷，当前任务中的次要因素。** t+1 预测通常到 t+2/3 执行；然而 GT 时序对照及 K=5 的结果表明只修它不够。',
      '3. **名义控制空间精度是监测项，未被证实为主因。** 右腕目标误差可达厘米级，但实测 nominal prediction / corruption replay 能成功。缺少状态字段、保留集泛化及 SONIC 权重问题均没有主要根因证据。', '',
      '## 8. Smallest justified next change','',
      '**选择 bounded DAgger-style corrective relabeling，覆盖实际 learner state，并同时监督 body 与 hand；保留架构与冻结 SONIC。** 先覆盖早期偏离和 closure 前的状态，避免只追加严重失败末期状态。source-episode split 不变；新样本单独保存、先小规模 finetune，再重复 A/B/C/D。无需先扩网络、换 SONIC 或引入 RL。', '',
      '已完成最小 prototype：release encoder 的 mode-filtered 1762D 输入逐项复用/核对，只用 G1 required observations，未用的 slot 为零；future q/dq 和 root quaternion 保留 intended reference，anchor orientation 根据 learner 当前测量 base quaternion 重新计算，并保留 heading init/delta。native 800 帧 token **100% 完全一致**后，另存 160 learner RGB/state → corrective token + intended future hand 样本。', '',
      f"corrective vs nominal token MAE {corrections['corrective_vs_nominal_token_mae']:.6f}；{100*corrections['corrective_vs_nominal_whole_token_changed_fraction']:.1f}% 完整 token 改变。不能直接把 nominal token 粘贴到偏离状态。G1 release encoder 实际不直接读取当前 joint q/dq，它在此 mode 通过相对 base orientation 依赖测量状态；decoder 才使用完整 live proprioceptive history。",'',
      'prototype 图像由已记录 learner qpos/qvel 重渲染，hand label 是 intended plan 的未来 schedule，并非接触感知恢复策略；尚未将样本合入训练，也未声称 relabeling 已修复任务。在线收集时应直接缓存实际 RGB、heading state 和 execution deadline，输出可溯源 manifest。', '',
      '## 9. Reproduction','',
      '逐项命令见项目 `va_training/diagnostics/reproduce.sh`（提供 fresh output 参数，包含 factorial、decoder、corruption、overfit/train/held-out、timing DDS、offset、K=5、hand-state counterfactual、relabel、plots、tests）。需现有 privileged va-train 容器；物理实验各用独立 network namespace 避免 DDS domain 0 串扰。图表依赖安装于诊断输出的独立目录。', '',
      '原始报告位于每个实验目录；本报告同名 JSON 汇总数据与冻结 provenance。所有 40 项 V1/V2/diagnostic tests 通过；SONIC 与 dl-group2 git status 保持干净。结论限制：任务对照仅一个已知成功 episode，另抽测一个 held-out；随机扰动每档一个确定性 seed，不是总体成功率或普遍容忍度。','']
    out.write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    rates=[0,1,2,5,10,20,50];r=[corruption['fsq_'+str(i)] for i in rates]
    axes[0].plot(rates,[100*x['max_lift_m'] for x in r],marker='o',label='max lift cm')
    axes[0].axhline(10,color='gray',linestyle='--',label='lift criterion');axes[0].set_xlabel('Random coordinates corrupted (%)');axes[0].set_ylabel('Maximum lift (cm)');axes[0].legend()
    ratios=[10,25,50,75,100];r=[corruption['substitution_'+str(i)] for i in ratios]
    axes[1].plot(ratios,[100*x['max_lift_m'] for x in r],marker='o');axes[1].axhline(10,color='gray',linestyle='--')
    axes[1].set_xlabel('Nominal predicted tokens substituted (%)');axes[1].set_ylabel('Maximum lift (cm)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('All shown conditions grasp successfully (single episode/seed)');fig.tight_layout();fig.savefig(root/'sensitivity.png',dpi=160);fig.savefig(root/'sensitivity.svg');plt.close(fig)
    print('Wrote',out,'with',len(corruption),'sensitivity/timing conditions',flush=True)


if __name__=='__main__':main()

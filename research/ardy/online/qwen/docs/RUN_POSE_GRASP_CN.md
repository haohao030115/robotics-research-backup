# 木块位姿驱动的 ARDY→SONIC 抓取（中文运行指南）

本实验只在 `/workspace/group2/research/ardy/online/qwen` 下写入代码和结果。`fuyuhan` 目录只作为已验证 demo、ARDY 模型和 SONIC 代码的只读来源。

## 1. 进入容器并设置环境

```bash
docker exec -it va-train bash

cd /workspace/group2/research/ardy/online/qwen
export PYTHONPATH=/workspace/group2/research/ardy/online/qwen:/workspace/group2/GR00T-WholeBodyControl:/workspace/group2/workspace/fuyuhan/ardy
export CUDA_VISIBLE_DEVICES=7
export MUSA_VISIBLE_DEVICES=7
```

## 2. 先跑位姿同步冒烟测试

测试会把 MuJoCo 中的木块横向移动 4 cm，然后重新读取位置和方向。代码不手动改抓取目标，只比较四组自动生成的关键帧是否同步平移。

```bash
python -m qwen_ardy.smoke_block_pose_sync \
  --scene /workspace/group2/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --ardy-repo /workspace/group2/workspace/fuyuhan/ardy \
  --output-root runs/block_pose_smoke \
  --demo-scene /workspace/group2/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --demo-simulation /workspace/group2/GR00T-WholeBodyControl/outputs/ardy_grasp/arms_at_sides_aligned/simulation.npz \
  --demo-result /workspace/group2/GR00T-WholeBodyControl/outputs/ardy_grasp/arms_at_sides_aligned/result.json
```

通过标志：终端输出 `pass: true`。报告在 `runs/block_pose_smoke/smoke_report.json`。

## 3. 读取木块位姿并生成 ARDY/SONIC reference

```bash
RUN_DIR="/workspace/group2/outputs/ardy_online/pose_grasp_$(date +%Y%m%d_%H%M%S)"

python -m qwen_ardy.prepare_pose_driven_grasp \
  --scene /workspace/group2/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --output-dir "$RUN_DIR/reference/target_block" \
  --ardy-repo /workspace/group2/workspace/fuyuhan/ardy \
  --sonic-root /workspace/group2/GR00T-WholeBodyControl \
  --demo-scene /workspace/group2/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --demo-simulation /workspace/group2/GR00T-WholeBodyControl/outputs/ardy_grasp/arms_at_sides_aligned/simulation.npz \
  --demo-result /workspace/group2/GR00T-WholeBodyControl/outputs/ardy_grasp/arms_at_sides_aligned/result.json \
  --gpu 7
```

这一步在开始时只读取一次 `wood_block` 的世界位置、旋转矩阵和 free-joint qpos，之后不修改木块。生成的 `block_pose_plan.json` 记录木块快照、坐标标定、四阶段目标以及 IK 误差。

## 4. 在 noVNC 中看 SONIC+MuJoCo 执行

先确保 VS Code 已转发 6080，并在浏览器打开 `http://127.0.0.1:6080/vnc.html`。然后在容器里执行：

```bash
cd /workspace/group2/GR00T-WholeBodyControl
export PYTHONPATH=/workspace/group2/research/ardy/online/qwen:/workspace/group2/GR00T-WholeBodyControl:/workspace/group2/workspace/fuyuhan/ardy
export DISPLAY=:1
export XAUTHORITY=/root/.Xauthority

python gear_sonic_deploy/reference/test_ardy_grasp.py \
  --viewer \
  --arms-at-sides \
  --robot-x 0.24 \
  --robot-y -0.03 \
  --hand-kp 6 \
  --motion-dir "$RUN_DIR/reference" \
  --scene gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --output-dir "$RUN_DIR/simulation"
```

`$RUN_DIR/simulation/result.json` 包含抓取是否成功、木块最大抬升高度、手与木块最小距离和最终接触数。

## 5. 人工 text 交互（不自动续段）

```bash
cd /workspace/group2/research/ardy/online/qwen
export PYTHONPATH=/workspace/group2/research/ardy/online/qwen:/workspace/group2/GR00T-WholeBodyControl:/workspace/group2/workspace/fuyuhan/ardy
export DISPLAY=:1
export XAUTHORITY=/root/.Xauthority
export CUDA_VISIBLE_DEVICES=7
export MUSA_VISIBLE_DEVICES=7

HUMAN_RUN="/workspace/group2/outputs/ardy_online/human_$(date +%Y%m%d_%H%M%S)"

python -m qwen_ardy.human_interactive \
  --viewer \
  --source /workspace/group2/GR00T-WholeBodyControl/gear_sonic_deploy/reference/ardy_grasp_sides/target_block \
  --reference-root "$HUMAN_RUN/references" \
  --output-dir "$HUMAN_RUN/output" \
  --scene gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --ardy-repo /workspace/group2/workspace/fuyuhan/ardy \
  --segment-seconds 8
```

直接输入英文动作描述并回车。当前动作会完整播放 8 秒，然后保持在末帧；不会自动生成下一段。你再输入新 text 才会生成新动作。执行中提前输入的新 text 会排队，不会丢失。

常用命令：

```text
/status   查看当前状态
/open     张开右手
/close    闭合右手
/quit     保存并退出
```

注意：ARDY 不能从 `block` 这个词自动得到 MuJoCo 中的坐标。本项目把无数值的成功 demo text 交给 ARDY，把精确的位姿通过 constraint 交给 ARDY。

## 当前验证边界

- 4 cm 横向位移的自动目标同步冒烟测试已通过，误差为浮点精度级。
- fuyuhan 原始 reference 的端到端抓取是成功基线。
- 新生成 reference 的基线场景端到端还没有稳定复现抓起；不要仅根据约束同步测试就宣称抓取成功，应以 `simulation/result.json` 为准。

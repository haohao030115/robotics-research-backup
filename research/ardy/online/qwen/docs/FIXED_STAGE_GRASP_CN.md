# 固定分阶段抓取运行指南

新流程中，Qwen3-VL-2B 只在启动时执行一次任务路由。如果识别为 `grasp_block`，就卸载 Qwen 并释放显存，然后执行 fuyuhan 成功 demo 的固定五阶段连续 reference。动作期间不再调用 Qwen。

## 运行带可视化的完整流程

```bash
docker exec -it va-train bash

cd /workspace/group2/research/ardy/online/qwen
export PYTHONPATH=/workspace/group2/research/ardy/online/qwen:/workspace/group2/GR00T-WholeBodyControl:/workspace/group2/workspace/fuyuhan/ardy
export ARDY_REPO=/workspace/group2/workspace/fuyuhan/ardy
export DISPLAY=:1
export XAUTHORITY=/root/.Xauthority

RUN_DIR="/workspace/group2/outputs/ardy_online/fixed_stage_$(date +%Y%m%d_%H%M%S)"

python -m qwen_ardy.fixed_stage_grasp \
  --task-prompt "Please grasp the wooden block and lift it" \
  --qwen-model /workspace/group2/workspace/ljy/models/Qwen3-VL-2B-Instruct \
  --qwen-device auto \
  --source /workspace/group2/GR00T-WholeBodyControl/gear_sonic_deploy/reference/ardy_grasp_sides/target_block \
  --scene /workspace/group2/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml \
  --output-dir "$RUN_DIR" \
  --sonic-root /workspace/group2/GR00T-WholeBodyControl \
  --viewer
```

在 VS Code 转发 6080 端口，然后打开 `http://127.0.0.1:6080/vnc.html`。

`auto` 会使用当前进程可见的第一张 MUSA 卡。如果已设置 `MUSA_VISIBLE_DEVICES=7`，物理 7 号卡在进程内的逻辑编号是 `musa:0`，不应再写 `musa:7`。

若只想重复跑物理抓取测试，不想每次重新加载 Qwen，增加：

```bash
--skip-qwen
```

## 输出

- `task_route.json`：Qwen 的唯一一次任务路由。
- `fixed_stage_plan.json`：五个固定 prompt、帧区间、成功 demo 手腕位姿和抓取中心。
- `simulation/result.json`：MuJoCo 原始物理结果。
- `result.json`：路由、阶段和抓取结果汇总。

成功判定以 `result.json` 中的 `success: true` 为准。

# Method B：固定约束分阶段抓取

## 已验证结果

2026-10-07 端到端仿真通过：

- 木块最终抬升：`0.102371 m`
- 手—木块最小距离：`0.015431 m`
- 最终接触组：`2`
- 稳定抓持抬升：`0.20 s`
- 机器人最低根高度：`0.770153 m`
- 结果：`success: true`

证据：

```text
/workspace/group2/outputs/ardy_online/method_b_fixed_07/method_b_plan.json
/workspace/group2/outputs/ardy_online/method_b_fixed_07_validation/result.json
/workspace/group2/outputs/ardy_online/method_b_fixed_07_validation/simulation.npz
```

## 算法流程

1. 从 fuyuhan 成功 demo 的 `constraints.json` 读取原始 full-body、root2d 和末端约束。
2. 按 ARDY 25 Hz 时间轴分为四段：`0–75`、`75–150`、`150–200`、`200–300`。
3. Qwen 每段读取该阶段起始画面，只能对已验证的完整 prompt 做轻微阶段强调。
4. 每段独立调用 ARDY，输入切分后的原始固定约束和 Qwen prompt。
5. 新段第 0 帧使用上一段真实生成的末帧作 full-body 连续约束。
6. 将短段 ARDY 残差限制在成功 demo 的固定约束流形附近，防止软约束采样偏离木块。
7. 对每个边界前 20 帧做平滑连续投影，第 0 帧严格等于上一段末帧。
8. 四段拼成一条 12 s 连续 reference 再交给 SONIC，不在播放时重置运动相位。
9. 手腕对齐阶段后由接触驱动手指闭合；抓住后执行 ARDY 抬升段。

## 一条命令复现

先确认没有旧仿真占用同一 DDS 域：

```bash
docker exec va-train bash -lc \
  "pgrep -af 'g1_deploy_onnx_ref|python.*qwen_ardy' || true"
```

然后执行：

```bash
docker exec -it va-train bash
cd /workspace/group2/research/ardy/online/qwen
chmod +x run_method_b_fixed.sh
./run_method_b_fixed.sh
```

带 MuJoCo 可视化：

```bash
./run_method_b_fixed.sh \
  "/workspace/group2/outputs/ardy_online/method_b_viewer_$(date +%Y%m%d_%H%M%S)" \
  --viewer
```

运行结束后查看：

```bash
cat /workspace/group2/outputs/ardy_online/method_b_*_validation/result.json
```

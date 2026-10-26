# Method B 真正在线分阶段抓取

## 最终流程

这版不会在仿真开始前生成并拼好全部动作。流程如下：

1. SONIC 播放当前阶段。
2. 阶段结束后停在最后一帧，继续运行控制循环；未来参考窗口全部填充为末帧。
3. MuJoCo 继续正常步进，夹爪命令保持为上一阶段末值。
4. 保存此刻的仿真图像和机器人/木块状态，现场调用本地 Qwen。
5. Qwen 根据该阶段实时画面选择一个轻微限定词，并输出视觉说明。若限定词已经存在于成功 prompt 中，则保留成功 prompt 的原文，避免重复措辞扰动 ARDY；否则最多追加一个限定词。
6. ARDY 只生成下一阶段；约束链沿用已验证的 fuyuhan/Method B 约束。
7. 把截至当前已经生成的轨迹前缀统一转换，从中切出新段，保证所有阶段共享相同根坐标上下文。
8. SONIC 将新段前 20 帧连续对齐到当前保持末帧；收到启动信号后才继续播放。

初次启动会加载 Qwen 和 ARDY 的本地权重，通常明显慢于后续阶段推理。模型加载发生在仿真启动前。模型常驻后，本次验证中第 2～4 段 ARDY 采样约为 1～1.3 秒。

## 运行

进入 `va-train` 容器后执行：

```bash
cd /workspace/group2/research/ardy/online/qwen
chmod +x run_method_b_online_hold.sh
./run_method_b_online_hold.sh
```

脚本默认带 `--viewer`。MuJoCo 物理使用稳定的 EGL 后端，noVNC 桌面上会出现名为 `MuJoCo live - Method B` 的独立实时监视窗口；左上角显示阶段、当前 phase 和仿真时间。不要另外启动第二个 SONIC 控制器或第二个仿真进程，否则 DDS `lo` 会互相干扰。脚本带单实例锁，重复运行会直接提示并退出。

脚本默认使用 GPU 5 运行 ARDY，Qwen 使用 GPU 7。需要时可以这样覆盖 ARDY 使用的卡：

```bash
ARDY_GPU=6 ./run_method_b_online_hold.sh
```

## 打开 noVNC

1. 在 VS Code 的“端口/Ports”面板确认远程端口 `6080` 已转发。
2. 浏览器打开 `http://127.0.0.1:6080/vnc.html`。
3. 输入密码 `123456` 并连接。
4. 然后在容器终端运行上面的脚本。模型加载完后，桌面会出现实时仿真窗口。

不要重新启动 `websockify`，服务器已有服务监听 6080。

## 如何确认它不是预生成

每次运行目录中的 `stage_events.jsonl` 会按时间记录：

- `stage_held`：上一段已经结束并主动保持；
- `qwen_requested_after_live_hold`：保持后才读取现场画面；
- `constraint_completed_ardy_requested`：现场约束完成后才请求 ARDY；
- `cumulative_conversion_completed_switch_requested`：新段完成并请求连续接入；
- `continuous_segment_started`：SONIC 收到启动信号并恢复播放。

对应阶段的 `*_live.png` 也是在 `stage_held` 之后生成的。

## 已通过的无界面验证

结果目录：

```text
/workspace/group2/outputs/ardy_online/method_b_online_hold_07
```

`result.json` 的关键结果：

- `success: true`
- 最终木块抬升：`0.1186357533 m`
- 最大木块抬升：`0.1397688098 m`
- 手到木块最小距离：`0.0133040596 m`
- 最终接触组：`2`
- 最低根高度：`0.7690993544 m`
- 推理等待期间 MuJoCo 保持运行：`true`

完整阶段事件、Qwen 输出、ARDY 输出、SONIC 日志和仿真采样都保存在同一运行目录。

## 日志与故障判断

每次运行都会创建 `runs/method_b_online_hold_时间戳/`。最终只以 `result.json` 中的 `success: true` 为成功标准。若 SONIC 在启动阶段丢失 LowState，程序现在会立即报错并附带控制器日志尾部，不会留下一个“窗口开着但机器人不动”的假运行。

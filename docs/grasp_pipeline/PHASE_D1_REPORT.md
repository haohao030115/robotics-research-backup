# Phase D.1 — Closed-loop CLI 修复与真实验证（2026-10-09）

Phase D.1 已完成。`python -m grasp_pipeline eval closed` 现在可以进入既有 native SONIC + DDS + MuJoCo 闭环，并保留完整运行证据。没有修改 V4 模型、loss、归一化、SONIC encoder/decoder、DDS 调度、`TabletopDDS.step`、PD 控制、canonical indexing 或物理成功判据。

## 实际根因

1. 最初复现使用的场景文件不是 JSON，CLI 在运行时前即退出；该失败没有进入 DDS 或 MuJoCo。
2. 修正场景后，wrapper 预先创建了输出目录，而既有 `v4.rollout.run()` 要求输出目录必须是 fresh directory，因此在 native 启动前抛出 `FileExistsError`。
3. 修正目录生命周期后，native runtime 已启动并完成 DDS 初始化，但在 MuJoCo `Renderer` 创建处抛出 `OpenGL platform library has not been loaded`。闭环 CLI 没有执行 canonical graphics/EGL bootstrap，worker 没有继承已验证的 MTT 图形环境。

## 修复

修改 `grasp_pipeline/cli.py`：

- 在调用既有 `v4.rollout.run()` 前验证场景 JSON，并要求输出目录不存在；不再抢先创建目录。
- 在导入/启动闭环 runtime 前执行 `canonical_va.rendering.bootstrap('auto')`，使 EGL、DRI、vendor 和动态库路径进入同一进程。
- 保留 explicit checkpoint/scenario/seed provenance。
- 捕获 runtime 异常并保存 `execution_status.json`、scenario、provenance、退出阶段和错误信息；成功后同样保存明确的 runtime/grasp 状态。

没有修改 `v4.rollout` 或 native runtime。比较视频使用同一 bootstrap 运行既有 `v4.comparison`，没有修改比较器。

## 真实验证

通过 isolated network namespace 在 `va-train` 中执行一次 bounded rollout，使用已有 qualified checkpoint（从 `outputs/va_v4_kimodo19` 复制到临时输入目录，原 checkpoint 和历史目录未改写）。结果：

- CLI exit code：0
- `strict_runtime_pass=true`
- `rollout_complete=true`
- 810/810 physics ticks；162 plans
- scheduler：0 uncovered ticks、0 deadline misses、0 repeated ticks
- `physical_grasp_success=false`
- `max_lift_m=0.0`，`max_continuous_hold_seconds=0.0`，`fall_steps=0`
- renderer：`GL_VENDOR=Moore Threads`，`GL_RENDERER=MTT S4000`，`GL_VERSION=4.5 build 57493aa1-dirty`，`MUJOCO_GL=egl`

任务失败是有效的 evaluation 结果；runtime 没有失败，也没有放宽 offline gate 或重复运行直到成功。

## 产物

主运行目录：[`closed_run3`](../../outputs/grasp_pipeline_phase_d1/closed_run3)

- [`report.json`](../../outputs/grasp_pipeline_phase_d1/closed_run3/report.json)
- [`metrics.json`](../../outputs/grasp_pipeline_phase_d1/closed_run3/metrics.json)
- [`physics.npz`](../../outputs/grasp_pipeline_phase_d1/closed_run3/physics.npz)
- [`execution_status.json`](../../outputs/grasp_pipeline_phase_d1/closed_run3/execution_status.json)
- [`evaluation_provenance.json`](../../outputs/grasp_pipeline_phase_d1/closed_run3/evaluation_provenance.json)
- [`renderer_worker.json`](../../outputs/grasp_pipeline_phase_d1/closed_run3/renderer_worker.json)
- [`report.html`](../../outputs/grasp_pipeline_phase_d1/closed_run3/report.html)

既有比较器产物：[`closed_comparison`](../../outputs/grasp_pipeline_phase_d1/closed_comparison)

- [`expert_vs_v4.mp4`](../../outputs/grasp_pipeline_phase_d1/closed_comparison/expert_vs_v4.mp4)
- [`video_metadata.json`](../../outputs/grasp_pipeline_phase_d1/closed_comparison/video_metadata.json)
- [`failure_localization.json`](../../outputs/grasp_pipeline_phase_d1/closed_comparison/failure_localization.json)

失败尝试的 stdout/stderr、exit code 和输出目录均保留在 [`outputs/grasp_pipeline_phase_d1`](../../outputs/grasp_pipeline_phase_d1)，包括 pre-runtime fresh-directory failure 与 missing-EGL failure。

## 检查

- `python3 -m py_compile grasp_pipeline/cli.py` 通过。
- Phase C 轻量回归套件仍为 21 项通过，记录在 [`tests_phase_c.log`](../../outputs/grasp_pipeline_phase_b/tests_phase_c.log)。
- 真实闭环命令通过；失败任务结果与 runtime failure 分离记录。
- 比较器在硬件 EGL bootstrap 下成功生成视频；视频按 exact canonical observation IDs 与实际 learner tick 对齐，没有 padding 或插值。
- 原有 checkpoint、native controller、DDS/MuJoCo 实现和历史 artifacts 未覆盖。

## 限制

本次 qualified checkpoint 的任务表现是物理抓取失败，不能据此宣称抓取成功。比较器日志仍显示底层 driver 的 `musa_dri.so` 警告，但最终 GL identity 是 Moore Threads/MTT S4000，且 MuJoCo EGL renderer 与完整闭环均成功。后续若更换主机，应重新核对 graphics profile 和硬件 renderer。

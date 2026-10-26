# 当前代码与清理说明

当前流程包含 13 个物理轨迹核心模块和 7 个数据规范/生产/训练读取模块。旧 supported-pelvis 入口、空场景 tracking 实验、旧 reference 导出器、DDS 相机采集入口和安装脚本已删除；原来的共享函数已抽出并验证等价。

| 文件 | 当前用途 |
|---|---|
| `generate_freebase_expert.py` | 生成 free-base 源轨迹；全身逆动力学控制器也用于回放前站立 |
| `refine_freebase_reference.py` | 基于实测手腕误差校正 SONIC reference |
| `run_sonic_grasp.py` | SONIC body + 独立 hand 的真实物理回放 |
| `verify_expert_episode.py` | joint mapping、物理指标、同步和文件一致性验证 |
| `export_expert_episode.py` | 从认证成功的回放导出完整数据集 |
| `render_expert_rollout.py` | 从保存的实际状态渲染三路视频与 RGB |
| `grasp_reference.py` | arm-only wrist IK 和 reach/grasp/lift 时间序列；只更新规划状态 |
| `reference_io.py` | 共用 CSV/NPY reference 写入函数 |
| `expert_trajectory.py` | joint order 源码校验、按名称映射、状态记录、接触判定和数据读写 |
| `hand_trajectory_controller.py` | 独立 14-DoF hand PD |
| `scene.py` | 组合官方 G1 模型与项目桌面场景 |
| `cases.py` | 读取 case、初始化方块位置 |
| `project_paths.py` | 项目路径、当前默认 episode、只读官方模型路径 |

新增的数据集模块：

| 文件 | 用途 |
|---|---|
| `episode_schema.py` | 统一 metadata、通用 neutral 双臂配置、RGB 对齐校验 |
| `standardize_episode.py` | 只补充旧 episode 元数据，保留原轨迹和图像 |
| `audit_neutral_pose.py` | 初始碰撞间隙与实测姿态的补充审查 |
| `run_xy_pilot.py` | 有限次数的小范围位置测试、校正、汇总与训练清单 |
| `produce_experts.py` | 连续 XY 采样、有限重试、筛选与可续跑数据生产 |
| `vision_action_dataset.py` | RGB/state/instruction → future body/hand chunk，episode split |
| `audit_expert_dataset.py` | 整批物理/文件/图像对齐、动作块和 split 检查 |

`configs/neutral_standing.json` 定义左右手通用起点；完整字段与运行说明见
[数据集规范](dataset_schema.md)。

`grasp.py`、`pick_red_cube.py`、`generate_sonic_reference.py`、`generate_sonic_reference_actual.py`、`generate_expert_candidates.py` 已移除。当前代码不再通过这些旧入口间接导入其他历史控制器。映射解析从 `audit_sonic_reference.py` 移入 `expert_trajectory.py`，审查功能由统一验证器负责。

runner 默认读取 `freebase_wbc_002/metadata.json` 的 selected_candidate，启动使用 freebase-wbc，warmup 为 0，elastic band 关闭。新 episode 尚未选择候选时默认使用 actual。验证器默认检查 freebase_wbc_002。

## 官方依赖

`dl-group2/third_party/GR00T-WholeBodyControl` 是未初始化的 submodule，锁定 commit 为 `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`。当前复用同一 commit 的 `~/GR00T-WholeBodyControl`，由 project_paths.py 校验版本。

`gear_sonic/` 是官方 Python 模型、配置和仿真通信代码；`gear_sonic_deploy/` 是官方部署程序、策略权重和 reference 读取实现。这两个目录是当前流程所需依赖。

本次在官方 checkout 中检查并清理了两项旧实验遗留：

- `gear_sonic/utils/mujoco_sim/simulator_factory.py`：此前把 init_channel 改为 pass 的补丁已恢复到 HEAD。当前 runner 直接初始化 DDS 并使用 UnitreeSdk2Bridge，不经过这个 factory。
- `gear_sonic_deploy/reference/example/grasp_center/`：旧抓取 reference 副本已删除。

恢复前的完整补丁和删除文件列表保存在 [code_cleanup.json](code_cleanup.json)。官方源码、已有部署二进制、模型权重、已安装环境和第三方依赖均保留。官方 checkout 的 git status --short 已为空。

## 数据保留范围

认证成功的 `outputs/expert_episodes/freebase_wbc_002/` 保留完整，包括成功回放、视频、源轨迹、reference 校正依赖链和当时的验证记录。它们是当前数据的来源证据。独立的失败源 `freebase_wbc_001` 已删除，其失败报告摘要保存在清理记录中。

原有 `outputs/recordings/` 是独立录像，无法仅凭文件名证明是重复数据，本次保留。旧无重力 playground 和 experimental scene、旧方法说明、过期迁移清单及 MuJoCo 调试日志已移除；项目课程目标仍在 project_overview.md。

验证结果见 [cleanup_validation.json](cleanup_validation.json)。当前运行说明见 [freebase_expert.md](freebase_expert.md)。

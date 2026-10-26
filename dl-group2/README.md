# dl-group2 — G1 红色方块抓取

HKU DASC7606C Track 4：在 MuJoCo 中让 Unitree G1 自由站立，完成桌面红色方块的 **reach → grasp → lift → hold**，并保存供 imitation learning / Vision-Action 使用的 reference、实际状态和图像。

## 当前结果

已验证一个中心位置方块的 expert episode：`freebase_wbc_002`，`expert_valid=true`。成功数据保存在当前工作区的 `outputs/expert_episodes/freebase_wbc_002/`。

| 验证项 | 已保存的成功 SONIC 回放 |
|---|---|
| 身体控制 | SONIC，29 DoF |
| 手部控制 | 独立 joint-space PD，14 DoF |
| 物理条件 | 重力开启、free base、真实手指接触；无 pelvis weld、elastic band、物体绑定或外力 |
| 方块最终抬升 | 0.13359 m |
| 连续稳定 hold | 4.385 s，拇指与食指均有正接触力，方块离开桌面 |
| 完整性 | 650 帧完整同步，无跌倒 |
| 采样 | 状态/reference 50 Hz；三路 RGB 10 Hz |

同一 reference 连续两次回放成功；代码清理后的独立复测也通过，抬升 0.13423 m、hold 4.38 s。早期机器人与桌面的接触已记录，成功 hold 期间没有机器人或方块接触桌面。当前结果验证的是这一条中心位置轨迹，其他方块位置尚未完成验证。

**Git 只保存源码、场景、配置和文档。** `outputs/` 中的轨迹、校正候选、视频和 RGB 数据被 `.gitignore` 排除，保留在现有工作区；仅 clone 仓库不会获得这些数据。下面的“回放已有 expert”命令要求该 episode 目录已存在。

## 方法与控制接口

源轨迹使用全身逆动力学维持站立，手腕 IK 规划抓取动作，独立 hand PD 控制手指。随后根据 SONIC 实测的手腕误差校正 body reference，再进行完整物理回放验证。

```text
同一条 reference 时间轴（50 Hz）
├── body_ref_q / body_ref_dq [T,29] → SONIC → 29-DoF body
└── hand_ref_q / hand_ref_dq [T,14] → 独立 hand PD → 14-DoF hand
```

SONIC 不控制手指。hand PD 当前使用位置目标，Kp=12、Kd=0.3；hand_ref_dq 保存供后续使用。两者按 SONIC 播放帧索引同步。回放开始前由无支撑全身控制器维持站立，播放期间身体完全交给 SONIC。

body 顺序从官方 deployment 源码读取并交叉校验，body/hand 均通过 joint name 映射到模型。完整 29/14 joint order 和判定条件见 [expert 技术说明](docs/freebase_expert.md)。

## 环境与依赖

当前工作区使用已有环境，无需重新安装：

- 项目：`~/dl-group2`
- 官方代码、模型和 SONIC 部署程序：`~/GR00T-WholeBodyControl`
- Python：`~/GR00T-WholeBodyControl/.venv_sim/bin/python`
- 已验证 GPU 节点：`gpu-4080-413`

`third_party/GR00T-WholeBodyControl` 当前是未初始化的 submodule。程序会只读复用外部官方 checkout，并检查其 commit 与 submodule 锁定版本一致。可通过 `SONIC_REPO` 指定其他官方 checkout。`gear_sonic/` 和 `gear_sonic_deploy/` 是必要的官方依赖目录；当前官方 Git 工作区已恢复干净。

运行 SONIC 时，同一节点应没有其他 domain-0 controller。先进入 GPU 节点并启用环境：

```bash
ssh gpu-4080-413
cd ~/dl-group2
source ~/GR00T-WholeBodyControl/.venv_sim/bin/activate
export PYTHONPATH="$HOME/GR00T-WholeBodyControl${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

## 回放已有 expert

在上述环境中运行：

```bash
python scripts/run_sonic_grasp.py \
  --episode outputs/expert_episodes/freebase_wbc_002 \
  --candidate cartesian_safe_003 \
  --band off --startup-support none \
  --startup-body freebase-wbc --warmup-frames 0 \
  --run-name "replay_$(date +%Y%m%d_%H%M%S)" \
  --no-images
```

每次回放写入独立的 `validation/<run-name>/`，已有同名目录会被拒绝覆盖。`--no-images` 关闭实时渲染，仍完整记录物理状态和接触判定；需要视频时可在回放后从记录的实际状态渲染。

runner 默认 episode 为 `freebase_wbc_002`，默认 candidate 读取该 episode 的 `metadata.json.selected_candidate`。上面的命令显式写出已验证参数，方便复现。

新的成功回放会更新所选 rollout。先按下节命令为该 run 保存 RGB；再刷新数据集 CSV 并检查一致性（schema v3 导出要求图像对齐检查通过）：

```bash
python scripts/export_expert_episode.py \
  --episode outputs/expert_episodes/freebase_wbc_002
python scripts/verify_expert_episode.py \
  --episode outputs/expert_episodes/freebase_wbc_002
```

验证器检查 joint mapping、帧覆盖、身体/手部同步、物理成功指标及 reference/actual 文件一致性。`expert_valid=false` 时返回 exit code 2；文件可加载本身不代表抓取成功。

## 查看视频与数据

已保存的成功视频在：

```text
outputs/expert_episodes/freebase_wbc_002/
└── validation/sonic_safe_confirm_001/vision/
    ├── task_overview.mp4    # 第三人称全景
    ├── head_rgb.mp4         # 头部相机
    ├── wrist_rgb.mp4        # 手腕相机
    └── images.npz          # RGB、reference 帧索引和时间戳
```

三路视频约 13 秒、10 FPS，由成功 rollout 的实际 qpos/qvel 渲染。对新的回放，可用以下命令生成图像；将 `YOUR_RUN_NAME` 替换为实际 run-name，已有 vision 目录不会覆盖：

```bash
python scripts/render_expert_rollout.py \
  --episode outputs/expert_episodes/freebase_wbc_002 \
  --rollout validation/YOUR_RUN_NAME
```

episode 根目录包含统一时间轴上的 reference 和所选成功回放的 actual 数据：

| 数据 | 当前 shape |
|---|---|
| body_ref_q、body_ref_dq、body_q、body_dq | `[650,29]` |
| hand_ref_q、hand_ref_dq、hand_q、hand_dq | `[650,14]` |
| timestamps | `[650]` |
| root_pos、root_lin_vel、root_ang_vel、cube_pos | `[650,3]` |
| root_quat、cube_quat | `[650,4]`，wxyz |
| 每路 RGB | `[130,240,320,3]` |

joint names 明确记录在 metadata.json 和 joint_names.json。`sonic_reference/` 包含 SONIC 所需的六个 CSV，以及 metadata.txt、info.txt；这些 CSV 与通过物理验证的 candidate 一致。原始源 rollout 和校正历史保存在 episode 内。

## 生成新的源轨迹

选择一个尚不存在的目录，生成默认 16 秒、neutral 起点的中心方块源轨迹：

```bash
python scripts/generate_freebase_expert.py \
  --episode outputs/expert_episodes/freebase_new
```

这一步只生成待验证 candidate。后续使用 `refine_freebase_reference.py` 进行需要的 reference 校正，并通过 `run_sonic_grasp.py` 验证；源抓取成功不会自动获得最终 expert 认证。现有成功 episode 的校正过程见 [技术说明](docs/freebase_expert.md)。


## 标准化数据集与 neutral 起点

新 episode 使用 schema v3，统一英文 instruction、相机角色和从 MuJoCo 初始化状态读取的
`task_config`。标准初始姿态由 [neutral_standing.json](configs/neutral_standing.json) 定义：
左右臂对称下垂、肘部轻微弯曲、双手张开，作为左右手或双手任务的共同起点。
当前右手执行抓取，左臂保持 neutral，完整保留 body29 和 hand14 的 reference/actual。

默认新轨迹为 16 秒，状态/reference 为 50 Hz、RGB 为 10 Hz。旧 `freebase_wbc_002`
保留原轨迹，仅补充元数据，并标明其旧初始姿态。新 neutral episode 还会拒绝自接触或
左臂环境接触。字段定义、同步关系、失败记录和 pilot 命令见
[数据集规范](docs/dataset_schema.md)。本次五点全部验证成功，四点直接成功，x+2 cm 经四次校正成功；详细指标见
[五点 pilot 结果](docs/pilot_xy_results.md)。pilot 结果位于
`outputs/expert_episodes/pilot_xy_summary.json`，通过物理与图像对齐验证的清单位于
`pilot_xy_training_manifest.json`。失败及开发记录不进入训练清单。

## 连续 XY 批量生产与 Dataset loader

[批量流程说明](docs/batch_dataset_pipeline.md) 提供可续跑的连续位置采样、有限次数物理回放/校正、
自动筛选和训练 manifest。`produce_experts.py` 的 `--num-episodes` 指候选位置数，失败会保留，
不会无限补采直到全部成功。schema v3 保持不变。

`vision_action_dataset.py` 按保存的 RGB reference frame 索引读取当前 body29/hand14 状态，
返回未来 `[H,29]` / `[H,14]` reference 动作块。支持尾部 mask 或丢弃不完整块，按完整 episode
划分 train/validation；cube GT 和 overview 图像不进入默认模型输入。

## 项目结构

```text
dl-group2/
├── configs/grasp_cases.json        # 方块位置配置
├── scenes/tabletop.xml            # 地面、桌子、红色方块和相机
├── scripts/                       # 生成、控制、数据规范、pilot 与验证模块
├── docs/
│   ├── freebase_expert.md          # 方法、joint order、验证与复现细节
│   ├── code_layout.md             # 各脚本职责和官方依赖说明
│   ├── cleanup_validation.json    # 清理后的物理回放验证结果
│   └── project_overview.md        # 课程原始目标与规划
├── outputs/                       # 本地生成数据与视频，不进入 Git
│   ├── expert_episodes/freebase_wbc_002/
│   └── recordings/
└── third_party/GR00T-WholeBodyControl
```

旧固定骨盆轨迹和无关实验入口已清理，必要的 IK 与导出函数分别保留在 `grasp_reference.py`、`reference_io.py`。完整清理范围见 [代码结构说明](docs/code_layout.md) 和 [清理记录](docs/code_cleanup.json)。

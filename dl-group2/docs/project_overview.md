# dl-group2
DASC7606C Track 4 Unitree G1 文本驱动桌面抓取项目

**项目目标**
本项目研究如何使用自然语言指令控制 Unitree G1 人形机器人，在仿真环境中完成桌面方块的伸手、抓取和抬起。
核心系统需要比较两种不同的运动生成方法——ARDY 与 Kimodo。两种方法必须使用：
- 相同的 Unitree G1 仿真模型；
- 相同的桌面、方块、相机和初始条件；
- 相同的 SONIC 低层控制器；
- 相同的手部或夹爪控制逻辑；
- 相同的指令集、成功标准和实验次数；
- 相同的日志、评估指标和失败分类。
项目最终目标不是只展示一次成功动画，而是形成一套可以复现、可以公平比较、能够解释成功和失败原因的完整流程：
自然语言指令
→ ARDY 或 Kimodo
→ 方法专属输出适配
→ 统一 G1 运动表示
→ SONIC 低层控制
→ 手部或夹爪控制
→ 仿真抓取并抬起方块
→ 日志、指标和失败分析
注意：课程的 Track 4 指整个课题；下文的 T4 指 GitHub Issue 中的 “ARDY Deployment”，二者不是同一个层级。

**课程硬性要求**
核心交付范围包括：
1. 参加 Archon 监督体验，理解并说明 demonstration recording 和 policy inference 的输入、记录数据、模型输出及机器人控制链路。
2. 在仿真中使用 Unitree G1 和 SONIC。
3. 接收自然语言文字指令，使 G1 抓取并抬起桌面方块。
4. 实现并比较至少两种真正不同的运动生成方法。只修改同一方法的 Prompt 不算第二种方法。
5. 说明运动表示、G1 本体适配、SONIC 接口、手部控制，以及只控制上半身时的下半身稳定方式。
6. 在统一条件下进行重复实验，报告成功率、响应时间、用户纠正次数和代表性失败。
7. 提供两种方法的成功和失败视频。
8. 建立公开 GitHub 仓库，包含代码、配置、安装说明、评估代码和复现步骤。
9. 提交 5–8 页报告，不含参考文献。
10. 进行 10 分钟展示，其中必须包含 Demo，随后进行 5 分钟问答。
11. 提交成员贡献说明、LLM 使用声明，并注明外部代码、模型、数据集和资产来源。
真机部署、Sim-to-Real、模型训练、语音输入、VLM/VLA、LeRobot 数据集和多物体泛化均属于可选扩展，不得阻塞核心 ARDY/Kimodo 仿真比较。任何真机部署都必须先完成仿真证据、安全检查并获得 TA 明确批准。

当前可以并行进行的工作
**第一阶段 现在即可并行**
- **T1 仿真环境：**完成物理、碰撞、相机、控制循环、初始站姿和确定性 reset。
- **T2 场景设计：**先定义物体、目标区域、场景编号、标准位置、成功条件和物理属性；最终可重复运行验收需要与 T1 汇合。
- **T3 ARDY Baseline：**独立复现官方示例，确认输入、输出、Checkpoint 和运行方式。
- **T6 Kimodo Baseline：**独立复现官方示例并分析输出表示。
- **T9 前置工作：**提前定义日志字段、实验矩阵、失败分类、成功检测和统计方法。
- **T10 前置工作：**持续整理 Archon 流程、系统架构、方法说明、LLM 使用记录和 README 骨架，但暂不写实验结论。
**第二阶段 两条方法链并行**
满足各自前置条件后，可同时进行：
- T4 ARDY Deployment → T5 ARDY Debug
- T7 Kimodo Deployment → T8 Kimodo Debug
两条链必须共用同一套环境、接口、SONIC 配置和评估标准，不能分别搭建互不兼容的独立系统。
**第三阶段 汇合**
T5 和 T8 都达到稳定验收条件后：
- 冻结代码、配置、Checkpoint、指令、场景、Seeds 和实验次数；
- 执行 T9 正式统一实验；
- 汇总日志、图表、成功和失败视频；
- 完成 T10 最终报告、展示、Demo 和 README。

<img width="892" height="284" alt="image" src="https://github.com/user-attachments/assets/aeae87a9-5fd2-4dd5-ac7c-06bad3d74ce0" />

**必须尽早冻结的公共接口**
建议所有方法统一输出：
CanonicalMotion
  joint_names
  q_ref[T, 29]
  qdot_ref[T, 29] 或统一速度计算规则
  timestamps[T]
  root_position[T]
  root_orientation[T]
  optional_hand_command[T]
  metadata:
    method
    instruction
    seed
    scene_id
    object_pose
    checkpoint
    config_version
还必须冻结：
- 坐标系定义和变换方向；
- 角度、距离和时间单位；
- G1 关节名称与顺序；
- 仿真频率、SONIC 控制频率和轨迹采样率；
- 缺失关节、裁剪和插值规则；
- standing/root reference；
- hand/gripper 命令格式和时序；
- 场景 ID、对象 ID 和目标区域；
- reset、随机种子和成功检测；
- 配置及 Checkpoint 版本。

**设备与资源**
核心设备和软件包括：
- Unitree G1 人形机器人；
- SONIC 低层控制器；
- MuJoCo 仿真；
- 当前仓库中的 G1 29-DoF 加 hand 模型；
- Archon 监督下的真机、遥操作、动捕和策略推理设施；
- 机器人第一视角和外部相机；
- HKU CS RTX 4080 GPU Farm；
- Moore Threads GPU。若实际集成并在最终报告提供 Profiling 和 Benchmark，可获得课程说明中的 5 分 Bonus。
培训材料提到的真机配置包含约 29 个机身自由度和五指 6-DoF 手，但仿真手型、SONIC 官方手型和 Archon 实际硬件未必一致。开始手部控制前必须确认实际自由度、关节映射和命令接口。
核心任务不要求自行训练模型或采集训练数据。Bonus 数据若开展，应至少包含同步的图像、机器人状态、动作、关节参考、手部命令、时间戳、语言指令和 Episode Boundary，并进行质量检查和互斥的训练/测试位置划分。

**实验和日志规范**
每次运行至少保存：
run_id
method
commit_sha
config_version
checkpoint
instruction
scene_id
seed
robot_initial_pose
object_initial_pose
motion_generation_time
execution_time
user_correction_count
grasp_success
lift_success
collision
joint_limit_violation
tracking_error
grasp_pose_error
failure_stage
failure_reason
video_or_log_link
失败阶段建议统一为：
- Instruction/Parsing
- Motion Generation
- Output Adaptation
- SONIC Tracking
- Hand/Gripper
- Collision/Contact
- Stability
- Timeout
- Success Check
正式实验开始前必须冻结成功标准，例如：
- 抓到的是指定方块；
- 方块离开桌面；
- 抬升高度达到统一阈值；
- 保持时间达到统一阈值；
- 未发生禁止碰撞、跌倒或越界；
- 在统一超时时间内完成。
T9 Issue 当前以每个位置、每种方法 10 次作为工作示例；最终 Proposal 建议 Pilot 后每个报告条件目标不少于 20 次。正式实验前必须统一这里的“条件”和 Trial 数定义，且两种方法使用相同次数。

**Bonus Gate**
Bonus 只在下列核心 Gate 完成后进入正式集成：
1. T1/T2 共享仿真和 reset 稳定；
2. SONIC 可以执行慢速 Scripted Motion；
3. ARDY 与 Kimodo 均完成至少一次端到端抓取抬起；
4. 公共接口、成功标准和日志格式冻结；
5. 核心提交进度不存在明显风险。
最终英文 Proposal 描述了 Speech-to-Text、可训练视觉语言模型以及分阶段 Sim-to-Real/ACT、条件式 GR00T 和可选 Qwen 规划；当前 GitHub 看板只列出 T11 VLM 和 T12 Generation Network。开始 Bonus 前需要先统一最终范围，并为每个 Bonus 补齐输入、输出、依赖、实验矩阵和验收标准。****

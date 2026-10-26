# Track 4：统一场景与控制接口

本文档用于统一 **Expert / VA / Kimodo / ARDY** 在 G1 桌面抓取任务中的场景、初始状态、轨迹接口、下游控制器和评测标准。

核心原则：

> 不同方法只负责产生上层运动 reference。  
> 机器人模型、场景、初始状态、SONIC、Hand PD 和成功判据保持一致。

---

## 1. 当前已验证的基线

当前 MThreads/MUSA 环境已经完成：

```text
SONIC MUSA integration: PASS
Physical replay: PASS
Grasp validation: PASS

800 / 800 frames
no fall
final cube lift ≈ 0.149 m
hold ≈ 4.375 s
```

因此后续 Kimodo、ARDY、VA 都应该尽量复用这一套已经跑通的 downstream stack，而不是各自重新实现低层控制。

---

# 2. 总体控制链路

统一架构：

```text
Expert / VA / Kimodo / ARDY
              ↓
        motion reference
              ↓
       ┌───────────────┐
       │               │
 body reference     hand reference
       ↓               ↓
     SONIC           Hand PD
       │               │
       └───────┬───────┘
               ↓
            MuJoCo
               ↓
         grasp evaluation
```

需要变化的是：

```text
Expert
VA
Kimodo
ARDY
```

不应随方法改变的是：

```text
G1 robot
tabletop scene
initial robot pose
cube properties
SONIC
Hand PD
physics
success criteria
```

---

# 3. Ground Truth 文件位置

不要复制一套新的机器人或场景参数到各自代码中。

应直接参考当前项目已有文件。

## 机器人模型

```text
GR00T-WholeBodyControl/
decoupled_wbc/control/robot_model/model_data/g1/
g1_29dof_with_hand_rev_1_0_activatedfinger.xml
```

当前 tabletop 模型编译后的维度：

```text
nq = 57
nv = 55
nu = 43
```

其中：

```text
body joints = 29 DoF
hands       = 14 DoF
```

---

## MuJoCo 场景

```text
dl-group2/scenes/tabletop.xml
```

场景加载逻辑：

```text
dl-group2/scripts/scene.py
```

不要为 Kimodo 或 ARDY 单独新建另一份 tabletop scene。

---

## 机器人初始姿态

```text
dl-group2/configs/neutral_standing.json
```

当前统一初始状态：

```text
neutral_standing_v1
```

主要关节角：

```text
knee          =  0.30 rad
hip_pitch     = -0.15 rad
ankle_pitch   = -0.15 rad

shoulder_pitch = 0.70 rad
shoulder_roll  = ±0.45 rad
shoulder_yaw   = 0

elbow          = 0.40 rad

wrist_roll     = 0
wrist_pitch    = 0
wrist_yaw      = 0

hands          = open
```

因此不同方法不要自行设计不同的“起手姿势”。

---

# 4. Tabletop 场景参数

当前场景中的主要固定参数：

| 参数 | 当前值 |
|---|---|
| Table center XY | `[0.65, 0]` |
| Table surface height | `0.75 m` |
| Cube size | `0.07 × 0.07 × 0.07 m` |
| Cube mass | `0.1 kg` |
| Cube friction | `[0.95, 0.01, 0.001]` |
| Default cube center | `[0.40, -0.15, 0.785]` |
| MuJoCo integrator | `implicitfast` |

这些参数的 ground truth 是：

```text
scenes/tabletop.xml
scripts/scene.py
```

不要在 Kimodo / ARDY 代码里重新 hard-code 另一套。

---

# 5. Cube 位置

已经成功生成并验证的数据中，主要随机区域为：

```text
x ∈ [0.38, 0.40]
y ∈ [-0.17, -0.13]
```

注意：

**这个范围不是 `tabletop.xml` 的固定场景参数。**

它来自 expert production / case sampling 设置。

因此：

- 桌子、cube 大小、质量等属于固定 scene；
- cube XY 属于 episode / evaluation case。

如果不同方法进行公平比较，应使用**同一组 cube initial positions**。

最简单的方式是所有方法直接使用相同的测试 case / seed / episode 列表。

---

# 6. 上层方法应该输出什么

Kimodo、ARDY、VA 的角色不是直接输出 torque。

它们负责产生：

```text
motion reference
```

再由已有控制器执行。

---

## 6.1 Body

SONIC 负责：

```text
29 DoF body
```

现有 SONIC motion reference 不只是一个 torque vector。

当前 reference 文件包含：

```text
joint_pos.csv
joint_vel.csv

body_pos.csv
body_quat.csv
body_lin_vel.csv
body_ang_vel.csv
```

也就是说，SONIC 使用的是一个完整的 body motion reference。

其中 joint reference 是：

```text
body q  : [T, 29]
body dq : [T, 29]
```

同时还存在 root/body：

```text
position
orientation quaternion
linear velocity
angular velocity
```

因此 Kimodo / ARDY 接入时：

> 应优先适配当前 SONIC 已有的 motion-reference interface，而不是直接给 SONIC 一个随意定义的 29 维向量。

如果模型本身只预测 joint trajectory，则需要明确设计 adapter，将它转换成当前 SONIC 所需要的 reference 表示。

---

## 6.2 Hand

手部不经过 SONIC。

当前链路：

```text
hand reference
    ↓
HandTrajectoryController
    ↓
Hand PD
    ↓
MuJoCo
```

Hand reference：

```text
hand_ref_q: [T, 14]
```

单位：

```text
radian
```

Hand PD：

```text
tau = 12 * (q_ref - q) - 0.3 * dq
```

即：

```text
kp = 12
kd = 0.3
```

Kimodo / ARDY 不应该为这次 benchmark 自己重新设计另一套手部低层控制器。

---

# 7. 当前 VA 接口

目前 VA 使用：

```text
head_rgb
body_q
hand_q
instruction
```

预测：

```text
body_ref_q
hand_ref_q
```

时间语义已经统一为：

```text
observation at t
        ↓
reference target starting at t
```

即：

```text
target = reference[t : t + H]
```

不是：

```text
reference[t + 1 : t + 1 + H]
```

这是后续 Kimodo / ARDY 如果复用 dataset 时需要特别注意的地方。

---

# 8. Camera 定义

当前项目有三路 camera：

```text
head_rgb
wrist_rgb
task_overview
```

角色：

```text
head_rgb
→ 当前默认 policy observation

wrist_rgb
→ optional observation

task_overview
→ evaluation / visualization only
```

当前 VA baseline 只需要：

```text
head_rgb
```

不要把 `task_overview` 当成 policy 输入。

如果 Kimodo / ARDY 需要其他视觉输入，应单独说明；不要因为 visualization 有三路相机，就默认模型也必须使用三路图像。

---

# 9. Joint Order

这里非常重要：

```text
dataset joint order
MuJoCo XML joint order
actuator order
SONIC / motor order
```

并不完全相同。

不要直接认为：

```text
array index == actuator index
```

现有代码使用 name-based mapping / `JointMap`。

---

## Dataset body order

29 DoF：

```text
left_hip_pitch_joint
right_hip_pitch_joint
waist_yaw_joint
left_hip_roll_joint
right_hip_roll_joint
waist_roll_joint
left_hip_yaw_joint
right_hip_yaw_joint
waist_pitch_joint
left_knee_joint
right_knee_joint
left_shoulder_pitch_joint
right_shoulder_pitch_joint
left_ankle_pitch_joint
right_ankle_pitch_joint
left_shoulder_roll_joint
right_shoulder_roll_joint
left_ankle_roll_joint
right_ankle_roll_joint
left_shoulder_yaw_joint
right_shoulder_yaw_joint
left_elbow_joint
right_elbow_joint
left_wrist_roll_joint
right_wrist_roll_joint
left_wrist_pitch_joint
right_wrist_pitch_joint
left_wrist_yaw_joint
right_wrist_yaw_joint
```

---

## Dataset hand order

14 DoF：

```text
left_hand_index_0_joint
left_hand_index_1_joint
left_hand_middle_0_joint
left_hand_middle_1_joint
left_hand_thumb_0_joint
left_hand_thumb_1_joint
left_hand_thumb_2_joint

right_hand_index_0_joint
right_hand_index_1_joint
right_hand_middle_0_joint
right_hand_middle_1_joint
right_hand_thumb_0_joint
right_hand_thumb_1_joint
right_hand_thumb_2_joint
```

不要自己重排，除非 adapter 明确按 joint name 转换。

---

# 10. Timing

当前经过验证的 baseline：

```text
SONIC control/reference : 50 Hz
replay physics          : 200 Hz
hand PD                 : 200 Hz
RGB dataset             : 10 Hz
```

Source expert generation 使用更高的 physics rate：

```text
500 Hz
```

但这不是 SONIC replay 的 physics rate。

Kimodo / ARDY 应该输出可转换到当前 reference timing 的 trajectory，不要自行改变下游 SONIC controller frequency。

---

# 11. SONIC 的作用

SONIC 不是 trajectory generator。

SONIC 的作用是：

```text
motion reference
+
current robot state/history
        ↓
SONIC encoder
        ↓
SONIC decoder
        ↓
29 body actions
        ↓
joint targets
        ↓
MuJoCo / robot
```

当前成功控制链：

```text
reference
→ SONIC encoder
→ SONIC decoder
→ 29 body actions
→ joint permutation
→ default joint angles + scaled action
→ body q_target
→ PD / MuJoCo
```

因此：

```text
Kimodo / ARDY
```

应该被理解为 SONIC 的**上游 motion generator**，不是 SONIC 的替代品。

---

# 12. Hand 和 SONIC 的关系

SONIC：

```text
只负责 body 29 DoF
```

Hand：

```text
独立 14 DoF Hand PD
```

统一结构：

```text
              trajectory generator
                      ↓
              motion reference
               /            \
              /              \
       body reference     hand reference
             ↓                 ↓
           SONIC            Hand PD
              \              /
               \            /
                  MuJoCo
```

---

# 13. 成功判据

不同方法必须使用同一套 evaluator。

核心成功条件包括：

```text
cube lift >= 0.10 m

右手形成有效 opposing contact

contact force > 1e-4 N

final hold 阶段 cube 不再接触 table

continuous successful hold >= 2 s

robot does not fall

replay complete and finite
```

同时禁止通过以下方式“完成”抓取：

```text
teleport cube
attach cube
weld cube to hand
virtual support
直接修改 object qpos
```

必须是真实 MuJoCo contact dynamics 下的抓取。

---

# 14. 公平比较原则

比较：

```text
Expert
VA
Kimodo
ARDY
```

时，下列内容必须保持一致：

```text
G1 model
initial pose
table
cube geometry
cube mass/friction
cube test positions
SONIC
Hand PD
physics settings
success evaluator
```

唯一应该变化的是：

```text
谁来生成 motion reference
```

因此最终比较结构应该是：

```text
                  SAME SCENE
                      │
         ┌────────────┼────────────┐
         │            │            │
        VA         Kimodo        ARDY
         │            │            │
         └──── reference trajectory ┘
                      ↓
               SAME SONIC
                  +
               SAME Hand PD
                      ↓
               SAME MuJoCo
                      ↓
               SAME evaluator
```

---

# 15. 对 Kimodo / ARDY 同学的最低接入要求

接入前请确认：

- [ ] 使用同一个 G1 XML
- [ ] 使用 `neutral_standing_v1`
- [ ] 使用同一个 tabletop scene
- [ ] 使用相同 cube test positions
- [ ] 不修改 cube / table 参数
- [ ] 明确模型输出 reference 的 joint order
- [ ] body reference 能转换到 SONIC 当前 reference interface
- [ ] hand reference 为 14 DoF
- [ ] body 仍由 SONIC 执行
- [ ] hand 仍由 Hand PD 执行
- [ ] 使用同一套 success evaluator
- [ ] 不使用 object attachment / teleport

---

# 16. Source of Truth

如果本文档与代码存在冲突，优先级如下：

```text
1. 当前已经通过 MUSA + SONIC physical replay 的实际代码和配置
2. 当前仓库中的 XML / JSON / Python / SONIC config
3. 本 README
```

因此本文档主要是接口说明，不应该成为另一份重复的 hard-coded configuration。

遇到参数不确定时，应先检查：

```text
dl-group2/configs/
dl-group2/scenes/
dl-group2/scripts/

GR00T-WholeBodyControl/
decoupled_wbc/
gear_sonic/
gear_sonic_deploy/
```

不要根据 README 重新复制一套独立场景。
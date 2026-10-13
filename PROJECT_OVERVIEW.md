# 项目总览（Day 1-37）

## 一、项目目标

**在 Mac M5 上开发宇树机器人的舞蹈编排与群控系统**：
- Go2（四足）单机 + 群控编舞
- G1（人形）单机 + 群控编舞
- 音乐驱动
- 真机部署路径

## 二、技术栈

| 技术 | 用途 |
|------|------|
| MuJoCo | 物理仿真 |
| unitree_sdk2py | 宇树 SDK |
| cyclonedds | DDS 通信 |
| librosa | 音乐分析 |
| pygame | 音频播放 |
| PyTorch | 深度学习（MPS）|
| NumPy | 数值计算 |

## 三、37 天进程

### 阶段 1：环境 + Go2 基础（Day 1-7）
- Day 1: 环境搭建（Mac M5 + Python 3.10 + MuJoCo）
- Day 2: SDK + DDS 通信
- Day 3: MuJoCo 加载 Go2 / G1
- Day 4: PD 控制 + 站立
- Day 5: 周期动作（点头 / 走路）
- Day 6: 动作序列 + 时间轴
- Day 7: JSON 舞蹈定义

### 阶段 2：Go2 编舞 + 群控（Day 8-17）
- Day 8: 音乐同步（librosa + pygame）
- Day 9: 舞蹈库系统
- Day 10-11: 双机群控（同步 / 错时 / 镜像）
- Day 12: 4 台群控 + 队形
- Day 13: 走路（边界）
- Day 14: 毕业作品（30 秒群控舞蹈）
- Day 15-17: 音乐群控 + 10 台

### 阶段 3：Go2 进阶（Day 18-24）
- Day 18-19: DDS 控制仿真
- Day 20-24: 编舞优化 + 物理驱动

### 阶段 4：G1 人形（Day 25-30）
- Day 25: G1 站立
- Day 26: G1 挥手
- Day 27: G1 打招呼序列
- Day 28: G1 编舞系统
- Day 29: G1 多段舞蹈 + 音乐
- Day 30: G1 毕业作品

### 阶段 5：G1 群控（Day 31-37）
- Day 31: 2 台 G1
- Day 32: 2 台 4 种模式
- Day 33: 4 台 G1
- Day 34: 4 台群控编舞
- Day 35: 6 台阅兵
- Day 36: 10 台阅兵
- Day 37: 10 台 + 音乐

## 四、核心代码

### 工具
- `src/robot_utils.py` —— Go2 工具（ctrl↔qpos 映射）
- `src/actions.py` —— Go2 动作库
- `src/actions_v2.py` —— Go2 优化版
- `src/actions_v3.py` —— Go2 全身动作
- `src/g1_actions.py` —— G1 动作库
- `src/make_dual.py` / `make_quad.py` / `make_multi.py` —— 多机 XML 生成

### 舞蹈
- `dances/*.json` —— Go2 舞蹈
- `dances/g1_*.json` —— G1 舞蹈

### 示例
- `examples/day1~37_*.py` —— 每天脚本

## 五、关键经验

### 1. ctrl ↔ qpos 映射
Go2 / G1 的"电机顺序"和"关节顺序"不一致，必须用 `model.jnt_qposadr` 自动生成映射。

### 2. macOS 上 mjpython
带窗口的 MuJoCo 必须在 `mjpython` 下运行；**且要用 venv 的 mjpython**。

### 3. viewer.sync()
`launch_passive` 模式下每步必须 `viewer.sync()`。

### 4. 23dof vs 29dof
G1 23dof 腰只有 yaw；29dof 腰有 yaw + roll + pitch（能鞠躬）。

### 5. 多机 XML
用"字符串替换"（`name="xxx"` → `name="xxx_B"`），注意 `meshdir`。

### 6. 群控模式
- 同步：同一 target
- 错时：B 用 `t-0.5`
- 镜像：不同 action
- 对角：位置 + 动作

## 六、性能

| 台数 | nq | nu | Mac M5 |
|------|-----|-----|--------|
| 1 台 Go2 | 19 | 12 | 流畅 |
| 4 台 Go2 | 76 | 48 | 流畅 |
| 10 台 Go2 | 190 | 120 | 流畅 |
| 1 台 G1 | 36 | 29 | 流畅 |
| 4 台 G1 | 144 | 116 | 流畅 |
| **10 台 G1** | **360** | **290** | **流畅** |

## 七、下一步方向

- [ ] 真机部署（Go2 / G1 实机）
- [ ] RL 强化学习（走路 / 舞蹈）
- [ ] G1 + Go2 混合群控
- [ ] 100 台大规模群控（真机）
- [ ] AI 编舞（LLM 生成舞蹈）

## 八、结语

**37 天，从"零基础"到"两栖机器人编舞工程师"。**

**"Go2 + G1"** = **"两栖群控"**。

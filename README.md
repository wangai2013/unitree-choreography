# Unitree Robot Choreography

在 Mac M5 上开发宇树机器人的舞蹈编排与群控系统

从零基础到 4 台机器人群控编舞，历时 14 天。

## 核心能力

- 单机舞蹈（Go2 四足机器人）
- 人形机器人（G1 / H1）
- 音乐同步（librosa + pygame）
- 群控编舞（最多 4 台 Go2）
- 队形变换（方阵 / 菱形 / 一字 / 圆形）
- 实时仿真（MuJoCo）

## 环境要求

- Mac M5（Apple Silicon）
- Python 3.10
- macOS 14+

## 安装

1. 克隆仓库：
   git clone https://github.com/你的真实用户名/unitree-choreography.git
   cd unitree-choreography

2. 创建虚拟环境：
   python3.10 -m venv venv
   source venv/bin/activate

3. 安装依赖：
   pip install -r requirements.txt

4. 安装宇树 SDK：
   git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
   cd unitree_sdk2_python
   pip install -e .
   cd ..

5. 克隆宇树 MuJoCo 模型：
   git clone https://github.com/unitreerobotics/unitree_mujoco.git

## 运行示例

环境检查：
   python examples/day3_load.py

单台 Go2 站立：
   mjpython examples/day4_stand.py

单台 Go2 音乐舞蹈：
   mjpython examples/day8_music_dance.py

双机独立控制：
   python src/make_dual.py
   mjpython examples/day10_dual_control.py

4 台群控舞蹈（毕业作品）：
   python src/make_quad.py
   mjpython examples/day14_group_choreo.py

## 项目结构

- README.md                  项目说明
- LEARNING_NOTES.md          学习笔记
- requirements.txt           依赖
- src/                       核心代码
  - robot_utils.py           通用工具
  - actions.py               动作库
  - make_dual.py             双机 XML 生成
  - make_quad.py             4 台 XML 生成
- dances/                    舞蹈定义（JSON）
- examples/                  示例脚本（Day 1~14）
- examples/archive/          历史版本

## 学习路径（14 天）

Day 1  环境搭建
Day 2  SDK + DDS
Day 3  模型加载
Day 4  PD 控制
Day 5  周期动作
Day 6  时间轴
Day 7  JSON 舞蹈
Day 8  音乐同步
Day 9  舞蹈库
Day 10 双机群控
Day 11 同步/错时/镜像
Day 12 4 台群控
Day 13 走路（边界）
Day 14 群控舞蹈（毕业作品）

## 关键技术点

1. ctrl 与 qpos 映射
   Go2 的电机顺序和关节顺序不一致，必须用 model.jnt_qposadr 自动生成。

2. macOS 的 MuJoCo GUI
   带窗口的程序必须用 mjpython 运行。

3. viewer.sync() 不能忘
   launch_passive 模式下，每步必须 viewer.sync() 刷新画面。

4. 多机 XML 生成
   用字符串替换法复制 base_link + actuator。

## 参考

- Unitree SDK2 Python: https://github.com/unitreerobotics/unitree_sdk2_python
- Unitree MuJoCo: https://github.com/unitreerobotics/unitree_mujoco
- MuJoCo 文档: https://mujoco.readthedocs.io/
- librosa: https://librosa.org/

## License

MIT


---

## 真机部署

从 "MuJoCo 仿真" 到 "真实宇树 Go2" —— 详见 docs/real_robot_deploy.md

核心思路：
- DDS 协议不变（rt/lowcmd / rt/lowstate）
- 改 3 行代码（网络接口 + 删 MuJoCo）
- 直接可用（同一套 SDK）

仿真：
    ChannelFactoryInitialize(0, "lo0")

真机：
    ChannelFactoryInitialize(0, "en0")

---

## 学习成果

- 19 天从零到"群控编舞系统"
- 单机 + 双机 + 4 台 + 10 台群控
- 音乐同步 + JSON 编排
- DDS 通信 + Sim-to-Real 路径
- 一套完整的学习笔记

这是一个"从学习到作品"的完整闭环。


---

## 🧍 G1 人形机器人编舞（Day 25-30）

**从"四足 Go2"到"人形 G1"** —— 完整编舞系统。

### 能力
- ✅ G1 站立（锁机身）
- ✅ 挥手 / 鞠躬 / 转身 / 双手挥
- ✅ 编舞系统（JSON 驱动）
- ✅ 音乐驱动
- ✅ 左右臂镜像

### 文件
- `src/g1_actions.py` —— G1 动作库（8+ 个动作）
- `dances/g1_*.json` —— G1 舞蹈
- `examples/day28_g1_player.py` —— G1 播放器
- `examples/day29_g1_music.py` —— G1 音乐驱动

### 运行示例

打招呼：

    mjpython examples/day28_g1_player.py dances/g1_greet.json

热情：

    mjpython examples/day28_g1_player.py dances/g1_enthusiastic.json

正式：

    mjpython examples/day28_g1_player.py dances/g1_formal.json

音乐：

    mjpython examples/day29_g1_music.py

毕业作品：

    mjpython examples/day28_g1_player.py dances/g1_masterpiece.json

**"两栖机器人编舞"** = **Go2（四足） + G1（人形）**。


---

## 🎬 G1 群控编舞（Day 31-37）

**从"单台 G1"到"10 台阅兵式"** —— 人形机器人群控完整实现。

### 能力
- ✅ 2 台 / 4 台 / 6 台 / **10 台** G1 群控
- ✅ **4 种模式**：同步 / 波浪 / 镜像 / 对角
- ✅ **10 台 + 音乐**（节拍驱动）
- ✅ **阅兵式**（5×2 方阵）

### 文件
- `examples/day31_g1_dual_view.py` —— 2 台 G1
- `examples/day32_g1_dual_modes.py` —— 2 台 4 种模式
- `examples/day33_g1_quad_view.py` —— 4 台 G1
- `examples/day34_g1_quad_modes.py` —— 4 台群控编舞
- `examples/day35_g1_hexa_parade.py` —— 6 台阅兵
- `examples/day36_g1_deca_parade.py` —— 10 台阅兵
- `examples/day37_g1_deca_music.py` —— 10 台 + 音乐

### 运行示例

2 台 G1 群控：

    cd ~/unitree-dev
    source venv/bin/activate
    mjpython examples/day32_g1_dual_modes.py

10 台 G1 阅兵：

    mjpython examples/day36_g1_deca_parade.py

10 台 G1 + 音乐：

    mjpython examples/day37_g1_deca_music.py

---

## 📊 项目总览（37 天）

### 阶段 1：Go2 四足（Day 1-14）
- 环境搭建（Mac M5 + MuJoCo）
- SDK + DDS 通信
- PD 控制 + 站立
- 周期动作（点头 / 摇摆 / 走路）
- 编舞系统（JSON 驱动）
- 音乐同步（librosa + pygame）
- 单机 → 双机 → 4 台 → 10 台群控

### 阶段 2：Go2 进阶（Day 15-24）
- 编舞优化（幅度 / 频率 / 队形）
- 全身动作（腿 + 机身）
- PD 驱动 + 混合约束
- DDS 控制仿真
- 真机部署文档

### 阶段 3：G1 人形（Day 25-30）
- G1 站立（29dof，锁机身）
- 挥手 / 鞠躬 / 转身
- 编舞系统（JSON + 动作库）
- 30 秒毕业作品

### 阶段 4：G1 群控（Day 31-37）
- 2 台 / 4 台 / 6 台 / 10 台
- 同步 / 波浪 / 镜像 / 对角
- 音乐群控

---

## 🛠️ 环境要求

- Mac M5（Apple Silicon）
- Python 3.10（venv 虚拟环境）
- MuJoCo 3.x
- librosa + pygame（音频）
- unitree_sdk2py + cyclonedds

---

## 📝 每次运行前必做

    cd ~/unitree-dev
    source venv/bin/activate      # ⭐ 必须
    which mjpython                 # 确认是 venv/bin/mjpython
    mjpython xxx.py

---

## 🎓 学习成果

- ✅ 从"零基础"到"两栖机器人编舞工程师"
- ✅ Go2 四足 + G1 人形，单机 + 群控
- ✅ 37 天完整代码 + 文档 + GitHub
- ✅ "阅兵级"人形群控（10 台）

**"两栖机器人编舞"** = **Go2（四足） + G1（人形）**。

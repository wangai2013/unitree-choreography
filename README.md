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
   git clone https://github.com/your-username/unitree-choreography.git
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

# Unitree 机器人开发 · 学习笔记

14 天从零到群控编舞的完整记录

## 一、环境信息

- 芯片：Apple M5 (arm64)
- 系统：macOS 26.6.2
- Python：3.10.22（虚拟环境）
- MuJoCo：3.x
- SDK：unitree_sdk2py 1.0.1
- DDS：cyclonedds 0.10.2

## 二、核心知识点

### 1. DDS 通信
- 发布-订阅模式
- rt/lowstate：机器人 -> 我们（读状态）
- rt/lowcmd：我们 -> 机器人（发指令）

### 2. MuJoCo 结构
- model：静态模型
- data：动态状态（qpos / qvel / ctrl）
- mj_step()：推进物理
- mj_forward()：更新派生量

### 3. Go2 的 qpos 结构（nq=19）
- qpos[0:3]：机身位置 (x, y, z)
- qpos[3:7]：机身四元数
- qpos[7:19]：12 个关节（顺序 FL -> FR -> RL -> RR）

### 4. ctrl 与 qpos 映射
- 电机顺序（ctrl）：FR -> FL -> RR -> RL
- 关节顺序（qpos）：FL -> FR -> RL -> RR
- 必须用 model.jnt_qposadr 自动生成

### 5. PD 控制
力矩 = Kp * (目标角度 - 当前角度) - Kd * 当前角速度

### 6. Go2 力矩上限（ctrlrange）
- 髋/大腿：±23.70 N·m
- 小腿：±45.43 N·m

### 7. 所有动作 = 时间的函数
target = 基准角度 + 振幅 * sin(2π * 频率 * t + 相位)

### 8. 动作分类
- 关节摆动：关节转动（挥手）
- 重心转移：腿伸缩（摇摆）
- 位移：腿蹬地（走路）

### 9. viewer.sync() 必须
launch_passive() 模式，每步都要刷新画面。

## 三、踩坑清单

1. 画面不动，数据在变 -> 缺 viewer.sync()
2. 机器人瘫软 -> 初始姿态错 / Kp 小
3. ctrl/qpos 映射错 -> 用 jnt_qposadr
4. 采样陷阱 -> 打印加密
5. 动作产生位移 -> 腿蹬地（正常）
6. mjpython vs python -> macOS GUI 要 mjpython
7. 多机 XML -> include 冲突，用字符串替换

## 四、多机群控

### 双机 XML 生成
- 提取 base_link + actuator
- 复制加 _B 后缀
- joint 重定向

### ctrl 索引
- A：ctrl[0~11]   -> qpos [10 11 12 7 8 9 16 17 18 13 14 15]
- B：ctrl[12~23]  -> qpos [29 30 31 26 27 28 35 36 37 32 33 34]
- C：ctrl[24~35]  -> qpos [48 49 50 45 46 47 54 55 56 51 52 53]
- D：ctrl[36~47]  -> qpos [67 68 69 64 65 66 73 74 75 70 71 72]

### 群控效果
- 同步：所有机器人同动作
- 错时：延迟传播（波浪）
- 镜像：反相（对称）

## 五、快捷命令

激活环境：
   cd ~/unitree-dev && source venv/bin/activate

清理进程：
   pkill -f mjpython

运行（带窗口）：
   mjpython examples/day14_group_choreo.py

生成多机 XML：
   python src/make_dual.py
   python src/make_quad.py

## 六、未来方向

- 接入真实音乐（librosa + pygame 集成到群控）
- JSON 化群控舞蹈
- 10+ 台群控
- G1 人形机器人
- 强化学习步态
- 实机部署

从零到群控编舞，14 天完成。

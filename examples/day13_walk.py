"""
Day 13: 单台 Go2 走直线（简化步态）
原理：对角腿交替摆动 + 给机身一个"前进推力"
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

# 步态参数
STEP_FREQ = 2.0       # 步频 2 Hz
STEP_LIFT = 0.4       # 抬腿幅度
STEP_SWING = 0.3      # 摆腿幅度

print("=" * 60)
print("🎬 Go2 走直线（对角步态）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        phase = 2 * np.pi * STEP_FREQ * t

        target = STAND.copy()

        # ⭐ 对角步态：
        # 组 A：FL(3,4,5) + RR(6,7,8) 用 sin(phase)
        # 组 B：FR(0,1,2) + RL(9,10,11) 用 sin(phase + π)
        phase_A = np.sin(phase)
        phase_B = np.sin(phase + np.pi)

        # FL（ctrl 3,4,5）
        target[4] = 0.8 - STEP_LIFT * max(0, phase_A)  # 大腿抬起
        target[5] = -1.5 + STEP_SWING * phase_A         # 小腿摆动

        # RR（ctrl 6,7,8）
        target[7] = 0.8 - STEP_LIFT * max(0, phase_A)
        target[8] = -1.5 + STEP_SWING * phase_A

        # FR（ctrl 0,1,2）
        target[1] = 0.8 - STEP_LIFT * max(0, phase_B)
        target[2] = -1.5 + STEP_SWING * phase_B

        # RL（ctrl 9,10,11）
        target[10] = 0.8 - STEP_LIFT * max(0, phase_B)
        target[11] = -1.5 + STEP_SWING * phase_B

        # ⭐ 给机身一个"前进推力"（模拟步态产生的位移）
        data.xfrc_applied[1] = [30.0, 0, 0, 0, 0, 0]   # x 方向 30N

        # PD 控制
        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

        if step % 1000 == 0:
            x = data.qpos[0]
            print(f"t={t:.1f}s | 机器人 x={x:.2f} m")

"""
Day 27: G1 打招呼（挥手 + 点头）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_23dof.xml")
data = mujoco.MjData(model)

CTRL_TO_QPOS = np.zeros(model.nu, dtype=int)
CTRL_TO_QVEL = np.zeros(model.nu, dtype=int)
for i in range(model.nu):
    jid = model.actuator_trnid[i][0]
    CTRL_TO_QPOS[i] = model.jnt_qposadr[jid]
    CTRL_TO_QVEL[i] = model.jnt_dofadr[jid]

STAND = np.zeros(model.nu)
STAND[0] = -0.2; STAND[3] = 0.4; STAND[4] = -0.2
STAND[6] = -0.2; STAND[9] = 0.4; STAND[10] = -0.2

RP, RR, RE = 0.4, -0.1, 0.8
STAND[15] = RP; STAND[16] = -RR; STAND[18] = RE
STAND[22] = RP; STAND[23] = RR;  STAND[25] = RE

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

ROLL_UP = 1.2
HEAD_NOD = 0.3

print("=" * 60)
print("🎬 G1 打招呼序列")
print("  0~2s:  站好")
print("  2~4s:  右臂抬起")
print("  4~7s:  挥手（yaw 摆）")
print("  7~9s:  点头（waist_pitch 弯）")
print("  9~11s: 手臂放下")
print("  11~13s: 站好")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        if t < 2.0:
            pass
        elif t < 4.0:
            # 抬右臂
            p = (t - 2.0) / 2.0
            target[22] = 0.4 + (0.0 - 0.4) * p
            target[23] = -0.1 + (ROLL_UP - (-0.1)) * p
            target[25] = 0.8 + (-1.2 - 0.8) * p
            target[26] = -1.57 * p
        elif t < 7.0:
            # 挥手
            target[22] = 0.0
            target[23] = ROLL_UP
            target[25] = -1.2
            target[26] = -1.57
            t_wave = t - 4.0
            target[24] = 0.5 * np.sin(2 * np.pi * 1.5 * t_wave)
        elif t < 9.0:
            # 点头（腰 pitch）
            target[22] = 0.0
            target[23] = ROLL_UP
            target[25] = -1.2
            target[26] = -1.57
            p = (t - 7.0) / 2.0
            target[14] = HEAD_NOD * np.sin(np.pi * p)  # 腰弯一下
        elif t < 11.0:
            # 手臂放下
            p = (t - 9.0) / 2.0
            target[22] = 0.0 + (0.4 - 0.0) * p
            target[23] = ROLL_UP + (-0.1 - ROLL_UP) * p
            target[25] = -1.2 + (0.8 - (-1.2)) * p
            target[26] = -1.57 + (0.0 - (-1.57)) * p
        else:
            pass

        if step % 500 == 0:
            print(f"t={t:5.1f}s | waist_pitch={target[14]:+.2f} | shoulder_roll={target[23]:+.2f} | yaw={target[24]:+.2f}")

        for i in range(model.nu):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)

        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.8
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[0:6] = 0

        mujoco.mj_forward(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

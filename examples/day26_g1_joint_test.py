"""
Day 26: G1 关节测试 —— 逐个测方向
每 2 秒切换一个关节的"目标值"，看手臂怎么动
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

# 全部先"0"（T-pose）
STAND = np.zeros(model.nu)

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

print("=" * 60)
print("🎬 G1 关节测试")
print("  每 2 秒改一个关节")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()

        # 0~2 秒：右肩 pitch = +0.5（测"方向"）
        # 2~4 秒：右肩 pitch = -0.5
        # 4~6 秒：右肩 roll = +0.5
        # 6~8 秒：右肩 roll = -0.5
        # 8~10 秒：右肘 = +0.5
        # 10~12 秒：右肘 = -0.5
        # 12~14 秒：右腕 = +0.5
        # 14~16 秒：右腕 = -0.5

        if t < 2:
            target[22] = 0.5   # R_shoulder_pitch
        elif t < 4:
            target[22] = -0.5
        elif t < 6:
            target[23] = 0.5   # R_shoulder_roll
        elif t < 8:
            target[23] = -0.5
        elif t < 10:
            target[25] = 0.5   # R_elbow
        elif t < 12:
            target[25] = -0.5
        elif t < 14:
            target[26] = 0.5   # R_wrist_roll
        elif t < 16:
            target[26] = -0.5

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

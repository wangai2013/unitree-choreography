"""
Day 26: G1 站立（试姿态）
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

# ⭐ 站立姿态（参考官方，可能不对，可调）
STAND_G1 = np.array([
    # 左腿
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    # 右腿
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    # 腰
     0.0, 0.0, 0.0,
    # 左臂（下垂）
     0.4, 0.3, 0.0, -1.2, 0.0, 0.0, 0.0,
    # 右臂（下垂）
     0.4, -0.3, 0.0, -1.2, 0.0, 0.0, 0.0,
])

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND_G1[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for i in range(model.nu):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (STAND_G1[i] - q) - KD * dq

        mujoco.mj_step(model, data)

        # 锁机身
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

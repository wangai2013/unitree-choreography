"""
Day 25: G1 站立（锁机身版）
- 29 关节 PD 控制
- 锁机身（防倒）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/g1/scene_23dof.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ G1: nq={model.nq}, nu={model.nu}")

# 映射
CTRL_TO_QPOS = np.zeros(model.nu, dtype=int)
CTRL_TO_QVEL = np.zeros(model.nu, dtype=int)
for i in range(model.nu):
    jid = model.actuator_trnid[i][0]
    CTRL_TO_QPOS[i] = model.jnt_qposadr[jid]
    CTRL_TO_QVEL[i] = model.jnt_dofadr[jid]

# G1 站立姿态（腿部弯曲，重心低）
STAND_G1 = np.array([
    # 左腿
    -0.3,   # L_hip_pitch
     0.0,   # L_hip_roll
     0.0,   # L_hip_yaw
     0.6,   # L_knee
    -0.3,   # L_ankle_pitch
     0.0,   # L_ankle_roll
    # 右腿
    -0.3,   # R_hip_pitch
     0.0,   # R_hip_roll
     0.0,   # R_hip_yaw
     0.6,   # R_knee
    -0.3,   # R_ankle_pitch
     0.0,   # R_ankle_roll
    # 腰
     0.0, 0.0, 0.0,
    # 左臂
     0.3, 0.3, 0.0, -0.5, 0.0, 0.0, 0.0,
    # 右臂
     0.3, -0.3, 0.0, -0.5, 0.0, 0.0, 0.0,
])

# 初始化
for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND_G1[i]

# 机身位置
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.8
data.qpos[3] = 1.0  # 四元数 w

mujoco.mj_forward(model, data)

# PD 参数（加大）
KP = 300.0
KD = 10.0

print("=" * 60)
print("🎬 G1 站立（锁机身）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 1. PD 控制
        for i in range(model.nu):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (STAND_G1[i] - q) - KD * dq

        # 2. 物理步进
        mujoco.mj_step(model, data)

        # 3. ⭐ 锁机身（防倒）
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.8
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[0] = 0
        data.qvel[1] = 0
        data.qvel[2] = 0
        data.qvel[3] = 0
        data.qvel[4] = 0
        data.qvel[5] = 0

        mujoco.mj_forward(model, data)

        viewer.sync()
        step += 1
        time.sleep(0.001)

        if step % 1000 == 0:
            print(f"t={step * 0.001:.1f}s | 高度 {data.qpos[2]:.3f}")

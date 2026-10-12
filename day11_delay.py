"""
Day 11-2: 双机错时（波浪效果）
A 先动，B 后 0.5 秒
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_dual_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS_A = np.zeros(12, dtype=int)
CTRL_TO_QVEL_A = np.zeros(12, dtype=int)
CTRL_TO_QPOS_B = np.zeros(12, dtype=int)
CTRL_TO_QVEL_B = np.zeros(12, dtype=int)

for i in range(12):
    jid_A = model.actuator_trnid[i][0]
    CTRL_TO_QPOS_A[i] = model.jnt_qposadr[jid_A]
    CTRL_TO_QVEL_A[i] = model.jnt_dofadr[jid_A]
    jid_B = model.actuator_trnid[i + 12][0]
    CTRL_TO_QPOS_B[i] = model.jnt_qposadr[jid_B]
    CTRL_TO_QVEL_B[i] = model.jnt_dofadr[jid_B]

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS_A[i]] = STAND[i]
    data.qpos[CTRL_TO_QPOS_B[i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0
DELAY = 0.5   # ⭐ B 延迟 0.5 秒

print("=" * 60)
print(f"🎬 双机错时点头（B 延迟 {DELAY}s）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # ⭐ A 用 t，B 用 (t - DELAY)
        t_A = t
        t_B = max(0.0, t - DELAY)

        # A 目标
        target_A = STAND.copy()
        swing_A = 0.4 * np.sin(2 * np.pi * 1.5 * t_A)
        target_A[2] = -1.5 + swing_A
        target_A[5] = -1.5 + swing_A

        # B 目标（延迟）
        target_B = STAND.copy()
        swing_B = 0.4 * np.sin(2 * np.pi * 1.5 * t_B)
        target_B[2] = -1.5 + swing_B
        target_B[5] = -1.5 + swing_B

        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_A[i]]
            dq = data.qvel[CTRL_TO_QVEL_A[i]]
            data.ctrl[i] = KP * (target_A[i] - q) - KD * dq

        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_B[i]]
            dq = data.qvel[CTRL_TO_QVEL_B[i]]
            data.ctrl[i + 12] = KP * (target_B[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

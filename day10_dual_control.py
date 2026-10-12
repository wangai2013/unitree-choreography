"""Day 10: 双机独立控制（A 点头，B 摇摆）"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_dual_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}")

# ⚠️ 关键：需要手动建立"两台机器人"的映射
# A：ctrl[0~11] 对应 base_link 的关节
# B：ctrl[12~23] 对应 base_link_B 的关节

CTRL_TO_QPOS_A = np.zeros(12, dtype=int)
CTRL_TO_QVEL_A = np.zeros(12, dtype=int)
CTRL_TO_QPOS_B = np.zeros(12, dtype=int)
CTRL_TO_QVEL_B = np.zeros(12, dtype=int)

for i in range(12):
    # A 的电机
    jid_A = model.actuator_trnid[i][0]
    CTRL_TO_QPOS_A[i] = model.jnt_qposadr[jid_A]
    CTRL_TO_QVEL_A[i] = model.jnt_dofadr[jid_A]
    
    # B 的电机
    jid_B = model.actuator_trnid[i + 12][0]
    CTRL_TO_QPOS_B[i] = model.jnt_qposadr[jid_B]
    CTRL_TO_QVEL_B[i] = model.jnt_dofadr[jid_B]

print(f"A 的 qpos 索引: {CTRL_TO_QPOS_A}")
print(f"B 的 qpos 索引: {CTRL_TO_QPOS_B}")

STAND = np.array([0.0, 0.8, -1.5] * 4)
KP, KD = 300.0, 15.0

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # A 目标：点头（前腿小腿摆动）
        target_A = STAND.copy()
        swing = 0.4 * np.sin(2 * np.pi * 1.5 * t)
        target_A[2] = -1.5 + swing
        target_A[5] = -1.5 + swing

        # B 目标：摇摆（大腿微动）
        target_B = STAND.copy()
        sway = 0.15 * np.sin(2 * np.pi * 1.0 * t)
        target_B[1] = 0.8 + sway
        target_B[4] = 0.8 - sway

        # A 控制
        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_A[i]]
            dq = data.qvel[CTRL_TO_QVEL_A[i]]
            data.ctrl[i] = KP * (target_A[i] - q) - KD * dq

        # B 控制
        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_B[i]]
            dq = data.qvel[CTRL_TO_QVEL_B[i]]
            data.ctrl[i + 12] = KP * (target_B[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

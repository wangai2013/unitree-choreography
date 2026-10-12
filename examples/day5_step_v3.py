"""
Day 5-3: Go2 原地踏步（对角步态）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

STAND = np.array([0.0, 0.8, -1.5] * 4)

for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 400.0, 20.0

print("=" * 60)
print("🎬 Go2 原地踏步（对角步态，1.5 Hz）")
print("  对角线腿同时抬起：FL+RR ↔ FR+RL")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        phase = 2 * np.pi * 1.5 * t

        # 对角相位
        lift_A = 0.5 * np.sin(phase)          # FL + RR
        lift_B = 0.5 * np.sin(phase + np.pi)  # FR + RL

        # 抬腿：大腿角减小（后收），小腿角增加（折叠）
        # FR (ctrl 0,1,2)
        target[1] = 0.8 - 0.3 * abs(lift_B)
        target[2] = -1.5 + 0.5 * lift_B
        # FL (ctrl 3,4,5)
        target[4] = 0.8 - 0.3 * abs(lift_A)
        target[5] = -1.5 + 0.5 * lift_A
        # RR (ctrl 6,7,8)
        target[7] = 0.8 - 0.3 * abs(lift_A)
        target[8] = -1.5 + 0.5 * lift_A
        # RL (ctrl 9,10,11)
        target[10] = 0.8 - 0.3 * abs(lift_B)
        target[11] = -1.5 + 0.5 * lift_B

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            print(f"t={t:.2f}s | 高度 {data.qpos[2]:.3f}")

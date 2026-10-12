"""
Day 5-4: Go2 左右摇摆（大动作）
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
print("🎬 Go2 左右摇摆（0.8 Hz，振幅 ±0.4）")
print("=" * 60)
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        sway = 0.1 * np.sin(2 * np.pi * 0.8 * t)
        # 所有髋关节同步摆动（ctrl[0], ctrl[3], ctrl[6], ctrl[9]）
        target[2] =-1.5 + sway 
        target[5] =-1.5 - sway 
        target[8] =-1.5 + sway 
        target[11] =-1.5 - sway 

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            print(f"t={t:.2f}s | 摆动 {sway:+.3f} | 高度 {data.qpos[2]:.3f}")

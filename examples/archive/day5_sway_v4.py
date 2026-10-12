"""
Day 5-4 修正: Go2 左右摇摆（不位移版）
关键：左侧腿和右侧腿反相，形成"摇而不走"
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
print("🎬 Go2 左右摇摆（左右反相，不位移）")
print("  左腿（FL, RL）→ +sway")
print("  右腿（FR, RR）→ -sway")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        sway = 0.4 * np.sin(2 * np.pi * 0.8 * t)

        # ⭐ 左右反相！
        # FR (ctrl 0) = -sway，FL (ctrl 3) = +sway
        # RR (ctrl 6) = -sway，RL (ctrl 9) = +sway
        target[0] = -sway   # FR_hip
        target[3] = +sway   # FL_hip
        target[6] = -sway   # RR_hip
        target[9] = +sway   # RL_hip

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 100 == 0:
            # 打印位置（看有没有位移）
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={t:.2f}s | 摆动 {sway:+.3f} | 位置 (x={x:+.3f}, y={y:+.3f}, z={z:.3f})")

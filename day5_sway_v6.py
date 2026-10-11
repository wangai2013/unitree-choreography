"""
Day 5-4 终极版: Go2 原地左右摇晃（不位移）
关键：同时改"大腿 + 小腿"，让腿原地伸缩
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
print("🎬 Go2 原地摇晃（大腿+小腿协同）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        sway = 0.3 * np.sin(2 * np.pi * 0.8 * t)

        # ⭐ 同时改"大腿 + 小腿"，保持腿总长度不变
        # 左侧腿：大腿向上收 + 小腿向下伸（原地伸缩）
        # 右侧腿：反相
        # FR (ctrl 0,1,2)
        target[1] = 0.8 + sway        # 大腿
        target[2] = -1.5 - sway       # 小腿
        # FL (ctrl 3,4,5)
        target[4] = 0.8 - sway        # 大腿
        target[5] = -1.5 + sway       # 小腿
        # RR (ctrl 6,7,8)
        target[7] = 0.8 + sway
        target[8] = -1.5 - sway
        # RL (ctrl 9,10,11)
        target[10] = 0.8 - sway
        target[11] = -1.5 + sway

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 100 == 0:
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={t:.2f}s | 摆动 {sway:+.3f} | 位置 (x={x:+.3f}, y={y:+.3f}, z={z:.3f})")

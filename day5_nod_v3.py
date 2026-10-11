"""
Day 5-2: Go2 点头（强化版）
用 robot_utils 自动映射，不再手填索引
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ⭐ 自动生成映射
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)
print(f"✓ 自动映射: ctrl→qpos = {CTRL_TO_QPOS}")

# 站立姿态（ctrl 顺序：FR, FL, RR, RL）
STAND = np.array([0.0, 0.8, -1.5] * 4)

# 设置初始姿态（用自动映射）
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

print("=" * 60)
print("🎬 Go2 点头中...（1 Hz，前腿小腿摆动 ±0.3）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        # ctrl[2]=FR_calf, ctrl[5]=FL_calf
        swing = 0.3 * np.sin(2 * np.pi * 1.0 * t)
        target[2] = -1.5 + swing
        target[5] = -1.5 + swing

        # ⭐ 通用 PD 控制
        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            print(f"t={t:.2f}s | 摆动 {swing:+.3f} | 高度 {data.qpos[2]:.3f}")

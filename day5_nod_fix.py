"""
Day 5-1 最终修复: Go2 点头
关键改动：
1. 不再设置 qpos[2] = 0.4（让它自然落地）
2. 初始姿态设为"趴着"（qpos 全 0），让 PD 自然推起
3. 加大 KP 到 400
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

BASE = np.array([0.0, 0.8, -1.5] * 4)

KP, KD = 400.0, 20.0   # ⭐ 更大！

# ⭐ 关键：不设 qpos[2] = 0.4，让它从默认位置开始
mujoco.mj_forward(model, data)

print("=" * 60)
print("🎬 Go2 点头（最终修复版）")
print("  机器人会先'从趴到站'，然后开始点头")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = BASE.copy()
        swing = 0.3 * np.sin(2 * np.pi * 0.8 * t)  # 0.8 Hz，0.3 振幅
        target[2] = -1.5 + swing   # FR calf
        target[5] = -1.5 + swing   # FL calf

        for i in range(12):
            q  = data.qpos[7 + i]
            dq = data.qvel[6 + i]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            print(f"t={t:.2f}s | 摆动 {swing:+.3f} | 高度 {data.qpos[2]:.3f} | FR大腿 {data.qpos[7+1]:+.3f}")

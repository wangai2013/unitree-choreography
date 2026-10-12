"""
Day 5-1 最终版: Go2 点头
关键修复：开局直接把关节设成站立姿态
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 站立姿态（每条腿 [髋, 大腿, 小腿]）
STAND = np.array([0.0, 0.8, -1.5] * 4)

# ⭐⭐⭐ 关键：开局直接设置关节角度 = 站姿 ⭐⭐⭐
for i in range(12):
    data.qpos[7 + i] = STAND[i]

# 让机器人从 0.3m 高度开始（站姿高度），避免落地冲击
data.qpos[2] = 0.3

# 通知 MuJoCo 重新计算所有派生量（重要！）
mujoco.mj_forward(model, data)

KP, KD = 400.0, 20.0

print("=" * 60)
print("🎬 Go2 点头（终极修复版）")
print(f"  初始姿态：{STAND}")
print(f"  初始高度：{data.qpos[2]:.3f} m")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        swing = 0.3 * np.sin(2 * np.pi * 0.8 * t)
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
            # 打印 3 个关键量：高度、FR大腿、FR小腿
            h = data.qpos[2]
            fr_thigh = data.qpos[8]
            fr_calf  = data.qpos[9]
            print(f"t={t:.2f}s | 高度 {h:.3f} | FR大腿 {fr_thigh:+.3f} | FR小腿 {fr_calf:+.3f}")

"""
Day 5-1 修复版: Go2 点头
改动：
1. 打印时用完整的 t（不是每 500 步才打印的过零点）
2. 加大 KP（120 → 300）
3. 摆动值用更明显的振幅（0.25 → 0.4）
4. 打印频率改为每 100 步（采样更密）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

BASE = np.array([0.0, 0.8, -1.5] * 4)

KP, KD = 300.0, 15.0   # ⭐ 加大！

data.qpos[2] = 0.4
mujoco.mj_forward(model, data)

print("=" * 60)
print("🎬 Go2 点头中...（1 Hz，振幅 0.4 rad，盯着窗口看）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = BASE.copy()
        # ⭐ 振幅加大到 0.4
        swing = 0.4 * np.sin(2 * np.pi * 1.0 * t)
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

        # ⭐ 每 100 步打印（采样更密，能看到正弦波）
        if step % 100 == 0:
            print(f"t={t:.2f}s | 摆动 {swing:+.3f} rad | 高度 {data.qpos[2]:.3f} m")

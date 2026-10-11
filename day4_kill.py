"""
Day 4 终极实验: 用巨大力矩掀翻机器人
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

STAND = np.array([0.0, 0.8, -1.5] * 4)
KP, KD = 60.0, 3.5

data.qpos[2] = 0.4
mujoco.mj_forward(model, data)

print("=" * 60)
print("🎬 两阶段实验")
print("  阶段 1（前 1 秒）：站立")
print("  阶段 2（1 秒后）：施加 2000 N·m 扭矩 → 必然翻倒")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        for i in range(12):
            q  = data.qpos[7 + i]
            dq = data.qvel[6 + i]
            data.ctrl[i] = KP * (STAND[i] - q) - KD * dq

        if t > 1.0:
            # ⭐ 直接给 2000 N·m 扭矩，是之前的 30 倍！
            data.xfrc_applied[1] = [0, 0, 0, 2000, 0, 0]
        else:
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

        mujoco.mj_step(model, data)
        step += 1

        if step % 200 == 0:
            quat = data.qpos[3:7]
            print(f"t={t:.1f}s | z={data.qpos[2]:.3f} | 扭矩={data.xfrc_applied[1][3]:.0f} | 四元数[{quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f}]")

        time.sleep(0.01)   # 慢速

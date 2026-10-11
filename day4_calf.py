"""
Day 4: 让小腿周期性摆动（明显可见）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 站立姿态
STAND = np.array([0.0, 0.8, -1.5] * 4)

KP = 100.0   # 合理范围
KD = 5.0

data.qpos[2] = 0.4
mujoco.mj_forward(model, data)

print("📐 站立姿态 + 小腿周期性摆动")
print("🎬 看右前腿小腿（FR_calf）在摆动\n")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001   # 时间

        # 让所有小腿做正弦摆动（明显可见）
        target = STAND.copy()
        for leg in range(4):
            # 每条腿的小腿关节做 ±0.5 rad 正弦摆动
            target[leg * 3 + 2] = -1.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)

        # PD 控制
        for i in range(12):
            q  = data.qpos[7 + i]
            dq = data.qvel[6 + i]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        step += 1

        if step % 500 == 0:
            fr_calf = data.qpos[7 + 2]
            print(f"步数 {step:5d} | 高度 {data.qpos[2]:.3f} | FR小腿角度 {fr_calf:+.3f} rad")

        time.sleep(0.001)

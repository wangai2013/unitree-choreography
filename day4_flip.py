"""
Day 4 破局: 让 Go2 翻个跟头（视觉绝对可见）
原理：直接给机身施加一个强力矩，看它倒地/翻转
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

KP = 60.0
KD = 3.5

data.qpos[2] = 0.4
mujoco.mj_forward(model, data)

print("=" * 60)
print("🎬 实验分两阶段：")
print("  阶段 1（前 3 秒）：正常站立")
print("  阶段 2（3 秒后）：给机身施加巨大扭矩 → 翻倒")
print("=" * 60)
print("👀 盯着窗口看！3 秒后机器人应该会翻倒！\n")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001   # 仿真时间

        # PD 控制（保持站立）
        for i in range(12):
            q  = data.qpos[7 + i]
            dq = data.qvel[6 + i]
            data.ctrl[i] = KP * (STAND[i] - q) - KD * dq

        # 3 秒后开始"捣乱"：给机身直接施加扭矩
        if t > 3.0:
            # xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]
            # body 1 是 base_link（机身根节点）
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 500.0]  # 绕 z 轴 500 N·m 扭矩
        else:
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

        mujoco.mj_step(model, data)
        step += 1

        if step % 500 == 0:
            # 打印机身姿态（四元数）
            quat = data.qpos[3:7]   # w, x, y, z
            print(f"时间 {t:.1f}s | 高度 {data.qpos[2]:.3f} | 机身四元数 [{quat[0]:+.2f}, {quat[1]:+.2f}, {quat[2]:+.2f}, {quat[3]:+.2f}]")

        time.sleep(0.001)

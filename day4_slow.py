"""
Day 4 破局: 慢动作翻倒 + 强制刷新
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
print("🎬 慢动作实验（每帧等待 10ms，肉眼可见）")
print("  第 1 阶段（前 2 秒）：正常站立")
print("  第 2 阶段（2 秒后）：施加扭矩 → 慢慢翻倒")
print("=" * 60)
print("\n⚠️  请盯着窗口看！不要看终端！\n")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # PD 控制
        for i in range(12):
            q  = data.qpos[7 + i]
            dq = data.qvel[6 + i]
            data.ctrl[i] = KP * (STAND[i] - q) - KD * dq

        # 逐渐加大扭矩（不是瞬间给 500，而是从 0 慢慢加到 200）
        if t > 2.0:
            torque = (t - 2.0) * 100.0   # 每秒加 100 N·m
            torque = min(torque, 200.0)  # 上限 200
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, torque]
        else:
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

        mujoco.mj_step(model, data)
        step += 1

        if step % 200 == 0:
            print(f"时间 {t:.1f}s | 高度 {data.qpos[2]:.3f} | 扭矩 {data.xfrc_applied[1][5]:.1f} N·m")

        # ⭐ 关键：放慢速度，让每帧都渲染
        time.sleep(0.01)   # 10ms 每步 → 肉眼可见

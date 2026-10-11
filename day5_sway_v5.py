"""
Day 5-4 最终修正: Go2 左右摇晃（重心移动，不位移）
关键：
- 髋关节保持 0（不摆动）
- 左侧腿"伸长"、右侧腿"缩短" → 身体向右倾
- 然后反过来 → 身体向左倾
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
print("🎬 Go2 左右摇晃（重心移动）")
print("  左腿小腿伸长 + 右腿小腿缩短 → 身体右倾")
print("  然后反相 → 身体左倾")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        sway = 0.3 * np.sin(2 * np.pi * 0.8 * t)

        # ⭐ 关键：不改髋，改"小腿"
        # 左腿小腿（ctrl[5]=FL_calf, ctrl[11]=RL_calf）：改变 → 让腿变长/短
        # 右腿小腿（ctrl[2]=FR_calf, ctrl[8]=RR_calf）：反相
        target[2] = -1.5 - sway   # FR 小腿：sway 正时缩短
        target[5] = -1.5 + sway   # FL 小腿：sway 正时伸长
        target[8] = -1.5 - sway   # RR 小腿
        target[11] = -1.5 + sway  # RL 小腿

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 100 == 0:
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={t:.2f}s | 摆动 {sway:+.3f} | 位置 (x={x:+.3f}, y={y:+.3f}, z={z:.3f})")

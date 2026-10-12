"""
直接测：FR 大腿固定 4 个值，看腿的实际位置
每个值持续 3 秒，共 12 秒
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/go2/scene.xml")
data = mujoco.MjData(model)
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

print("=" * 60)
print("🎬 依次测 FR 大腿 = 0.8 → 1.5 → 0.0 → 0.8")
print("  每个值 3 秒")
print("=" * 60)
print("👀 盯住右前腿，观察")
print("  t=3~6s（大腿=1.5）腿往哪？")
print("  t=6~9s（大腿=0.0）腿往哪？")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        if t < 3.0:
            target[1] = 0.8     # 正常
        elif t < 6.0:
            target[1] = 1.5     # 增大
        elif t < 9.0:
            target[1] = 0.0     # 减小
        else:
            target[1] = 0.8     # 恢复

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, 300, 15)
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

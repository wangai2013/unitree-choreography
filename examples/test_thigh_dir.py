"""
直接测试: FR 大腿角度变化时腿往哪个方向动
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
print("🎬 测试 FR 大腿（ctrl[1]）的 3 个值：")
print("   0~3 秒：0.8（正常）")
print("   3~6 秒：0.3（减小）← 观察：腿往前？往后？")
print("   6~9 秒：1.3（增大）← 观察：腿往前？往后？")
print("=" * 60)
print("👀 盯住右前腿（画面右前方的那条腿）！")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        if t < 3.0:
            target[1] = 0.8        # 正常
        elif t < 6.0:
            target[1] = 0.3        # 减小
        else:
            target[1] = 1.3        # 增大

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, 300, 15)
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

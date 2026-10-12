"""Day 7: 舞蹈播放器（含可视化跳跃）"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import sys
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

DANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "dance1.json"
with open(DANCE_FILE, "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 舞蹈: {dance['name']}")
print("=" * 60)

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

def get_current_segment(t):
    acc = 0.0
    for seg in dance["sequence"]:
        if t < acc + seg["duration"]:
            return seg["action"], t - acc, seg["duration"]
        acc += seg["duration"]
    return None, None, None

step = 0
last_action = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        name, t_local, dur = get_current_segment(t)

        if name is None:
            target = STAND.copy()
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        else:
            if name != last_action:
                print(f"\n🎬 t={t:.1f}s → 动作: {name}")
                last_action = name
            target = ACTIONS[name](t_local, dur)

            # ⭐ 跳跃时施加向上的力
            if name == "jump":
                phase = t_local / dur
                if 0.3 < phase < 0.5:
                    data.xfrc_applied[1] = [0, 0, 10, 0, 0, 0]   # 200N 向上
                else:
                    data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
            else:
                data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

        if step % 1000 == 0:
            print(f"  t={t:.1f}s | {(name or '结束'):8s} | 高度 {data.qpos[2]:.3f}")

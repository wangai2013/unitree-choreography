"""
Day 15: 群控 JSON 播放器
从 JSON 加载 4 台群控舞蹈
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import sys
import os

# 允许从项目根目录或 examples/ 运行
if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")

MUSIC_ON = True
DANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "dances/group_demo.json"

with open(DANCE_FILE, "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 群控舞蹈: {dance['name']}")
print(f"🥁 BPM: {dance['bpm']}")
print("=" * 60)

# 音乐
if MUSIC_ON:
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(dance.get("music", "music.mp3"))
        music_ready = True
    except Exception as e:
        print(f"⚠️ 音乐加载失败: {e}")
        music_ready = False
else:
    music_ready = False

# 模型
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 4 台映射
N = 4
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

CTRL_TO_QPOS = []
CTRL_TO_QVEL = []
for r in range(N):
    qpos_r = np.zeros(12, dtype=int)
    qvel_r = np.zeros(12, dtype=int)
    for i in range(12):
        jid = model.actuator_trnid[r * 12 + i][0]
        qpos_r[i] = model.jnt_qposadr[jid]
        qvel_r[i] = model.jnt_dofadr[jid]
    CTRL_TO_QPOS.append(qpos_r)
    CTRL_TO_QVEL.append(qvel_r)

STAND = np.array([0.0, 0.8, -1.5] * 4)
for r in range(N):
    for i in range(12):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0
HEIGHT = 0.32
FORMATION = dance["formation"]

# 计算时间轴
BPM = dance["bpm"]
BEAT_INTERVAL = 60.0 / BPM

timeline = []
acc = 0.0
for seg in dance["sequence"]:
    dur = seg["beats"] * BEAT_INTERVAL
    timeline.append((seg["action"], acc, acc + dur))
    acc += dur
total_duration = acc

print(f"⏱  总时长: {total_duration:.1f} 秒")
print("=" * 60)

def get_action(t):
    for action, t_start, t_end in timeline:
        if t_start <= t < t_end:
            return action, t - t_start
    return None, None

# 主循环
step = 0
music_started = False
last_action = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        if t >= 2.0 and music_ready and not music_started:
            pygame.mixer.music.play()
            music_started = True

        music_t = t - 2.0

        targets = [STAND.copy() for _ in range(N)]

        if music_t > 0:
            action, t_local = get_action(music_t)

            if action != last_action and action is not None:
                print(f"🎬 {music_t:.2f}s → {action}")
                last_action = action

            if action == "sync_nod":
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
                for r in range(N):
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "wave_nod":
                for r in range(N):
                    t_r = max(0, t_local - r * 0.3)
                    swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "mirror_sway":
                sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
                for r in range(N):
                    sign = +1 if r % 2 == 0 else -1
                    targets[r][1] = 0.8 + sign * sway
                    targets[r][4] = 0.8 - sign * sway

        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = FORMATION[r][0]
            data.qpos[QPOS_BASES[r] + 1] = FORMATION[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT
            data.qvel[QPOS_BASES[r]:QPOS_BASES[r] + 6] = 0

            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_ready:
    pygame.mixer.music.stop()

"""
Day 19: 带窗口的 DDS 控制仿真（锁位置版）
关键：mj_step 后锁定机身 x/y + 姿态（防漂移）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmd_default,
    unitree_go_msg_dds__LowState_ as LowState_default,
)
from robot_utils import build_ctrl_to_qpos_map

# ===== 1. MuJoCo =====
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)
print(f"✅ MuJoCo: nq={model.nq}, nu={model.nu}")

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

# ===== 2. DDS =====
print("🔌 DDS 初始化...")
ChannelFactoryInitialize(0, "lo0")

pub_lowstate = ChannelPublisher("rt/lowstate", LowState_)
pub_lowstate.Init()

latest_cmd = [None]
cmd_count = [0]

def cmd_handler(msg):
    latest_cmd[0] = msg
    cmd_count[0] += 1

sub_lowcmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub_lowcmd.Init(cmd_handler, 10)

print("✅ DDS 就绪")

# ===== 3. 控制线程 =====
def control_thread():
    time.sleep(1.0)
    print("🎬 控制线程开始发送 DDS 指令...")
    
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    step = 0
    while True:
        t = step * 0.02
        
        cmd = LowCmd_default()
        swing = 0.4 * np.sin(2 * np.pi * 1.0 * t)
        
        for i in range(12):
            target = STAND[i]
            if i == 2 or i == 5:
                target = -1.5 + swing
            
            cmd.motor_cmd[i].q = target
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 300.0
            cmd.motor_cmd[i].kd = 15.0
            cmd.motor_cmd[i].tau = 0.0
        
        pub.Write(cmd)
        step += 1
        time.sleep(0.02)

t = threading.Thread(target=control_thread, daemon=True)
t.start()

# ===== 4. 主循环 =====
print("\n" + "=" * 60)
print("🎬 打开 MuJoCo 窗口...")
print("=" * 60)

step = 0
last_print = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 应用 DDS 指令
        if latest_cmd[0] is not None:
            cmd = latest_cmd[0]
            for i in range(12):
                q_target = cmd.motor_cmd[i].q
                kp = cmd.motor_cmd[i].kp
                kd = cmd.motor_cmd[i].kd
                
                q = data.qpos[CTRL_TO_QPOS[i]]
                dq = data.qvel[CTRL_TO_QVEL[i]]
                data.ctrl[i] = kp * (q_target - q) - kd * dq
        
        # 推进物理
        mujoco.mj_step(model, data)
        
        # ⭐ 锁位置（防漂移）
        data.qpos[0] = 0.0     # x
        data.qpos[1] = 0.0     # y
        # z 不锁（让腿撑住）
        data.qpos[3] = 1.0     # 四元数（姿态正向）
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        # 清速度
        data.qvel[0] = 0       # vx
        data.qvel[1] = 0       # vy
        data.qvel[3] = 0       # wx
        data.qvel[4] = 0       # wy
        data.qvel[5] = 0       # wz
        
        # 刷新窗口
        viewer.sync()
        step += 1
        time.sleep(0.001)
        
        # 每 1 秒打印
        now = time.time()
        if now - last_print >= 1.0:
            print(f"t={now:.1f}s | 收到 {cmd_count[0]:4d} 条 | "
                  f"位置 ({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f}) | "
                  f"高度 {data.qpos[2]:.3f} | "
                  f"FR小腿 {data.qpos[CTRL_TO_QPOS[2]]:.3f}")
            last_print = now

print("\n✅ 结束")

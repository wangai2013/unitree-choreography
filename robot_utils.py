"""
宇树机器人 MuJoCo 控制通用工具
- 自动生成 ctrl ↔ qpos 映射
- 提供 PD 控制函数
- 所有 Go2/G1/H1 脚本都用这个
"""
import mujoco
import numpy as np

def build_ctrl_to_qpos_map(model):
    """
    自动生成 ctrl[i] → qpos 索引 的映射表
    返回：
      ctrl_to_qpos: np.array, 长度 = model.nu
      ctrl_to_qvel: np.array, 长度 = model.nu
    """
    ctrl_to_qpos = []
    ctrl_to_qvel = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i][0]
        ctrl_to_qpos.append(model.jnt_qposadr[joint_id])
        ctrl_to_qvel.append(model.jnt_dofadr[joint_id])
    return np.array(ctrl_to_qpos), np.array(ctrl_to_qvel)

def get_actuator_names(model):
    """返回所有执行器的名字列表"""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
            for i in range(model.nu)]

def get_joint_names(model):
    """返回所有关节的名字列表"""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) 
            for i in range(model.njnt)]

def pd_control(model, data, ctrl_to_qpos, ctrl_to_qvel, target, kp, kd):
    """
    通用 PD 控制器
    target: 目标关节角度（按 ctrl 顺序）
    """
    for i in range(model.nu):
        q  = data.qpos[ctrl_to_qpos[i]]
        dq = data.qvel[ctrl_to_qvel[i]]
        data.ctrl[i] = kp * (target[i] - q) - kd * dq

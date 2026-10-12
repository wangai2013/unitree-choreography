# 真机部署指南（Sim-to-Real）

> 从 "MuJoCo 仿真" 到 "真实宇树 Go2" 的完整路径

---

## 一、前置准备

### 1.1 硬件
- Unitree Go2（Air / Pro / EDU 任意型号）
- Mac M5（开发主机）
- 路由器（推荐 5GHz WiFi，低延迟）
- 网线（可选，更稳定）

### 1.2 软件
- Python 3.10
- unitree_sdk2py（已装）
- cyclonedds（已装）

### 1.3 网络配置

Go2 默认 IP：192.168.123.161（机身）
电脑需要同网段：192.168.123.xxx

连接方式：
- 方式 A（推荐）：网线直连 Go2 + Mac
- 方式 B：WiFi 连同一路由器

检查连接：
    ping 192.168.123.161

预期：64 bytes from 192.168.123.161... （说明通）

---

## 二、仿真 vs 真机的差异

| 项目 | 仿真 | 真机 |
|------|------|------|
| 网络接口 | lo0（回环） | en0 / en1（真实网卡） |
| 物理引擎 | MuJoCo | Go2 本体 |
| 状态发布 | 我们发布 rt/lowstate | Go2 内部发布 |
| 指令订阅 | 我们订阅 rt/lowcmd | Go2 内部订阅 |
| 平衡控制 | 我们软锁位置 | Go2 内置平衡器 |
| 代码 | —— | 改 3 行 |

---

## 三、代码迁移（3 步）

### 第 1 步：改网络接口

仿真：
    ChannelFactoryInitialize(0, "lo0")

真机：
    ChannelFactoryInitialize(0, "en0")

查看网卡：
    ifconfig | grep -E "^en"

### 第 2 步：删掉 MuJoCo 相关代码

仿真有：
    import mujoco
    model = mujoco.MjModel.from_xml_path(...)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

真机不需要（删掉）

### 第 3 步：保留 DDS 收发

真机代码：
    pub_lowcmd = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub_lowcmd.Init()

    sub_lowstate = ChannelSubscriber("rt/lowstate", LowState_)
    sub_lowstate.Init(handler, 10)

    cmd = LowCmd_default()
    for i in range(12):
        cmd.motor_cmd[i].q = 0.8
        cmd.motor_cmd[i].kp = 60.0
        cmd.motor_cmd[i].kd = 3.5

    pub_lowcmd.Write(cmd)

---

## 四、真机部署示例

### 4.1 站立（最简单）

    import time
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_default

    # 1. 连接
    ChannelFactoryInitialize(0, "en0")

    # 2. 发布者
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    # 3. 站立姿态
    STAND = [0.0, 0.8, -1.5] * 4

    # 4. 循环发送
    while True:
        cmd = LowCmd_default()
        for i in range(12):
            cmd.motor_cmd[i].q = STAND[i]
            cmd.motor_cmd[i].kp = 60.0
            cmd.motor_cmd[i].kd = 3.5
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        pub.Write(cmd)
        time.sleep(0.002)

### 4.2 点头

参考 examples/day18_dds_control_sim.py：
- 把 lo0 改成 en0
- 删掉 MuJoCo 相关
- 其他不变

### 4.3 安全停止

    cmd = LowCmd_default()
    for i in range(12):
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    pub.Write(cmd)

---

## 五、安全须知

### 5.1 第一次真机部署

- 用支架撑起（避免摔倒）
- 小幅度动作（不要上"跳跃"）
- 有人在旁边（随时按急停）
- 充满电（避免中途断电）

### 5.2 紧急停止

- 硬件：Go2 机身有急停按钮
- 软件：发"零力矩"指令

### 5.3 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 连不上 | 网段不同 | 改成 192.168.123.xxx |
| 没响应 | 网卡错 | ifconfig 查真实网卡 |
| 抖动 | KP/KD 不对 | 调参 |
| 摔倒 | 位置没锁 | 用 SportClient 平衡 |

---

## 六、进阶方向

### 6.1 高层运动接口（推荐）

不用管关节，直接"命令机器人走路"：

    from unitree_sdk2py.go2.sport.sport_client import SportClient

    client = SportClient()
    client.SetTimeout(10.0)
    client.Init()

    client.StandUp()           # 站起
    client.Move(0.3, 0, 0)     # 前进 0.3 m/s
    client.Move(0, 0, 0.5)     # 转圈
    client.StopMove()          # 停

优势：
- 稳定（官方步态）
- 简单（不用管平衡）
- 不精细（不能做"挥手"）

### 6.2 多机群控（真机）

架构：
    Mac
    ├── 路由器（5GHz WiFi）
    │   ├── Go2-A (192.168.123.161)
    │   ├── Go2-B (192.168.123.162)
    │   ├── Go2-C (192.168.123.163)
    │   └── Go2-D (192.168.123.164)
    └── 控制程序
        ├── 给 A 发 lowcmd
        ├── 给 B 发 lowcmd
        └── ...

注意：
- 每台机器人独立 IP（需要配置）
- 通信延迟（10~50 ms）
- 时间同步（NTP / PTP）

### 6.3 现场表演

- 场地：平整地板 + 标记
- 对位：UWB / 视觉标记
- 音乐：Mac 播放 + DDS 触发
- 安全：急停按钮 + 安全边界

---

## 七、代码对照表

| 仿真 | 真机 |
|------|------|
| ChannelFactoryInitialize(0, "lo0") | ChannelFactoryInitialize(0, "en0") |
| model = mujoco.MjModel... | 删除 |
| mujoco.mj_step() | 删除 |
| pub_lowstate.Write() | 删除（Go2 自动发） |
| sub_lowcmd.Init() | 删除（Go2 自动收） |
| pub_lowcmd.Write() | 保留 |
| sub_lowstate.Init() | 保留 |

---

## 八、总结

"仿真 → 真机"的本质：
- DDS 协议不变（rt/lowcmd / rt/lowstate）
- 消息结构不变（LowCmd_ / LowState_）
- 只改"对端"（MuJoCo → Go2）

19 天学习，你已拥有"真机部署"的全部知识。
只差"真机"这一个设备。

---

## 附：相关命令速查

    # 查看网卡
    ifconfig | grep -E "^en"

    # Ping Go2
    ping 192.168.123.161

    # 运行仿真
    mjpython examples/day19_dds_gui.py

    # 运行真机（改网卡后）
    python examples/real_robot_stand.py

# 动作库参考

> 所有可用动作的说明

---

## 一、基础动作

| 动作名 | 描述 | 关节 | 参数 |
|--------|------|------|------|
| stand | 站立 | 全身 | 无 |
| nod | 点头 | FR/FL calf | 1.5 Hz, ±0.4 |
| sway | 摇摆 | FR/FL thigh | 1.0 Hz, ±0.15 |
| bow | 鞠躬 | 前腿 | 半波 |
| squat | 下蹲 | 全身 | 半波 |

## 二、进阶动作

| 动作名 | 描述 | 难度 |
|--------|------|------|
| jump | 跳跃 | 中等 |
| wave_hi | 挥手 | 较难 |
| kick | 踢腿 | 较难 |
| spin | 转圈 | 中等 |

## 三、群控动作

| 动作名 | 描述 | 台数 |
|--------|------|------|
| sync_nod | 同步点头 | 2~N |
| wave_nod | 波浪点头 | 2~N |
| mirror_sway | 镜像摇摆 | 2~N |

## 四、队形

| 队形 | 坐标示例 |
|------|----------|
| square | 5×2 矩阵 |
| diamond | 菱形 |
| line | 一字 |
| circle | 圆形 |
| arrow | 箭头 |
| heart | 心形 |

## 五、扩展自己的动作

在 src/actions.py 里加函数：

    def act_your_action(t_local, duration):
        target = STAND.copy()
        f = fade(t_local, duration)
        # 你的动作逻辑
        return target

    ACTIONS["your_action"] = act_your_action

然后在 JSON 舞蹈里引用：

    {"action": "your_action", "beats": 4}

---

## 六、参数推荐

### 单机

| 场景 | KP | KD | 频率 |
|------|-----|-----|------|
| 站立 | 60 | 3.5 | —— |
| 点头 | 300 | 15 | 1.5 Hz |
| 摇摆 | 300 | 15 | 1.0 Hz |
| 跳跃 | 300 | 15 | 半波 |

### 群控

| 场景 | 台数 | 延迟 |
|------|------|------|
| 同步 | 2~N | 0 |
| 波浪 | 2~N | 0.15~0.3s |
| 镜像 | 2~N | 反相 |

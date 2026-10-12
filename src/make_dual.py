"""
生成双机 Go2 XML（完整版）
- 复制 base_link → base_link_B
- 复制 actuator → actuator_B（加 _B 后缀）
"""
import os
import re

GO2_DIR = "unitree_mujoco/unitree_robots/go2"
GO2_XML = os.path.join(GO2_DIR, "go2.xml")

with open(GO2_XML, "r") as f:
    content = f.read()
lines = content.split("\n")


def find_section(lines, tag):
    """精确找到顶层 <tag>...</tag> 段"""
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"<{tag}>"):
            start = i
            break
    if start is None:
        return None, None, ""
    depth = 0
    end = None
    for i in range(start, len(lines)):
        line = lines[i]
        opens = len(re.findall(f'<{tag}[ >]', line))
        closes = len(re.findall(f'</{tag}>', line))
        depth += opens - closes
        if depth == 0 and i >= start:
            end = i
            break
    return start, end, "\n".join(lines[start:end+1])


# ===== 1. 提取各段 =====
_, _, default_section = find_section(lines, "default")
_, _, asset_section = find_section(lines, "asset")
_, _, actuator_section = find_section(lines, "actuator")

print(f"✅ 提取: default / asset / actuator")

# ===== 2. 提取 base_link =====
start_idx = next(i for i, l in enumerate(lines) if '<body name="base_link"' in l)
depth = 0
end_idx = None
for i in range(start_idx, len(lines)):
    line = lines[i]
    line_nc = re.sub(r'<body[^>]*/>', '', line)
    depth += len(re.findall(r'<body[^>]*>', line_nc))
    depth -= len(re.findall(r'</body>', line_nc))
    if depth == 0 and i > start_idx:
        end_idx = i
        break

base_link_A = "\n".join(lines[start_idx:end_idx+1])
print(f"✅ base_link: line {start_idx+1} ~ {end_idx+1}")

# ===== 3. 复制 base_link → B =====
base_link_B = re.sub(r'name="([^"]+)"', r'name="\1_B"', base_link_A)
base_link_B = base_link_B.replace('pos="0 0 0.445"', 'pos="1.2 0 0.445"', 1)

# ===== 4. 提取 actuator 内部 =====
actuator_A_inner = (
    actuator_section
    .replace("<actuator>", "")
    .replace("</actuator>", "")
    .strip()
)

# ===== 5. 复制 actuator → B（改 name + joint）=====
actuator_B_inner = re.sub(r'name="([^"]+)"', r'name="\1_B"', actuator_A_inner)
actuator_B_inner = re.sub(r'joint="([^"]+)"', r'joint="\1_B"', actuator_B_inner)

# ⭐ 每个 motor 的 ctrlrange 也可以保留（复制即可）
print(f"✅ actuator 已复制（含 joint 重定向）")

# ===== 6. 拼接完整 XML =====
dual_xml = f'''<mujoco model="go2 dual scene">
    <compiler angle="radian" meshdir="assets"/>
    <option gravity="0 0 -9.81"/>

{default_section}

{asset_section}

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" rgba="0.3 0.5 0.7 1"/>

{base_link_A}

{base_link_B}
    </worldbody>

    <actuator>
{actuator_A_inner}
{actuator_B_inner}
    </actuator>
</mujoco>
'''

out_path = os.path.join(GO2_DIR, "scene_dual_final.xml")
with open(out_path, "w") as f:
    f.write(dual_xml)
print(f"\n✅ 生成: {out_path}")

# ===== 7. 测试加载 =====
import mujoco
try:
    m = mujoco.MjModel.from_xml_path(out_path)
    print(f"\n✅ 加载成功: nq={m.nq}, nu={m.nu}, nbody={m.nbody}")
    if m.nq == 38 and m.nu == 24:
        print("🎉 完美！2 台 Go2 + 24 个电机！")
    elif m.nq == 38:
        print(f"⚠️ 有 2 台，但 nu={m.nu}（应该是 24）")
    else:
        print(f"❓ nq={m.nq}")
    
    # 打印所有电机名字（验证 A/B）
    print("\n📋 前 24 个电机：")
    for i in range(min(m.nu, 30)):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i:2d}] {name}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

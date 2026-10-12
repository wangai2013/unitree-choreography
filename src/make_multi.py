"""
Day 17: 生成 N 台 Go2 XML（可配置台数）
用法：python src/make_multi.py [N]
默认 N=10
"""
import os
import re
import sys

# 台数（默认 10）
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

GO2_DIR = "unitree_mujoco/unitree_robots/go2"
GO2_XML = os.path.join(GO2_DIR, "go2.xml")

with open(GO2_XML, "r") as f:
    content = f.read()
lines = content.split("\n")


def find_section(lines, tag):
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


_, _, default_section = find_section(lines, "default")
_, _, asset_section = find_section(lines, "asset")
_, _, actuator_section = find_section(lines, "actuator")

# 提取 base_link
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

base_link_orig = "\n".join(lines[start_idx:end_idx+1])
actuator_inner_orig = actuator_section.replace("<actuator>", "").replace("</actuator>", "").strip()

# 自动排布：矩阵（不超过 5 列）
COLS = min(N, 5)
ROWS = (N + COLS - 1) // COLS
SPACING_X = 1.2
SPACING_Y = 0.8

positions = []
for i in range(N):
    row = i // COLS
    col = i % COLS
    x = (col - (COLS - 1) / 2) * SPACING_X
    y = (row - (ROWS - 1) / 2) * SPACING_Y
    positions.append((x, y))

# 生成 body/actuator
body_sections = []
actuator_sections = []

for i, (x, y) in enumerate(positions):
    suffix = chr(65 + i) if i < 26 else f"R{i}"
    if i == 0:
        body = base_link_orig.replace('pos="0 0 0.445"', f'pos="{x} {y} 0.445"', 1)
        act = actuator_inner_orig
    else:
        body = re.sub(r'name="([^"]+)"', rf'name="\1_{suffix}"', base_link_orig)
        body = body.replace('pos="0 0 0.445"', f'pos="{x} {y} 0.445"', 1)
        act = re.sub(r'name="([^"]+)"', rf'name="\1_{suffix}"', actuator_inner_orig)
        act = re.sub(r'joint="([^"]+)"', rf'joint="\1_{suffix}"', act)
    
    body_sections.append(body)
    actuator_sections.append(act)

xml = f'''<mujoco model="go2 multi scene ({N} robots)">
    <compiler angle="radian" meshdir="assets"/>
    <option gravity="0 0 -9.81"/>

{default_section}

{asset_section}

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" rgba="0.3 0.5 0.7 1"/>

{"".join(body_sections)}
    </worldbody>

    <actuator>
{"".join(actuator_sections)}
    </actuator>
</mujoco>
'''

out_path = os.path.join(GO2_DIR, f"scene_multi_{N}.xml")
with open(out_path, "w") as f:
    f.write(xml)

print(f"✅ 生成: {out_path}")

# 测试
import mujoco
m = mujoco.MjModel.from_xml_path(out_path)
print(f"✅ nq={m.nq}, nu={m.nu}, nbody={m.nbody}")
if m.nq == 19 * N and m.nu == 12 * N:
    print(f"🎉 {N} 台 Go2 成功！")

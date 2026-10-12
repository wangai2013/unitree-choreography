"""
Day 10: 双机 Go2（用 spec API 复制）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

# ===== 1. 加载单机 spec =====
print("🔧 加载单机 spec...")
spec = mujoco.MjSpec.from_file("unitree_mujoco/unitree_robots/go2/scene.xml")

# ===== 2. 找到 base_link =====
base_link = None
for b in spec.worldbody.bodies:
    if b.name == "base_link":
        base_link = b
        break

if base_link is None:
    print("❌ 找不到 base_link")
    exit(1)

print(f"✅ 找到 base_link: pos={base_link.pos}")

# ===== 3. 复制 base_link =====
# 注意：MuJoCo spec API 的复制方法可能不同，我们尝试几种
try:
    # 尝试用 spec.copy_body 或类似方法
    print("\n🔧 尝试复制 base_link...")
    
    # MuJoCo 3.x 支持 spec.copy() 复制整个模型
    # 我们先把当前 spec 保存为 template，然后创建新 spec
    spec2 = mujoco.MjSpec.from_file("unitree_mujoco/unitree_robots/go2/scene.xml")
    
    # 在 spec2 里找到 base_link
    base_link_2 = None
    for b in spec2.worldbody.bodies:
        if b.name == "base_link":
            base_link_2 = b
            break
    
    # 把第二个 base_link 移动到另一个位置
    base_link_2.pos = np.array([1.5, 0, 0.445])
    
    # 改名字避免冲突
    base_link_2.name = "base_link_B"
    
    # ===== 4. 合并两个 spec =====
    # MuJoCo 支持 spec.attach() 把另一个 spec 附加进来
    print("🔧 合并两个 spec...")
    spec.attach(spec2)
    
    print("✅ 合并成功")
    
except Exception as e:
    print(f"❌ 失败: {e}")
    print("\n💡 MuJoCo spec API 可能不支持此操作")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== 5. 编译成 model =====
print("\n🔧 编译 model...")
try:
    model = spec.compile()
    data = mujoco.MjData(model)
    print(f"✅ 编译成功: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")
    
    if model.nq == 19:
        print("⚠️  只有 1 台机器人（复制失败）")
    elif model.nq == 38:
        print("🎉 成功！有 2 台机器人！")
    else:
        print(f"❓ 未知状态：nq={model.nq}")
except Exception as e:
    print(f"❌ 编译失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

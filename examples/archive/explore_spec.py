"""探索 MuJoCo spec 结构"""
import mujoco

spec = mujoco.MjSpec.from_file("unitree_mujoco/unitree_robots/go2/scene.xml")
print(f"✅ 加载成功")
print(f"📋 spec 中 bodies: {len(spec.bodies)} 个")

# 列出顶层 body
print("\n📋 worldbody 的直接子 body:")
for b in spec.worldbody.bodies:
    print(f"  名字: '{b.name}'")
    print(f"  位置: {b.pos}")
    print(f"  子 body 数: {len(b.bodies)}")
    print(f"  子 joint 数: {len(b.joints)}")
    print(f"  子 geom 数: {len(b.geoms)}")
    print(f"  子 freejoint: {b.has_free_joint() if hasattr(b, 'has_free_joint') else '未知'}")
    print()

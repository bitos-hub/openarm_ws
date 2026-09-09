import re, os

base = "assets/robot/openarm_v1.0/urdf"

# 1. openarm_v10.urdf.xacro
f1 = os.path.join(base, "openarm_v10.urdf.xacro")
with open(f1, "r") as f:
    c1 = f.read()
c1 = re.sub(r'(<xacro:arg\s+name="use_fake_hardware"[^/>]*/>)', r'\1\n  <xacro:arg name="use_mujoco" default="false" />', c1)
c1 = re.sub(r'(use_fake_hardware="\$\(arg use_fake_hardware\)")', r'\1\n        use_mujoco="$(arg use_mujoco)"', c1)
with open(f1, "w") as f:
    f.write(c1)

# 2. openarm_robot.xacro
f2 = os.path.join(base, "robot/openarm_robot.xacro")
with open(f2, "r") as f:
    c2 = f.read()
c2 = re.sub(r'(use_fake_hardware:=false)', r'\1\n        use_mujoco:=false', c2, count=1)
c2 = re.sub(r'(use_fake_hardware="\$\{use_fake_hardware\}")', r'\1\n              use_mujoco="${use_mujoco}"', c2)
with open(f2, "w") as f:
    f.write(c2)

# 3. openarm.bimanual.ros2_control.xacro
f3 = os.path.join(base, "ros2_control/openarm.bimanual.ros2_control.xacro")
with open(f3, "r") as f:
    c3 = f.read()
c3 = re.sub(r'(use_fake_hardware:=\^\|false)', r'\1\n                 use_mujoco:=false', c3)
target = '<xacro:unless value="${use_fake_hardware}">'
replacement = """<xacro:if value="${use_mujoco}">
            <plugin>mujoco_ros2_control/MujocoSystem</plugin>
          </xacro:if>
          <xacro:unless value="${use_fake_hardware or use_mujoco}">"""
c3 = c3.replace(target, replacement)
with open(f3, "w") as f:
    f.write(c3)

print("Parche aplicado exitosamente.")

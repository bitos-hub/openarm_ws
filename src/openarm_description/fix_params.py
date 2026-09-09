import os

base = "assets/robot/openarm_v1.0/urdf"

# 1. Modificar openarm_v10.urdf.xacro
f1_path = os.path.join(base, "openarm_v10.urdf.xacro")
with open(f1_path, "r") as f:
    c1 = f.read()
c1 = c1.replace(
    '<xacro:arg name="use_fake_hardware" default="false" />',
    '<xacro:arg name="use_fake_hardware" default="false" />\n  <xacro:arg name="use_mujoco" default="false" />'
)
c1 = c1.replace(
    'use_fake_hardware="$(arg use_fake_hardware)"',
    'use_fake_hardware="$(arg use_fake_hardware)"\n        use_mujoco="$(arg use_mujoco)"'
)
with open(f1_path, "w") as f:
    f.write(c1)

# 2. Modificar openarm_robot.xacro
f2_path = os.path.join(base, "robot/openarm_robot.xacro")
with open(f2_path, "r") as f:
    c2 = f.read()
c2 = c2.replace(
    'use_fake_hardware:=false',
    'use_fake_hardware:=false\n        use_mujoco:=false',
    1
)
c2 = c2.replace(
    'use_fake_hardware="${use_fake_hardware}"',
    'use_fake_hardware="${use_fake_hardware}"\n              use_mujoco="${use_mujoco}"'
)
with open(f2_path, "w") as f:
    f.write(c2)

# 3. Modificar openarm.bimanual.ros2_control.xacro
f3_path = os.path.join(base, "ros2_control/openarm.bimanual.ros2_control.xacro")
with open(f3_path, "r") as f:
    c3 = f.read()

# Agregar use_mujoco a la lista de parametros params="..." de la macro
c3 = c3.replace('params="', 'params="use_mujoco:=false\n                 ', 1)

# Insertar el plugin de MuJoCo dentro del bloque de hardware
target = '<xacro:unless value="${use_fake_hardware}">'
replacement = """<xacro:if value="${use_mujoco}">
            <plugin>mujoco_ros2_control/MujocoSystem</plugin>
          </xacro:if>
          <xacro:unless value="${use_fake_hardware or use_mujoco}">"""
c3 = c3.replace(target, replacement)

with open(f3_path, "w") as f:
    f.write(c3)

print("Parámetro use_mujoco inyectado correctamente.")

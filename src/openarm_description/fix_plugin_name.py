import os

path = "assets/robot/openarm_v1.0/urdf/ros2_control/openarm.bimanual.ros2_control.xacro"

with open(path, "r") as f:
    content = f.read()

# Reemplazar la clase del plugin por MujocoSystemInterface
content = content.replace("mujoco_ros2_control/MujocoSystem", "mujoco_ros2_control/MujocoSystemInterface")

with open(path, "w") as f:
    f.write(content)

print("Nombre del plugin actualizado a MujocoSystemInterface.")

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Definir rutas a los paquetes
    # Ajusta 'openarm_description' y 'openarm_control' si tus paquetes se llaman diferente
    desc_pkg = get_package_share_directory('openarm_description')
    control_pkg = get_package_share_directory('openarm_control')

    # Ruta al archivo URDF (basada en tu estructura del workspace)
    urdf_file = os.path.expanduser('~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1_edited.urdf')
    
    # Ruta al archivo de configuración de RViz que contiene el RobotModel y los InteractiveMarkers
    rviz_config_file = os.path.join(desc_pkg, 'rviz', 'openarm_teleop.rviz')

    # Leer el URDF para el robot_state_publisher
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 2. Nodo Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # 3. Nodo del Marcador Interactivo (tu archivo rviz_marker_node_3.py)
    interactive_marker_node = Node(
        package='openarm_control',
        executable='rviz_marker_node', # Nombre del ejecutable definido en tu setup.py o CMakeLists
        name='arm_interactive_marker_node',
        output='screen'
    )

    # 4. Nodo RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        interactive_marker_node,
        rviz_node
    ])
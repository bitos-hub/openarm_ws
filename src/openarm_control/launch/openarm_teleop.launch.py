import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # URDF file path
    urdf_file = os.path.expanduser('~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1_edited.urdf')
    
    # RViz configuration file path
    desc_pkg = get_package_share_directory('openarm_description')
    rviz_config_file = os.path.join(desc_pkg, 'rviz', 'openarm_teleop.rviz')

    # Read the URDF for robot_state_publisher
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # Interactive teleoperation node
    interactive_teleop_node = Node(
        package='openarm_control',
        executable='interactive_teleop_node',
        name='arm_interactive_marker_node',
        output='screen'
    )

    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        interactive_teleop_node,
        rviz_node
    ])
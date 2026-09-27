import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback

class ArmInteractiveMarkerNode(Node):
    def __init__(self):
        super().__init__('arm_interactive_marker_node')
        self.server = InteractiveMarkerServer(self, 'openarm_teleop_markers')

        self.pub_left = self.create_publisher(PoseStamped, '/openarm/left/target_pose', 10)
        self.pub_right = self.create_publisher(PoseStamped, '/openarm/right/target_pose', 10)

        self.sub_init_l = self.create_subscription(PoseStamped, '/openarm/left/current_pose', self._init_left, 10)
        self.sub_init_r = self.create_subscription(PoseStamped, '/openarm/right/current_pose', self._init_right, 10)

        self.left_initialized = False
        self.right_initialized = False

    def _init_left(self, msg):
        if not self.left_initialized:
            self._create_tcp_marker('left', msg.pose)
            self.left_initialized = True

    def _init_right(self, msg):
        if not self.right_initialized:
            self._create_tcp_marker('right', msg.pose)
            self.right_initialized = True

    def _create_tcp_marker(self, arm_name, pose):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = f"marker_{arm_name}"
        int_marker.description = f"TCP {arm_name.upper()}"
        int_marker.pose = pose
        int_marker.scale = 0.18

        # Controles 6DoF (ejes y aros) sin cubos opacos
        for axis_name, axis_vec in [('x', [1,0,0]), ('y', [0,0,1]), ('z', [0,1,0])]:
            control_t = InteractiveMarkerControl()
            control_t.name = f"move_{axis_name}"
            control_t.orientation.w = 1.0
            control_t.orientation.x = float(axis_vec[0])
            control_t.orientation.y = float(axis_vec[1])
            control_t.orientation.z = float(axis_vec[2])
            control_t.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control_t)

            control_r = InteractiveMarkerControl()
            control_r.name = f"rotate_{axis_name}"
            control_r.orientation.w = 1.0
            control_r.orientation.x = float(axis_vec[0])
            control_r.orientation.y = float(axis_vec[1])
            control_r.orientation.z = float(axis_vec[2])
            control_r.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control_r)

        self.server.insert(int_marker, feedback_callback=self._process_feedback)
        self.server.applyChanges()

    def _process_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            msg = PoseStamped()
            msg.header = feedback.header
            msg.pose = feedback.pose

            if feedback.marker_name == 'marker_left':
                self.pub_left.publish(msg)
            elif feedback.marker_name == 'marker_right':
                self.pub_right.publish(msg)

def main():
    rclpy.init()
    node = ArmInteractiveMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
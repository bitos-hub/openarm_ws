import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, InteractiveMarkerFeedback

class NaturalTeleopServer(Node):
    def __init__(self):
        super().__init__('natural_teleop_server')
        # Server
        self.server = InteractiveMarkerServer(self, 'openarm_teleop')
        
        # Publishers
        self.pub_target_l = self.create_publisher(PoseStamped, '/openarm/left/target_pose', 10)
        self.pub_target_r = self.create_publisher(PoseStamped, '/openarm/right/target_pose', 10)

        # Current TCPs' poses
        self.curr_pose_l = None
        self.curr_pose_r = None
        
        # Subscribers
        self.create_subscription(PoseStamped, '/openarm/left/current_pose', self._cb_curr_l, 10)
        self.create_subscription(PoseStamped, '/openarm/right/current_pose', self._cb_curr_r, 10)

        # Timer
        self.timer = self.create_timer(0.1, self._init_markers_once)
        
        # Flag
        self.initialized = False

    def _cb_curr_l(self, msg):
        """
        Get left TCP's current pose.
        """
        self.curr_pose_l = msg.pose

    def _cb_curr_r(self, msg):
        """
        Get right TCP's current pose.
        """
        self.curr_pose_r = msg.pose

    def _init_markers_once(self):
        """
        Initialize the markers only once.
        """
        if self.initialized or self.curr_pose_l is None or self.curr_pose_r is None:
            return

        self._create_arm_marker('left', self.curr_pose_l)
        self._create_arm_marker('right', self.curr_pose_r)
        
        self.server.applyChanges()
        self.initialized = True
        self.timer.cancel()
        self.get_logger().info("Teleoperation ready.")

    def _create_arm_marker(self, side, initial_pose):
        """
        Create the markers for both TCPs
        """
        # Setup
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = f"hand_sphere_{side}"
        int_marker.description = f"{side.upper()} TCP (6D)"
        int_marker.scale = 0.22
        int_marker.pose = initial_pose

        # Sphere for 3D translation
        sphere_marker = Marker()
        sphere_marker.type = Marker.SPHERE
        sphere_marker.scale.x = 0.06
        sphere_marker.scale.y = 0.06
        sphere_marker.scale.z = 0.06
        sphere_marker.color.r = 0.1 if side == 'left' else 0.9
        sphere_marker.color.g = 0.7
        sphere_marker.color.b = 0.2 if side == 'left' else 0.1
        sphere_marker.color.a = 0.85
        drag_control = InteractiveMarkerControl()
        drag_control.name = "sphere_move"
        drag_control.always_visible = True
        drag_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        drag_control.markers.append(sphere_marker)
        int_marker.controls.append(drag_control)

        # Ring for X-axis rotation
        rot_x_control = InteractiveMarkerControl()
        rot_x_control.name = "rotate_x"
        rot_x_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        rot_x_control.orientation_mode = InteractiveMarkerControl.INHERIT
        rot_x_control.orientation.w = 0.7071068
        rot_x_control.orientation.x = 0.7071068
        rot_x_control.orientation.y = 0.0
        rot_x_control.orientation.z = 0.0
        int_marker.controls.append(rot_x_control)

        # Ring for Y-axis rotation
        rot_y_control = InteractiveMarkerControl()
        rot_y_control.name = "rotate_y"
        rot_y_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        rot_y_control.orientation_mode = InteractiveMarkerControl.INHERIT
        rot_y_control.orientation.w = 0.7071068
        rot_y_control.orientation.x = 0.0
        rot_y_control.orientation.y = 0.0
        rot_y_control.orientation.z = 0.7071068
        int_marker.controls.append(rot_y_control)

        # Ring for Z-axis rotation
        rot_z_control = InteractiveMarkerControl()
        rot_z_control.name = "rotate_z"
        rot_z_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        rot_z_control.orientation_mode = InteractiveMarkerControl.INHERIT
        rot_z_control.orientation.w = 0.7071068
        rot_z_control.orientation.x = 0.0
        rot_z_control.orientation.y = 0.7071068
        rot_z_control.orientation.z = 0.0
        int_marker.controls.append(rot_z_control)

        # Insert the markers
        self.server.insert(
            int_marker, 
            feedback_callback=lambda fb, s=side: self._process_feedback(fb, s)
        )

    def _process_feedback(self, feedback, side):
        """
        Process the feedback from the user moving the markers and publish the
        new target poses
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            q = np.array([
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w
            ])
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-6:
                q /= q_norm
                feedback.pose.orientation.x = float(q[0])
                feedback.pose.orientation.y = float(q[1])
                feedback.pose.orientation.z = float(q[2])
                feedback.pose.orientation.w = float(q[3])

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose = feedback.pose

            if side == 'left':
                self.pub_target_l.publish(msg)
            else:
                self.pub_target_r.publish(msg)

def main():
    rclpy.init()
    node = NaturalTeleopServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
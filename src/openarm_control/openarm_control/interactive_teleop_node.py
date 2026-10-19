import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, InteractiveMarkerFeedback
from scipy.spatial.transform import Rotation as R
import tf2_ros

# Anthropomorphic limits configuration
R_MAX = 0.63  # Max. reach of the arm
R_MIN = 0.10  # Min. distance from the torso
SHOULDER_POS_LEFT = np.array([0.0, 0.18, 0.45])
SHOULDER_POS_RIGHT = np.array([0.0, -0.18, 0.45])

# Tool center points
TCP_FRAMES = {
    'left': 'openarm_left_hand_tcp',
    'right': 'openarm_right_hand_tcp'
}

class NaturalTeleopServer(Node):
    def __init__(self):
        super().__init__('natural_teleop_server')
        self.server = InteractiveMarkerServer(self, 'openarm_teleop')
        
        # Publishers of targets for the controllers
        self.pub_target_l = self.create_publisher(PoseStamped, '/openarm/left/target_pose', 10)
        self.pub_target_r = self.create_publisher(PoseStamped, '/openarm/right/target_pose', 10)
        self.pub_wrist_r  = self.create_publisher(Float64, '/openarm/right/wrist_cmd', 10)

        # Default static states 
        self.user_roll_l = 0.0
        self.user_roll_r = 0.0
        self.wrist_angle_r = 0.0

        # Initial reference states
        self.q_start = None
        self.wrist_start_r = 0.0
        self.roll_start_l = 0.0
        self.roll_start_r = 0.0

        # TF Listener to read the current position of TCPs in the 'world'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Wait for TFs to be available
        self.timer = self.create_timer(0.5, self._init_markers_once)
        self.initialized = False

    def _init_markers_once(self):
        # Initialization of the interactive markers
        if self.initialized:
            return
            
        pos_left = self._get_tcp_position('left', np.array([0.2, 0.25, 0.2]))
        pos_right = self._get_tcp_position('right', np.array([0.2, -0.25, 0.2]))

        self._create_arm_marker('left', SHOULDER_POS_LEFT, pos_left)
        self._create_arm_marker('right', SHOULDER_POS_RIGHT, pos_right)
        
        self.server.applyChanges()
        self.initialized = True
        self.timer.cancel()
        self.get_logger().info("Teleoperation server ready.")

    def _get_tcp_position(self, side, fallback_pos):
        # Get the TCP position w.r.t the 'world'
        tcp_frame = TCP_FRAMES[side]
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('world', tcp_frame, now, timeout=rclpy.duration.Duration(seconds=1.0))
            pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            self.get_logger().info(f"TCP {side} alineado en TF: {pos}")
            return pos
        except Exception as e:
            self.get_logger().warn(f"TF no disponible para {tcp_frame}, usando posición estimada: {e}")
            return fallback_pos

    def _create_arm_marker(self, side, shoulder_pos, initial_pos):
        # Create the visual structure for an arm

        #Setup
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = f"hand_sphere_{side}"
        int_marker.description = f"{side.upper()} hand"
        int_marker.scale = 0.15
        int_marker.pose.position.x = float(initial_pos[0])
        int_marker.pose.position.y = float(initial_pos[1])
        int_marker.pose.position.z = float(initial_pos[2])

        # Sphere for 3D free movement
        sphere_marker = Marker()
        sphere_marker.type = Marker.SPHERE
        sphere_marker.scale.x = 0.07
        sphere_marker.scale.y = 0.07
        sphere_marker.scale.z = 0.07
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

        # Roll rotation ring for the wrist
        roll_control = InteractiveMarkerControl()
        roll_control.name = "wrist_roll"
        roll_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        roll_control.orientation_mode = InteractiveMarkerControl.INHERIT
        roll_control.orientation.w = 0.7071
        roll_control.orientation.x = 0.0
        roll_control.orientation.y = 0.7071
        roll_control.orientation.z = 0.0
        int_marker.controls.append(roll_control)

        # Pitch rotation ring for the wirst (only right arm)
        if side == 'right':
            wrist_pitch_control = InteractiveMarkerControl()
            wrist_pitch_control.name = "wrist_flexion"
            wrist_pitch_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            wrist_pitch_control.orientation_mode = InteractiveMarkerControl.INHERIT
            wrist_pitch_control.orientation.w = 0.7071
            wrist_pitch_control.orientation.x = 0.7071
            wrist_pitch_control.orientation.y = 0.0
            wrist_pitch_control.orientation.z = 0.0
            int_marker.controls.append(wrist_pitch_control)

        # Register markers
        self.server.insert(
            int_marker, 
            feedback_callback=lambda fb, s=side, sh=shoulder_pos: self._process_feedback(fb, s, sh)
        )

    def _process_feedback(self, feedback, side, shoulder_pos):
        # Process the marker changes in real time

        # Current orientation
        q_curr = R.from_quat([
            feedback.pose.orientation.x,
            feedback.pose.orientation.y,
            feedback.pose.orientation.z,
            feedback.pose.orientation.w
        ])

        # Capture orientation when clicking
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.q_start = q_curr
            self.wrist_start_r = self.wrist_angle_r
            self.roll_start_l = self.user_roll_l
            self.roll_start_r = self.user_roll_r
            return

        # Reestablish reference when stopping clicking
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.q_start = None

        # Position and proyection in reachable space
        raw_pos = np.array([
            feedback.pose.position.x,
            feedback.pose.position.y,
            feedback.pose.position.z
        ])

        arm_vec = raw_pos - shoulder_pos
        dist = np.linalg.norm(arm_vec)

        if dist > 0.0001:
            dir_vec = arm_vec / dist
            clamped_dist = np.clip(dist, R_MIN, R_MAX)
            valid_pos = shoulder_pos + dir_vec * clamped_dist
        else:
            valid_pos = raw_pos
            dir_vec = np.array([1.0, 0.0, 0.0])

        valid_pos[2] = max(valid_pos[2], 0.02)

        # Make the z-axis point along the arm vector
        z_dir = dir_vec
        up_vector = np.array([0.0, 0.0, 1.0])

        if abs(np.dot(z_dir, up_vector)) > 0.95:
            up_vector = np.array([0.0, 1.0, 0.0])

        x_dir = np.cross(up_vector, z_dir)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        R_auto = np.column_stack((x_dir, y_dir, z_dir))

        # Calculate the angular delta of wrist flexion or roll
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE and self.q_start is not None:
            R_diff = self.q_start.inv() * q_curr
            rot_vec = R_diff.as_rotvec()

            if feedback.control_name == "wrist_flexion" and side == 'right':
                delta_flexion = rot_vec[0]
                self.wrist_angle_r = float(np.clip(self.wrist_start_r + delta_flexion, -1.134, 0.610))

            elif feedback.control_name == "wrist_roll":
                delta_roll = rot_vec[2] if abs(rot_vec[2]) > abs(rot_vec[0]) else rot_vec[0]
                if side == 'left':
                    self.user_roll_l = float(self.roll_start_l + delta_roll)
                else:
                    self.user_roll_r = float(self.roll_start_r + delta_roll)

        # Get the final quaternion using the orientation
        current_roll = self.user_roll_l if side == 'left' else self.user_roll_r
        R_roll = R.from_euler('z', current_roll).as_matrix()
        R_final = R_auto @ R_roll
        q_final = R.from_matrix(R_final).as_quat()

        # Update pose in RViz
        feedback.pose.position.x = float(valid_pos[0])
        feedback.pose.position.y = float(valid_pos[1])
        feedback.pose.position.z = float(valid_pos[2])
        feedback.pose.orientation.x = float(q_final[0])
        feedback.pose.orientation.y = float(q_final[1])
        feedback.pose.orientation.z = float(q_final[2])
        feedback.pose.orientation.w = float(q_final[3])

        self.server.setPose(feedback.marker_name, feedback.pose)
        self.server.applyChanges()

        # Publish PoseStamped objective
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(valid_pos[0])
        msg.pose.position.y = float(valid_pos[1])
        msg.pose.position.z = float(valid_pos[2])
        msg.pose.orientation.x = float(q_final[0])
        msg.pose.orientation.y = float(q_final[1])
        msg.pose.orientation.z = float(q_final[2])
        msg.pose.orientation.w = float(q_final[3])

        if side == 'left':
            self.pub_target_l.publish(msg)
        else:
            self.pub_target_r.publish(msg)
            wrist_msg = Float64()
            wrist_msg.data = self.wrist_angle_r
            self.pub_wrist_r.publish(wrist_msg)

def main():
    rclpy.init()
    node = NaturalTeleopServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
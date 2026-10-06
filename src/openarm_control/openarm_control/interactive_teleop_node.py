import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from scipy.spatial.transform import Rotation as R
import tf2_ros

# Configuración de límites antropomórficos
R_MAX = 0.63  # Alcance máximo del brazo (metros)
R_MIN = 0.10  # Límite cercano al torso
SHOULDER_POS_LEFT = np.array([0.0, 0.18, 0.45])
SHOULDER_POS_RIGHT = np.array([0.0, -0.18, 0.45])

TCP_FRAMES = {
    'left': 'openarm_left_hand_tcp',
    'right': 'openarm_right_hand_tcp'
}

class NaturalTeleopServer(Node):
    def __init__(self):
        super().__init__('natural_teleop_server')
        self.server = InteractiveMarkerServer(self, 'openarm_teleop')
        
        # Publicadores de objetivos para el controlador
        self.pub_target_l = self.create_publisher(PoseStamped, '/openarm/left/target_pose', 10)
        self.pub_target_r = self.create_publisher(PoseStamped, '/openarm/right/target_pose', 10)

        # TF Listener para leer la posición actual exacta de los TCPs en 'world'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Esperar brevemente a que las TFs estén disponibles
        self.timer = self.create_timer(0.5, self._init_markers_once)
        self.initialized = False

    def _init_markers_once(self):
        if self.initialized:
            return
            
        pos_left = self._get_tcp_position('left', np.array([0.2, 0.25, 0.2]))
        pos_right = self._get_tcp_position('right', np.array([0.2, -0.25, 0.2]))

        self._create_arm_marker('left', SHOULDER_POS_LEFT, pos_left)
        self._create_arm_marker('right', SHOULDER_POS_RIGHT, pos_right)
        
        self.server.applyChanges()
        self.initialized = True
        self.timer.cancel()
        self.get_logger().info("Servidor de teleoperación listo. Marcadores alineados en los TCPs.")

    def _get_tcp_position(self, side, fallback_pos):
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
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = f"hand_sphere_{side}"
        int_marker.description = f"Mano {side.upper()}"
        int_marker.scale = 0.15

        int_marker.pose.position.x = float(initial_pos[0])
        int_marker.pose.position.y = float(initial_pos[1])
        int_marker.pose.position.z = float(initial_pos[2])

        # 1. Visual de la Esfera (Movimiento 3D libre)
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
        drag_control.always_visible = True
        drag_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        drag_control.markers.append(sphere_marker)
        int_marker.controls.append(drag_control)

        # 2. Anillo de rotación sobre eje Z local (Perpendicular al brazo)
        roll_control = InteractiveMarkerControl()
        roll_control.name = "wrist_roll"
        roll_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        roll_control.orientation_mode = InteractiveMarkerControl.INHERIT
        roll_control.orientation.w = 0.7071
        roll_control.orientation.x = 0.0
        roll_control.orientation.y = 0.7071
        roll_control.orientation.z = 0.0
        int_marker.controls.append(roll_control)

        self.server.insert(
            int_marker, 
            feedback_callback=lambda fb, s=side, sh=shoulder_pos: self._process_feedback(fb, s, sh)
        )

    def _process_feedback(self, feedback, side, shoulder_pos):
        raw_pos = np.array([
            feedback.pose.position.x,
            feedback.pose.position.y,
            feedback.pose.position.z
        ])

        # Proyección y límites antropomórficos
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

        # --- CONSTRUCCIÓN DEL MARCO BASE ANTROPOMÓRFICO ---
        z_dir = dir_vec
        up_vector = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_dir, up_vector)) > 0.95:
            up_vector = np.array([0.0, 1.0, 0.0])

        x_dir = np.cross(up_vector, z_dir)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)

        # Matriz base de apuntamiento del brazo
        R_auto = np.column_stack((x_dir, y_dir, z_dir))

        # --- EXTRACCIÓN ROBUSTA DE ROLL EN EL MARCO LOCAL ---
        R_marker = R.from_quat([
            feedback.pose.orientation.x,
            feedback.pose.orientation.y,
            feedback.pose.orientation.z,
            feedback.pose.orientation.w
        ]).as_matrix()

        # Matriz de rotación relativa respecto al apuntamiento del brazo
        R_rel = R_auto.T @ R_marker

        # Extraer única y exclusivamente la rotación sobre el eje Z local del anillo
        user_roll = np.arctan2(R_rel[1, 0], R_rel[0, 0])

        # Reconstruir orientación final limpia
        R_roll = R.from_euler('z', user_roll).as_matrix()
        R_final = R_auto @ R_roll
        q_final = R.from_matrix(R_final).as_quat()

        # Actualizar pose en RViz
        feedback.pose.position.x = float(valid_pos[0])
        feedback.pose.position.y = float(valid_pos[1])
        feedback.pose.position.z = float(valid_pos[2])
        feedback.pose.orientation.x = float(q_final[0])
        feedback.pose.orientation.y = float(q_final[1])
        feedback.pose.orientation.z = float(q_final[2])
        feedback.pose.orientation.w = float(q_final[3])

        self.server.setPose(feedback.marker_name, feedback.pose)
        self.server.applyChanges()

        # Publicar objetivo PoseStamped
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

def main():
    rclpy.init()
    node = NaturalTeleopServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
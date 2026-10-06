import os
import numpy as np
import mujoco
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R_scipy, Slerp

from openarm_control.impedance_controller import PinocchioImpedanceController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

MOTOR_LIMITS = {
    'DM8009': 40.0,
    'DM4340': 27.0,
    'DM4310': 7.0
}

ARM_MOTOR_TYPES = ['DM8009', 'DM8009', 'DM4340', 'DM4340', 'DM4310', 'DM4310', 'DM4310']
TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])

# Límites articulares de posición (en radianes) extraídos del URDF/XML
JOINT_LIMITS_LEFT = [
    (-3.490659, 1.396263),   # Joint 1
    (-3.316125, 0.174533),   # Joint 2
    (-1.570796, 1.570796),   # Joint 3
    (0.0, 2.443461),         # Joint 4
    (-1.570796, 1.570796),   # Joint 5
    (-0.785398, 0.785398),   # Joint 6
    (-1.570796, 1.570796)    # Joint 7
]

JOINT_LIMITS_RIGHT = [
    (-1.396263, 3.490659),   # Joint 1
    (-0.174533, 3.316125),   # Joint 2
    (-1.570796, 1.570796),   # Joint 3
    (0.0, 2.443461),         # Joint 4
    (-1.570796, 1.570796),   # Joint 5
    (-0.785398, 0.785398),   # Joint 6
    (-1.570796, 1.570796)    # Joint 7
]

class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__('rviz_bridge_node')
        self.left_pose = None
        self.right_pose = None

        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_curr_l = self.create_publisher(PoseStamped, '/openarm/left/current_pose', 10)
        self.pub_curr_r = self.create_publisher(PoseStamped, '/openarm/right/current_pose', 10)

        # Topics para iti monitoring ti torques
        self.pub_torques_l = self.create_publisher(Float64MultiArray, '/openarm/left/joint_torques', 1)
        self.pub_torques_r = self.create_publisher(Float64MultiArray, '/openarm/right/joint_torques', 1)

        self.create_subscription(PoseStamped, '/openarm/left/target_pose', self._cb_left, 10)
        self.create_subscription(PoseStamped, '/openarm/right/target_pose', self._cb_right, 10)

    def _cb_left(self, msg):
        self.left_pose = msg.pose
        self.on_marker_pose_updated('left', msg.pose)

    def _cb_right(self, msg):
        self.right_pose = msg.pose
        self.on_marker_pose_updated('right', msg.pose)

    def on_marker_pose_updated(self, arm_side, pose):
        """Callback invocado inmediatamente cuando llega una nueva pose de marker/topic."""
        self.get_logger().debug(f"Nueva pose objetivo recibida para brazo {arm_side}: {pose.position}")

    def publish_torques(self, tau_l, tau_r):
        msg_l = Float64MultiArray()
        msg_l.data = [float(x) for x in tau_l]
        self.pub_torques_l.publish(msg_l)

        msg_r = Float64MultiArray()
        msg_r.data = [float(x) for x in tau_r]
        self.pub_torques_r.publish(msg_r)

    def publish_states(self, pos_l, R_l, pos_r, R_r, joint_names, q_vals):
        now = self.get_clock().now().to_msg()
        
        js = JointState()
        js.header.stamp = now
        js.name = joint_names
        js.position = [float(x) for x in q_vals]
        self.pub_js.publish(js)

        msg_l = PoseStamped()
        msg_l.header.stamp = now
        msg_l.header.frame_id = "world"
        msg_l.pose.position.x, msg_l.pose.position.y, msg_l.pose.position.z = pos_l
        q_l = R_scipy.from_matrix(R_l).as_quat()
        msg_l.pose.orientation.x, msg_l.pose.orientation.y, msg_l.pose.orientation.z, msg_l.pose.orientation.w = q_l
        self.pub_curr_l.publish(msg_l)

        msg_r = PoseStamped()
        msg_r.header.stamp = now
        msg_r.header.frame_id = "world"
        msg_r.pose.position.x, msg_r.pose.position.y, msg_r.pose.position.z = pos_r
        q_r = R_scipy.from_matrix(R_r).as_quat()
        msg_r.pose.orientation.x, msg_r.pose.orientation.y, msg_r.pose.orientation.z, msg_r.pose.orientation.w = q_r
        self.pub_curr_r.publish(msg_r)

def check_motor_overload(node, arm_name, torques):
    for i, tau in enumerate(torques):
        m_type = ARM_MOTOR_TYPES[i]
        limit = MOTOR_LIMITS[m_type]
        if abs(tau) >= limit * 0.92:
            node.get_logger().warn(
                f"OVERLOAD WARNING [{arm_name}] Joint {i+1} ({m_type}): "
                f"Torque {tau:.2f} Nm exceeds threshold! (Max: {limit} Nm)"
            )

def check_collisions(node, robot, reported_contacts=set()):
    """Detecta colisiones e imprime la línea de exclusión solo una vez por cada par de cuerpos."""
    for i in range(robot.data.ncon):
        contact = robot.data.contact[i]
        
        b1_id = robot.model.geom_bodyid[contact.geom1]
        b2_id = robot.model.geom_bodyid[contact.geom2]
        
        if b1_id != b2_id:
            b1_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b1_id) or f"body_{b1_id}"
            b2_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b2_id) or f"body_{b2_id}"
            
            # Ordenar el par para tratar (A, B) y (B, A) como la misma colisión
            pair = tuple(sorted([b1_name, b2_name]))
            
            if pair not in reported_contacts:
                node.get_logger().warn(
                    f"COLISIÓN DETECTADA: <exclude body1=\"{pair[0]}\" body2=\"{pair[1]}\" />"
                )
                reported_contacts.add(pair)

def main():
    pkg_path = get_package_share_directory('openarm_mujoco')
    xml_path = os.path.join(pkg_path, 'v1', 'openarm_bimanual_edited.xml')
    urdf_path = os.path.expanduser("~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1_edited.urdf")
    
    rclpy.init()
    node = RVizBridgeNode()
    robot = MujocoRobotInterface(xml_path)
    
    controller = PinocchioImpedanceController(
        urdf_path,
        left_ee_frame="openarm_left_hand_tcp",
        right_ee_frame="openarm_right_hand_tcp"
    )

    pos_init_l, R_init_l = None, None
    pos_init_r, R_init_r = None, None
    q_grip_left_d, q_grip_right_d = None, None
    q_nominal_left, q_nominal_right = None, None

    all_joint_names = [controller.model.names[i] for i in range(1, len(controller.model.joints))]

    # Factor de suavizado EMA (0.2 = respuesta instantánea sin temblor de ratón)
    alpha_pos = 0.25
    alpha_rot = 0.25

    # Poses filtradas que alimentan directamente al controlador
    pos_d_l, R_d_l = None, None
    pos_d_r, R_d_r = None, None
    
    try:
        while robot.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            robot.update_kinematics()

            q_pin = np.zeros(controller.model.nq)
            dq_pin = np.zeros(controller.model.nv)

            for i in range(1, len(controller.model.joints)):
                j_name = controller.model.names[i]
                j_id_mj = robot._get_joint_qpos_adr(j_name)
                if j_id_mj is not None:
                    q_pin[controller.model.joints[i].idx_q] = robot.data.qpos[j_id_mj]
                    dq_pin[controller.model.joints[i].idx_v] = robot.data.qvel[j_id_mj]

            pin.forwardKinematics(controller.model, controller.data, q_pin, dq_pin)
            pin.updateFramePlacements(controller.model, controller.data)

            # Inicialización en la pose actual de la mano
            if pos_d_l is None:
                H_l = controller.data.oMf[controller.left_ee_id]
                pos_d_l = H_l.translation.copy()
                R_d_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_d_r = H_r.translation.copy()
                R_d_r = H_r.rotation.copy()

            # Lectura del marcador con filtro EMA (Exponencial)
            if node.left_pose is not None:
                target_p = np.array([node.left_pose.position.x, node.left_pose.position.y, node.left_pose.position.z])
                q_l = node.left_pose.orientation
                
                # 1. Posición filtrada EMA
                pos_d_l = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_l

                # 2. Cuaternión actual y cuaternión objetivo [x, y, z, w]
                q_curr = R_scipy.from_matrix(R_d_l).as_quat()
                q_targ = np.array([q_l.x, q_l.y, q_l.z, q_l.w])

                # Invertir el cuaternión si el producto punto es negativo para tomar el camino más corto
                if np.dot(q_curr, q_targ) < 0.0:
                    q_targ = -q_targ

                rot_curr = R_scipy.from_quat(q_curr)
                rot_targ = R_scipy.from_quat(q_targ)

                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_l = slerp([alpha_rot])[0].as_matrix()

            if node.right_pose is not None:
                target_p = np.array([node.right_pose.position.x, node.right_pose.position.y, node.right_pose.position.z])
                q_r = node.right_pose.orientation
                
                # 1. Posición filtrada EMA
                pos_d_r = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_r

                # 2. Cuaternión actual y cuaternión objetivo [x, y, z, w]
                q_curr = R_scipy.from_matrix(R_d_r).as_quat()
                q_targ = np.array([q_r.x, q_r.y, q_r.z, q_r.w])

                # Invertir el cuaternión si el producto punto es negativo para tomar el camino más corto
                if np.dot(q_curr, q_targ) < 0.0:
                    q_targ = -q_targ

                rot_curr = R_scipy.from_quat(q_curr)
                rot_targ = R_scipy.from_quat(q_targ)

                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_r = slerp([alpha_rot])[0].as_matrix()

            # Espacio nulo adaptativo: q_nominal sigue a las articulaciones actuales
            # para no generar torques parásitos de retorno
            q_arm_l_curr = q_pin[controller.left_arm_v_ids]
            q_arm_r_curr = q_pin[controller.right_arm_v_ids]

            # Calcular torques completos (Impedancia + Espacio Nulo + Gravedad)
            tau_l = controller.compute_arm_torque(
                q_pin, dq_pin, controller.left_ee_id, controller.left_arm_v_ids,
                pos_d_l, np.zeros(3), R_d_l, q_nominal=q_arm_l_curr, is_right_arm=False, include_gravity=True
            )
            tau_r = controller.compute_arm_torque(
                q_pin, dq_pin, controller.right_ee_id, controller.right_arm_v_ids,
                pos_d_r, np.zeros(3), R_d_r, q_nominal=q_arm_r_curr, is_right_arm=True, include_gravity=True
            )

            # Saturar directamente por los límites físicos reales de cada motor
            tau_l_clamped = np.clip(tau_l, -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_r_clamped = np.clip(tau_r, -TORQUE_LIMITS, TORQUE_LIMITS)

            # 6. Asignar los torques finales a cmd_dict
            cmd_dict = {}
            for idx, act_id in enumerate(robot.left_act_ids):
                if act_id is not None:
                    cmd_dict[act_id] = tau_l_clamped[idx]

            for idx, act_id in enumerate(robot.right_act_ids):
                if act_id is not None:
                    cmd_dict[act_id] = tau_r_clamped[idx]

            # Enviar torques y dar un paso en la simulación
            robot.send_torques(cmd_dict)
            robot.step()

            # Monitorear colisiones y sobrecargas
            check_collisions(node, robot)
            check_motor_overload(node, "LEFT", tau_l_clamped)
            check_motor_overload(node, "RIGHT", tau_r_clamped)

            node.publish_torques(tau_l_clamped, tau_r_clamped)

            q_list = [float(q_pin[controller.model.joints[i].idx_q]) for i in range(1, len(controller.model.joints))]
            q_arm_vals = np.array(q_list, dtype=np.float64)
            node.publish_states(pos_d_l, R_d_l, pos_d_r, R_d_r, all_joint_names, q_arm_vals)
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
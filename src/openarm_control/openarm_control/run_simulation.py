import os
import numpy as np
import mujoco
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R_scipy

from openarm_control.impedance_controller import PinocchioImpedanceController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

MOTOR_LIMITS = {
    'DM8009': 40.0,
    'DM4340': 27.0,
    'DM4310': 7.0
}

ARM_MOTOR_TYPES = ['DM8009', 'DM8009', 'DM4340', 'DM4340', 'DM4310', 'DM4310', 'DM4310']

class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__('rviz_bridge_node')
        self.left_pose = None
        self.right_pose = None

        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_curr_l = self.create_publisher(PoseStamped, '/openarm/left/current_pose', 10)
        self.pub_curr_r = self.create_publisher(PoseStamped, '/openarm/right/current_pose', 10)

        # Topics para iti monitoring ti torques
        self.pub_torques_l = self.create_publisher(Float64MultiArray, '/openarm/left/joint_torques', 10)
        self.pub_torques_r = self.create_publisher(Float64MultiArray, '/openarm/right/joint_torques', 10)

        self.create_subscription(PoseStamped, '/openarm/left/target_pose', self._cb_left, 10)
        self.create_subscription(PoseStamped, '/openarm/right/target_pose', self._cb_right, 10)

    def _cb_left(self, msg): self.left_pose = msg.pose
    def _cb_right(self, msg): self.right_pose = msg.pose

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

    act_lf1 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger1_ctrl")
    act_lf2 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger2_ctrl")

    all_joint_names = [controller.model.names[i] for i in range(1, len(controller.model.joints))]
    
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

            if pos_init_l is None:
                H_l = controller.data.oMf[controller.left_ee_id]
                pos_init_l = H_l.translation.copy()
                R_init_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_init_r = H_r.translation.copy()
                R_init_r = H_r.rotation.copy()

            if q_nominal_left is None:
                q_nominal_left = robot.data.qpos[0:7].copy()
                q_nominal_right = robot.data.qpos[9:16].copy()

            if node.left_pose is not None:
                pos_init_l = np.array([node.left_pose.position.x, node.left_pose.position.y, node.left_pose.position.z])
                q = node.left_pose.orientation
                R_init_l = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            if node.right_pose is not None:
                pos_init_r = np.array([node.right_pose.position.x, node.right_pose.position.y, node.right_pose.position.z])
                q = node.right_pose.orientation
                R_init_r = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            tau_l = controller.compute_arm_torque(
                q_pin, dq_pin, controller.left_ee_id, controller.left_arm_v_ids,
                pos_init_l, np.zeros(3), R_init_l, q_nominal=q_nominal_left, is_right_arm=False
            )
            tau_r = controller.compute_arm_torque(
                q_pin, dq_pin, controller.right_ee_id, controller.right_arm_v_ids,
                pos_init_r, np.zeros(3), R_init_r, q_nominal=q_nominal_right, is_right_arm=True
            )
            
            tau_dyn = controller.compute_nonlinear_effects(q_pin, dq_pin)

            # Construir el diccionario de torques para MuJoCo
            cmd_dict = {}

            # Límites físicos por articulación (Brazo de 7 DOF)
            TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])

            # Dentro del bucle while, al calcular tau_l y tau_r:
            tau_l_total = tau_l + tau_dyn[controller.left_arm_v_ids]
            tau_r_total = tau_r + tau_dyn[controller.right_arm_v_ids]

            # 1. Saturar torques al máximo físico real
            tau_l_clamped = np.clip(tau_l_total, -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_r_clamped = np.clip(tau_r_total, -TORQUE_LIMITS, TORQUE_LIMITS)

            # 2. Rampa de arranque suave (Aumenta progresivamente de 0 a 1 en 1 segundo)
            sim_time = robot.get_time()
            ramp_factor = min(1.0, sim_time / 1.0)

            tau_l_clamped *= ramp_factor
            tau_r_clamped *= ramp_factor

            # 3. Asignar los torques finales a cmd_dict
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

            q_list = []
            for i in range(1, len(controller.model.joints)):
                idx_q = controller.model.joints[i].idx_q
                q_list.append(float(q_pin[idx_q]))

            q_arm_vals = np.array(q_list, dtype=np.float64)
            node.publish_states(pos_init_l, R_init_l, pos_init_r, R_init_r, all_joint_names, q_arm_vals)
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
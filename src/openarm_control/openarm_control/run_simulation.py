import os
import numpy as np
import mujoco
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R_scipy

from openarm_control.impedance_controller import PinocchioImpedanceController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__('rviz_bridge_node')
        self.left_pose = None
        self.right_pose = None

        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_curr_l = self.create_publisher(PoseStamped, '/openarm/left/current_pose', 10)
        self.pub_curr_r = self.create_publisher(PoseStamped, '/openarm/right/current_pose', 10)

        self.create_subscription(PoseStamped, '/openarm/left/target_pose', self._cb_left, 10)
        self.create_subscription(PoseStamped, '/openarm/right/target_pose', self._cb_right, 10)

    def _cb_left(self, msg): self.left_pose = msg.pose
    def _cb_right(self, msg): self.right_pose = msg.pose

    def publish_states(self, pos_l, R_l, pos_r, R_r, joint_names, q_vals):
        now = self.get_clock().now().to_msg()
        
        # Publicar TF/JointState para RViz
        js = JointState()
        js.header.stamp = now
        js.name = joint_names
        js.position = q_vals.tolist()
        self.pub_js.publish(js)

        # Publicar Poses de las garras
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

def main():
    pkg_path = get_package_share_directory('openarm_mujoco')
    xml_path = os.path.join(pkg_path, 'v1', 'openarm_bimanual.xml')
    urdf_path = os.path.expanduser("~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf")

    rclpy.init()
    node = RVizBridgeNode()
    robot = MujocoRobotInterface(xml_path)
    controller = PinocchioImpedanceController(urdf_path)

    pos_init_l, R_init_l = None, None
    pos_init_r, R_init_r = None, None
    q_grip_left_d, q_grip_right_d = None, None
    q_nominal_left, q_nominal_right = None, None
    act_lf1 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger1_ctrl")
    act_lf2 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger2_ctrl")
    act_rf1 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl")
    act_rf2 = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger2_ctrl")

    all_joint_names = (
        [f"openarm_left_joint{i}" for i in range(1, 8)] +
        ["openarm_left_finger_joint1", "openarm_left_finger_joint2"] +
        [f"openarm_right_joint{i}" for i in range(1, 8)] +
        ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
    )
    
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

            # Captura de la pose inicial cartesiana
            if pos_init_l is None:
                H_l = controller.data.oMf[controller.left_ee_id]
                pos_init_l = H_l.translation.copy() + H_l.rotation.copy() @ controller.tcp_offset
                R_init_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_init_r = H_r.translation.copy() + H_r.rotation.copy() @ controller.tcp_offset
                R_init_r = H_r.rotation.copy()

            # Captura fija de la pose de spawn de grippers y postura articular de espacio nulo
            if q_grip_left_d is None:
                q_grip_left_d = robot.data.qpos[7:9].copy()
                q_grip_right_d = robot.data.qpos[16:18].copy()

                q_nominal_left = robot.data.qpos[0:7].copy()
                q_nominal_right = robot.data.qpos[9:16].copy()

            # Actualización desde ROS solo si hay un comando explícito
            if node.left_pose is not None:
                pos_init_l = np.array([node.left_pose.position.x, node.left_pose.position.y, node.left_pose.position.z])
                q = node.left_pose.orientation
                R_init_l = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            if node.right_pose is not None:
                pos_init_r = np.array([node.right_pose.position.x, node.right_pose.position.y, node.right_pose.position.z])
                q = node.right_pose.orientation
                R_init_r = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            # 1. Torques de Impedancia Cartesiana + Control de Espacio Nulo
            tau_l = controller.compute_arm_torque(
                q_pin, dq_pin, controller.left_ee_id, controller.left_arm_v_ids,
                pos_init_l, np.zeros(3), R_init_l, q_nominal=q_nominal_left
            )
            tau_r = controller.compute_arm_torque(
                q_pin, dq_pin, controller.right_ee_id, controller.right_arm_v_ids,
                pos_init_r, np.zeros(3), R_init_r, q_nominal=q_nominal_right
            )
            
            # 2. Compensación Dinámica (Gravedad + Coriolis)
            tau_dyn = controller.compute_nonlinear_effects(q_pin, dq_pin)

            # 3. Compensación de Fricción desactivada
            tau_fric_l = np.zeros(7)
            tau_fric_r = np.zeros(7)

            cmd_dict = {}

            for idx, act_id in enumerate(robot.left_act_ids):
                if act_id is not None:
                    v_id = controller.left_arm_v_ids[idx]
                    cmd_dict[act_id] = tau_l[idx] + tau_dyn[v_id] + tau_fric_l[idx]

            for idx, act_id in enumerate(robot.right_act_ids):
                if act_id is not None:
                    v_id = controller.right_arm_v_ids[idx]
                    cmd_dict[act_id] = tau_r[idx] + tau_dyn[v_id] + tau_fric_r[idx]

            # 5. Asignar torques calculados de los grippers
            if q_grip_left_d is not None:
                q_curr_l = robot.data.qpos[7:9]
                dq_curr_l = robot.data.qvel[7:9]
                tau_grip_l = controller.compute_gripper_torque(q_curr_l, dq_curr_l, q_grip_left_d)

                q_curr_r = robot.data.qpos[16:18]
                dq_curr_r = robot.data.qvel[16:18]
                tau_grip_r = controller.compute_gripper_torque(q_curr_r, dq_curr_r, q_grip_right_d)

                cmd_dict[act_lf1] = tau_grip_l[0]
                cmd_dict[act_lf2] = tau_grip_l[1]
                cmd_dict[act_rf1] = tau_grip_r[0]
                cmd_dict[act_rf2] = tau_grip_r[1]

            robot.send_torques(cmd_dict)
            
            q_arm_vals = np.array([robot.data.qpos[robot._get_joint_qpos_adr(name)] for name in all_joint_names])
            node.publish_states(pos_init_l, R_init_l, pos_init_r, R_init_r, all_joint_names, q_arm_vals)
            
            robot.step()
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
import os
import numpy as np
import mujoco
import pinocchio as pin
import rclpy
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
from scipy.spatial.transform import Rotation as R_scipy, Slerp

from openarm_control.impedance_controller import PinocchioImpedanceController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

G_ACC           = 9.81  # m/s^2

MOTOR_LIMITS = {
    'DM8009': 40.0,
    'DM4340': 27.0,
    'DM4310': 7.0
}

ARM_MOTOR_TYPES = ['DM8009', 'DM8009', 'DM4340', 'DM4340', 'DM4310', 'DM4310', 'DM4310']
TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])

# Map MuJoCo joint names to URDF joint names
MJ_TO_URDF_JOINTS = {
    'right_wrist': 'R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e',
    'right_p-abd':  'P-AP_f5e42b61_to_R-Carpals_8d1f1041',
    'right_p-mcp':  'P-PP_1d411b9b_to_P-AP_f5e42b61',
    'right_p-pip':  'P-FingerTipAssembly_cd219176_to_P-PP_1d411b9b',
    'right_r-abd':  'M-AP_6ec59111_to_R-Carpals_8d1f1041',
    'right_r-mcp':  'M-PP_8660a1eb_to_M-AP_6ec59111',
    'right_r-pip':  'M-FingerTipAssembly_424a8e75_to_M-PP_8660a1eb',
    'right_m-abd':  'M-AP_e04a96f2_to_R-Carpals_8d1f1041',
    'right_m-mcp':  'M-PP_08efa608_to_M-AP_e04a96f2',
    'right_m-pip':  'M-FingerTipAssembly_34afb748_to_M-PP_08efa608',
    'right_i-abd':  'I-AP-R_d95d02d1_to_R-Carpals_8d1f1041',
    'right_i-mcp':  'I-PP_bacbd481_to_I-AP-R_d95d02d1',
    'right_i-pip':  'I-FingerTipAssembly_ec49c16c_to_I-PP_bacbd481',
    'right_t-cmc':  'T-TP-R_1c2b802d_to_R-Carpals_8d1f1041',
    'right_t-abd':  'R-T-AP_a9723101_to_T-TP-R_1c2b802d',
    'right_t-mcp':  'T-PP_68395e98_to_R-T-AP_a9723101',
    'right_t-pip':  'T-DP_b7429e50_to_T-PP_68395e98',
}

# Inverse mapping URDF -> MuJoCo
URDF_TO_MJ_JOINTS = {v: k for k, v in MJ_TO_URDF_JOINTS.items()}

def get_tip_direct_contact_force(model, data, clarius_body_id, tip_site_id):
    # Measure the collision forces on the transducer
    
    # Default return if it does not find the transducer or its tip
    if clarius_body_id == -1 or tip_site_id == -1:
        return 0.0

    # Get the z-axis of the transducer
    R_tip_world = data.site_xmat[tip_site_id].reshape(3, 3)
    n_tip_world = R_tip_world[:, 2] 

    # Get the force in the z-axis
    f_tip_ext_world = np.zeros(3)

    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]

        b1_is_tip = (b1 == clarius_body_id)
        b2_is_tip = (b2 == clarius_body_id)

        if b1_is_tip != b2_is_tip:
            c_wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_wrench)
            R_contact = contact.frame.reshape(3, 3)
            f_world = (1.0 if b1_is_tip else -1.0) * (R_contact @ c_wrench[:3])
            f_tip_ext_world += f_world

    return max(0.0, float(np.dot(f_tip_ext_world, n_tip_world)))

def get_subtree_body_ids(model, root_body_id):
    # Get every body ID which belongs to a root body
    if root_body_id == -1:
        return set()
    subtree = {root_body_id}
    for b_id in range(root_body_id + 1, model.nbody):
        if model.body_parentid[b_id] in subtree:
            subtree.add(b_id)
    return subtree


def get_subtree_contact_wrench_world(model, data, subtree_body_ids, wrist_pos_world):
    # Get the collision forces and torques actuating on the subtree
    
    f_ext_world = np.zeros(3)
    tau_ext_world = np.zeros(3)

    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        b1_in = b1 in subtree_body_ids
        b2_in = b2 in subtree_body_ids

        if b1_in != b2_in:
            c_wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_wrench)
            
            R_contact = contact.frame.reshape(3, 3)
            f_world = (1.0 if b1_in else -1.0) * (R_contact @ c_wrench[:3])
            tau_world = (1.0 if b1_in else -1.0) * (R_contact @ c_wrench[3:])

            r_contact_wrist = contact.pos - wrist_pos_world
            
            f_ext_world += f_world
            tau_ext_world += tau_world + np.cross(r_contact_wrist, f_world)

    return f_ext_world, tau_ext_world

def read_sensone_raw_wrench(model, data, ee_adapter_id, wrist_site_id, subtree_body_ids):
    # Simulate measure made by the sensor
    R_w_world = data.site_xmat[wrist_site_id].reshape(3, 3)
    p_w_world = data.site_xpos[wrist_site_id]

    f_ext_world, tau_ext_world = get_subtree_contact_wrench_world(
        model, data, subtree_body_ids, p_w_world
    )

    f_ext_w = R_w_world.T @ f_ext_world
    tau_ext_w = R_w_world.T @ tau_ext_world

    m_sub = float(model.body_subtreemass[ee_adapter_id])
    p_com_world = data.subtree_com[ee_adapter_id]

    g_world = np.array([0.0, 0.0, -G_ACC])
    f_grav_w = m_sub * (R_w_world.T @ g_world)

    r_com_w = R_w_world.T @ (p_com_world - p_w_world)
    tau_grav_w = np.cross(r_com_w, f_grav_w)

    f_sensone_raw_w = f_ext_w + f_grav_w
    tau_sensone_raw_w = tau_ext_w + tau_grav_w

    return f_sensone_raw_w, tau_sensone_raw_w

class RVizBridgeNode(Node):
    def __init__(self):
        # Setup
        super().__init__('rviz_bridge_node')
        self.left_pose = None
        self.right_pose = None
        self.target_wrist_angle = 0.0
        self.target_gripper_pos = 0.035

        # Publishers of joint states and arms poses
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_curr_l = self.create_publisher(PoseStamped, '/openarm/left/current_pose', 10)
        self.pub_curr_r = self.create_publisher(PoseStamped, '/openarm/right/current_pose', 10)

        # Publishers of joint torques
        self.pub_torques_l = self.create_publisher(Float64MultiArray, '/openarm/left/joint_torques', 1)
        self.pub_torques_r = self.create_publisher(Float64MultiArray, '/openarm/right/joint_torques', 1)

        # Publishers of forces in the transducer tip
        self.pub_force_tip = self.create_publisher(Float64, '/openarm/right/measured_force_tip_z', 10)
        self.pub_force_calc = self.create_publisher(Float64, '/openarm/right/calculated_force_tip_z', 10)

        # Subscriptions to targets
        self.create_subscription(PoseStamped, '/openarm/left/target_pose', self._cb_left, 10)
        self.create_subscription(PoseStamped, '/openarm/right/target_pose', self._cb_right, 10)
        self.create_subscription(Float64, '/openarm/right/wrist_cmd', self._cb_wrist, 10)
        self.create_subscription(Float64, '/openarm/left/gripper_cmd', self._cb_gripper, 10)

    def _cb_left(self, msg):
        # Get left arm pose
        self.left_pose = msg.pose
        self.on_marker_pose_updated('left', msg.pose)

    def _cb_right(self, msg):
        # Get right arm pose
        self.right_pose = msg.pose
        self.on_marker_pose_updated('right', msg.pose)

    def _cb_wrist(self, msg):
        # Get wrist angle
        self.target_wrist_angle = float(msg.data)

    def _cb_gripper(self, msg):
        # Get gripper command
        val = float(np.clip(msg.data, 0.0, 0.044))
        self.target_gripper_pos = val

    def on_marker_pose_updated(self, arm_side, pose):
        # Notify update of the target pose
        self.get_logger().debug(f"New target pose received for the {arm_side} arm: {pose.position}")

    def publish_torques(self, tau_l, tau_r):
        # Send torques for both arms
        msg_l = Float64MultiArray()
        msg_l.data = [float(x) for x in tau_l]
        self.pub_torques_l.publish(msg_l)

        msg_r = Float64MultiArray()
        msg_r.data = [float(x) for x in tau_r]
        self.pub_torques_r.publish(msg_r)

    def publish_forces(self, f_tip_direct, f_tip_calc):
        # Send force on the transducer tip
        msg_tip = Float64()
        msg_tip.data = float(f_tip_direct)
        self.pub_force_tip.publish(msg_tip)

        msg_calc = Float64()
        msg_calc.data = float(f_tip_calc)
        self.pub_force_calc.publish(msg_calc)

    def publish_states(self, pos_l, R_l, pos_r, R_r, joint_names, q_vals):
        # Publish joint states
    
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
    # Check if the applied torques reach the motor limits
    for i, tau in enumerate(torques):
        m_type = ARM_MOTOR_TYPES[i]
        limit = MOTOR_LIMITS[m_type]
        if abs(tau) >= limit:
            node.get_logger().warn(
                f"OVERLOAD WARNING [{arm_name}] Joint {i+1} ({m_type}): "
                f"Torque {tau:.2f} Nm exceeds threshold! (Max: {limit} Nm)"
            )

def check_collisions(node, robot, reported_contacts=set()):
    # Detect collisions of bodies
    for i in range(robot.data.ncon):
        contact = robot.data.contact[i]
        
        b1_id = robot.model.geom_bodyid[contact.geom1]
        b2_id = robot.model.geom_bodyid[contact.geom2]
        
        if b1_id != b2_id:
            b1_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b1_id) or f"body_{b1_id}"
            b2_name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, b2_id) or f"body_{b2_id}"
            
            pair = tuple(sorted([b1_name, b2_name]))
            
            if pair not in reported_contacts:
                node.get_logger().warn(
                    f"DETECTED COLLISION: <exclude body1=\"{pair[0]}\" body2=\"{pair[1]}\" />"
                )
                reported_contacts.add(pair)

def main():
    # Get file paths
    pkg_path = get_package_share_directory('openarm_mujoco')
    xml_path = os.path.join(pkg_path, 'v1', 'openarm_bimanual_edited.xml')
    urdf_path = os.path.expanduser("~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1_edited.urdf")
    
    # Setup
    rclpy.init()
    node = RVizBridgeNode()
    robot = MujocoRobotInterface(xml_path)
    controller = PinocchioImpedanceController(
        urdf_path,
        left_ee_frame="openarm_left_hand_tcp",
        right_ee_frame="openarm_right_hand_tcp"
    )
    
    # Get IDs
    wrist_sensor_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SENSOR, "contact_force_ee")
    wrist_site_id   = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "force_measure_point")

    tip_sensor_id   = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SENSOR, "contact_force_tip")
    tip_site_id     = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "ultrasound_tip")

    clarius_body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "clarius_c3_hd3")
    ee_adapter_id   = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_adapter")
    subtree_body_ids = get_subtree_body_ids(robot.model, ee_adapter_id)

    # Get the EE subtree mass
    M_SUBTREE_SIM = float(robot.model.body_subtreemass[ee_adapter_id]) if ee_adapter_id != -1 else 2.137838

    # Initial values
    pos_init_l, R_init_l = None, None
    pos_init_r, R_init_r = None, None
    q_grip_left_d, q_grip_right_d = None, None
    q_nominal_left, q_nominal_right = None, None

    # Fixed finger poses
    fixed_fingers = {
        'right_r-mcp': 1.72,
        'right_r-pip': 1.7,
        'right_p-mcp': 1.72,
        'right_p-pip': 1.5,
        'right_m-mcp': 1.72,
        'right_m-pip': 1.7,
        'right_i-mcp': 1.4,
        'right_i-pip': 1.3,
        'right_t-cmc': -0.78,
        'right_t-abd': 0.95,
        'right_t-mcp': 0.1,
        'right_t-pip': 0.5
    }
    
    # Initialize the gripper as open
    j_adr_g1 = robot._get_joint_qpos_adr('openarm_left_finger_joint1')
    j_adr_g2 = robot._get_joint_qpos_adr('openarm_left_finger_joint2')
    if j_adr_g1 is not None: robot.data.qpos[j_adr_g1] = 0.035
    if j_adr_g2 is not None: robot.data.qpos[j_adr_g2] = 0.035

    # Initialize the wrist pose
    j_adr_w = robot._get_joint_qpos_adr('right_wrist')
    if j_adr_w is not None:
        robot.data.qpos[j_adr_w] = 0.0
    for j_name, angle in fixed_fingers.items():
        j_adr = robot._get_joint_qpos_adr(j_name)
        if j_adr is not None:
            robot.data.qpos[j_adr] = angle

    all_joint_names = [controller.model.names[i] for i in range(1, len(controller.model.joints))]

    # EMA Smoothness factor
    alpha_pos = 0.25
    alpha_rot = 0.25

    # Initial target poses
    pos_d_l, R_d_l = None, None
    pos_d_r, R_d_r = None, None

    # Update kinematics
    mujoco.mj_forward(robot.model, robot.data)
    
    try:
        while robot.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            robot.update_kinematics()

            # Wrist controller
            
            j_adr_w, v_adr_w = robot._get_joint_adrs('right_wrist')
            q_w_curr = robot.data.qpos[j_adr_w] if j_adr_w is not None else 0.0
            dq_w_curr = robot.data.qvel[v_adr_w] if v_adr_w is not None else 0.0
            
            wrist_cmd = controller.compute_wrist_cmd(
                q_w_curr, dq_w_curr, node.target_wrist_angle, is_position_actuator=True
            )
            
            wrist_act_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_actuator")
            if wrist_act_id != -1:
                robot.data.ctrl[wrist_act_id] = wrist_cmd

            # Fingers control

            for j_name, angle in fixed_fingers.items():
                actuator_name = j_name + "_actuator"
                act_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if act_id != -1:
                    robot.data.ctrl[act_id] = angle

            q_pin = np.zeros(controller.model.nq)
            dq_pin = np.zeros(controller.model.nv)

            for i in range(1, len(controller.model.joints)):
                urdf_jname = controller.model.names[i]
                mj_jname = URDF_TO_MJ_JOINTS.get(urdf_jname, urdf_jname)
                
                j_adr = robot._get_joint_qpos_adr(mj_jname)
                if j_adr is not None:
                    q_pin[controller.model.joints[i].idx_q] = robot.data.qpos[j_adr]
                    dq_pin[controller.model.joints[i].idx_v] = robot.data.qvel[j_adr]

            pin.forwardKinematics(controller.model, controller.data, q_pin, dq_pin)
            pin.updateFramePlacements(controller.model, controller.data)

            # Initial current hand poses

            if pos_d_l is None:
                H_l = controller.data.oMf[controller.left_ee_id]
                pos_d_l = H_l.translation.copy()
                R_d_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_d_r = H_r.translation.copy()
                R_d_r = H_r.rotation.copy()

            # Marker lecture with the EMA filter
            if node.left_pose is not None:
                target_p = np.array([node.left_pose.position.x, node.left_pose.position.y, node.left_pose.position.z])
                q_l = node.left_pose.orientation
                
                # EMA filtered position
                pos_d_l = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_l

                # Current and target quaternions
                q_curr = R_scipy.from_matrix(R_d_l).as_quat()
                q_targ = np.array([q_l.x, q_l.y, q_l.z, q_l.w])

                # Change the quaternion to take the shortest path if necessary
                if np.dot(q_curr, q_targ) < 0.0:
                    q_targ = -q_targ

                rot_curr = R_scipy.from_quat(q_curr)
                rot_targ = R_scipy.from_quat(q_targ)

                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_l = slerp([alpha_rot])[0].as_matrix()

            if node.right_pose is not None:
                target_p = np.array([node.right_pose.position.x, node.right_pose.position.y, node.right_pose.position.z])
                q_r = node.right_pose.orientation
                
                # EMA filtered position
                pos_d_r = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_r

                # Current and target quaternions
                q_curr = R_scipy.from_matrix(R_d_r).as_quat()
                q_targ = np.array([q_r.x, q_r.y, q_r.z, q_r.w])

                # Change the quaternion to take the shortest path if necessary
                if np.dot(q_curr, q_targ) < 0.0:
                    q_targ = -q_targ

                rot_curr = R_scipy.from_quat(q_curr)
                rot_targ = R_scipy.from_quat(q_targ)

                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_r = slerp([alpha_rot])[0].as_matrix()

            # Current joints
            q_arm_l_curr = q_pin[controller.left_arm_v_ids]
            q_arm_r_curr = q_pin[controller.right_arm_v_ids]

            # Calculate full torques
            tau_l = controller.compute_arm_torque(
                q_pin, dq_pin, controller.left_ee_id, controller.left_arm_v_ids,
                pos_d_l, np.zeros(3), R_d_l, q_nominal=q_arm_l_curr, is_right_arm=False, include_gravity=True
            )
            tau_r = controller.compute_arm_torque(
                q_pin, dq_pin, controller.right_ee_id, controller.right_arm_v_ids,
                pos_d_r, np.zeros(3), R_d_r, q_nominal=q_arm_r_curr, is_right_arm=True, include_gravity=True
            )

            # Clamp torques
            tau_l_clamped = np.clip(tau_l, -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_r_clamped = np.clip(tau_r, -TORQUE_LIMITS, TORQUE_LIMITS)

            # Assigned torques
            cmd_dict = {}
            for idx, act_id in enumerate(robot.left_act_ids):
                if act_id is not None:
                    cmd_dict[act_id] = tau_l_clamped[idx]

            for idx, act_id in enumerate(robot.right_act_ids):
                if act_id is not None:
                    cmd_dict[act_id] = tau_r_clamped[idx]

            # Control fingers in each step
            for j_name, angle in fixed_fingers.items():
                actuator_name = j_name + "_actuator"
                act_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                
                if act_id != -1:
                    robot.data.ctrl[act_id] = angle

            # Gripper control
            j_adr_g1, v_adr_g1 = robot._get_joint_adrs('openarm_left_finger_joint1')
            q_g_curr = robot.data.qpos[j_adr_g1] if j_adr_g1 is not None else 0.035
            dq_g_curr = robot.data.qvel[v_adr_g1] if v_adr_g1 is not None else 0.0

            tau_gripper = controller.compute_gripper_torque(q_g_curr, dq_g_curr, node.target_gripper_pos)

            if robot.left_finger_act_id is not None:
                cmd_dict[robot.left_finger_act_id] = tau_gripper
            if robot.left_finger2_act_id is not None:
                cmd_dict[robot.left_finger2_act_id] = tau_gripper

            # Send torques and take a step in the simulation
            robot.send_torques(cmd_dict)
            robot.step()

            # Force in the sensor and the transducer tip

            # Initial values
            f_norm_direct, f_norm_calc = 0.0, 0.0

            # Ground truth value
            f_norm_direct = get_tip_direct_contact_force(
                robot.model, robot.data, clarius_body_id, tip_site_id
            )

            # Simulation of sensor and transformation to the tip
            if ee_adapter_id != -1 and wrist_site_id != -1 and tip_site_id != -1:
                # Raw measure
                f_raw_w, tau_raw_w = read_sensone_raw_wrench(
                    robot.model, robot.data, ee_adapter_id, wrist_site_id, subtree_body_ids
                )

                # Gravity compensation and transformation

                R_w_world = robot.data.site_xmat[wrist_site_id].reshape(3, 3)
                p_w_world = robot.data.site_xpos[wrist_site_id]
                R_t_world = robot.data.site_xmat[tip_site_id].reshape(3, 3)
                p_t_world = robot.data.site_xpos[tip_site_id]

                m_sub = float(robot.model.body_subtreemass[ee_adapter_id])
                p_com_world = robot.data.subtree_com[ee_adapter_id]
                r_com_w = R_w_world.T @ (p_com_world - p_w_world)

                g_w = R_w_world.T @ np.array([0.0, 0.0, -G_ACC])
                f_grav_w = m_sub * g_w
                tau_grav_w = np.cross(r_com_w, f_grav_w)

                # Substract the gravity effect
                f_ext_w = f_raw_w - f_grav_w
                tau_ext_w = tau_raw_w - tau_grav_w

                # Translation to the tip
                R_t_w = R_t_world.T @ R_w_world
                r_tip_w = R_w_world.T @ (p_t_world - p_w_world)

                f_ext_t = R_t_w @ f_ext_w
                tau_ext_t = R_t_w @ (tau_ext_w - np.cross(r_tip_w, f_ext_w))

                # Z-axis projection
                n_tip_local = np.array([0.0, 0.0, 1.0])
                f_norm_calc = max(0.0, float(np.dot(f_ext_t, n_tip_local)))

            # Publish the forces
            node.publish_forces(f_norm_direct, f_norm_calc)
            node.get_logger().info(
                f"[TELEOP CONTACT] Ground Truth: {f_norm_direct:.2f} N | Sensor: {f_norm_calc:.2f} N | Error: {abs(f_norm_calc - f_norm_direct):.3f} N",
                throttle_duration_sec=0.2
            )

            # Check problems
            check_collisions(node, robot)
            check_motor_overload(node, "LEFT", tau_l_clamped)
            check_motor_overload(node, "RIGHT", tau_r_clamped)

            # Publish torques
            node.publish_torques(tau_l_clamped, tau_r_clamped)

            # Publish states to RViz
            q_list = [float(q_pin[controller.model.joints[i].idx_q]) for i in range(1, len(controller.model.joints))]
            node.publish_states(pos_d_l, R_d_l, pos_d_r, R_d_r, all_joint_names, np.array(q_list, dtype=np.float64))

    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
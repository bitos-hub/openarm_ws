import os
import time
import signal
import multiprocessing
import numpy as np
import mujoco
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, Bool
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt

from openarm_control.impedance_controller import BimanualCartesianController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

# Gravity
G_ACC = 9.81

# Torque limits
TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])

# Mapping of MuJoCo to URDF joints
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

# Mapping of URDF to MuJoCo joints
URDF_TO_MJ_JOINTS = {v: k for k, v in MJ_TO_URDF_JOINTS.items()}

def plot_force_exerted(t_data, f_target_data, f_calc_data, f_direct_data):
    """
    Plot the force exerted vs time.
    """
    # Restore default signal handling in the child process
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    #Plot configuration
    plt.figure(figsize=(10, 5))
    plt.plot(t_data, f_target_data, 'r--', linewidth=2.0, label='Desired Force $F_d$ [N]')
    plt.plot(t_data, f_calc_data, 'b-', linewidth=1.8, label='Calculated Force (Wrist Sensor) [N]')
    plt.plot(t_data, f_direct_data, 'g:', linewidth=1.2, alpha=0.7, label='Real Force in the Tip (Ground Truth) [N]')
    plt.xlabel('Accumulated time in contact [s]', fontsize=11)
    plt.ylabel('Normal Force $F_y$ [N]', fontsize=11)
    plt.title('Temporal Admittance Controller Response in Contact', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()

    img_filename = "force_plot.png"
    plt.savefig(img_filename, dpi=300)
    print(f"\n[PLOTTING] Plot saved as: '{img_filename}'")
    
    try:
        plt.show()
    except Exception as e:
        print(f"[PLOTTING] Error when opening the window: {e}")

def compute_tcp_rotation_from_normal(normal_world, R_user=None):
    """
    Compute the TCP rotation matrix to align the Y-axis with the normal
    vector of a surface.
    """
    # Asign the unit normal vector as the Y-axis
    n = normal_world / np.linalg.norm(normal_world)
    y_tcp = n
    
    # Get the user reference for the X-axis
    if R_user is None:
        R_user = np.eye(3)     
    x_user = R_user[:, 0]

    # Make the X_TCP perpendicular to Y_TCP
    x_tcp = x_user - np.dot(x_user, y_tcp) * y_tcp
    
    # Singularity protection
    if np.linalg.norm(x_tcp) < 1e-4:
        x_user_alt = np.array([0.0, 0.0, 1.0]) if abs(y_tcp[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        x_tcp = x_user_alt - np.dot(x_user_alt, y_tcp) * y_tcp
        
    # Normalize the X-axis and compute the Z-axis
    x_tcp /= np.linalg.norm(x_tcp)
    z_tcp = np.cross(x_tcp, y_tcp)
    z_tcp /= np.linalg.norm(z_tcp)
    
    return np.column_stack((x_tcp, y_tcp, z_tcp))

def get_abdomen_normal_and_contact(robot):
    """
    Detect the contact of the transducer with the phantom and get the
    normal vector to the surface.
    """
    # Transducer and phantom IDs
    transducer_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "transducer_head")
    abdomen_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "abdomen_geom")
    
    # Initial values
    in_contact = False
    normal_world = np.array([0.0, 0.0, 1.0])

    # Search for collisions
    for i in range(robot.data.ncon):
        c = robot.data.contact[i]
        if (c.geom1 == transducer_geom_id and c.geom2 == abdomen_geom_id) or \
           (c.geom2 == transducer_geom_id and c.geom1 == abdomen_geom_id):
            # Update values
            in_contact = True
            normal_world = c.frame[:3].copy()
            if c.geom1 == transducer_geom_id:
                normal_world = -normal_world
            break

    return in_contact, normal_world

def get_tip_direct_contact_force(model, data, clarius_body_id, tip_site_id):
    """
    Get the direct contact force measured on the transducer tip.
    """
    if clarius_body_id == -1 or tip_site_id == -1:
        return 0.0

    # Get the Z-axis in the world frame
    R_tip_world = data.site_xmat[tip_site_id].reshape(3, 3)
    n_tip_world = R_tip_world[:, 2] 

    # Initial value
    f_tip_ext_world = np.zeros(3)

    for i in range(data.ncon):
        # Identify the contact
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]

        b1_is_tip = (b1 == clarius_body_id)
        b2_is_tip = (b2 == clarius_body_id)

        if b1_is_tip != b2_is_tip:
            # Get the wrench in the world frame
            c_wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_wrench)
            R_contact = contact.frame.reshape(3, 3)
            f_world = (1.0 if b1_is_tip else -1.0) * (R_contact @ c_wrench[:3])
            f_tip_ext_world += f_world

    return max(0.0, float(np.dot(f_tip_ext_world, n_tip_world)))

def get_subtree_body_ids(model, root_body_id):
    """
    Get a subtree body IDs.
    """
    if root_body_id == -1:
        return set()
    subtree = {root_body_id}
    for b_id in range(root_body_id + 1, model.nbody):
        if model.body_parentid[b_id] in subtree:
            subtree.add(b_id)
    return subtree

def get_subtree_contact_wrench_world(model, data, subtree_body_ids, wrist_pos_world):
    """
    Compute the wrench actuating on a subtree and translate it to the wrist.
    """
    # Initial values
    f_ext_world = np.zeros(3)
    tau_ext_world = np.zeros(3)

    # Look for contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        b1_in = b1 in subtree_body_ids
        b2_in = b2 in subtree_body_ids

        if b1_in != b2_in:
            # Get the wrench in the world frame
            c_wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_wrench)
            R_contact = contact.frame.reshape(3, 3)
            f_world = (1.0 if b1_in else -1.0) * (R_contact @ c_wrench[:3])
            tau_world = (1.0 if b1_in else -1.0) * (R_contact @ c_wrench[3:])

            # Translate it to the wrist
            r_contact_wrist = contact.pos - wrist_pos_world
            f_ext_world += f_world
            tau_ext_world += tau_world + np.cross(r_contact_wrist, f_world)

    return f_ext_world, tau_ext_world

def read_sensone_raw_wrench(model, data, ee_adapter_id, wrist_site_id, subtree_body_ids):
    """
    Simulate the sensor raw wrench measurement.
    """
    # Get position and rotation
    R_w_world = data.site_xmat[wrist_site_id].reshape(3, 3)
    p_w_world = data.site_xpos[wrist_site_id]

    # Get wrench translated to the wrist
    f_ext_world, tau_ext_world = get_subtree_contact_wrench_world(
        model, data, subtree_body_ids, p_w_world
    )

    # Transform the wrench to the wrist frame
    f_ext_w = R_w_world.T @ f_ext_world
    tau_ext_w = R_w_world.T @ tau_ext_world

    # Include the gravity effect
    m_sub = float(model.body_subtreemass[ee_adapter_id])
    p_com_world = data.subtree_com[ee_adapter_id]
    g_world = np.array([0.0, 0.0, -G_ACC])
    f_grav_w = m_sub * (R_w_world.T @ g_world)
    r_com_w = R_w_world.T @ (p_com_world - p_w_world)
    tau_grav_w = np.cross(r_com_w, f_grav_w)

    # Get the simulated measurement
    f_sensone_raw_w = f_ext_w + f_grav_w
    tau_sensone_raw_w = tau_ext_w + tau_grav_w

    return f_sensone_raw_w, tau_sensone_raw_w

class RVizBridgeNode(Node):
    def __init__(self):
        super().__init__('rviz_bridge_node')
        # Initial values
        self.left_pose = None
        self.right_pose = None
        self.target_wrist_angle = 0.0
        self.target_gripper_pos = 0.035
        self.contact_mode = False
        self.target_force = 0.0

        # Publishers
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_curr_l = self.create_publisher(PoseStamped, '/openarm/left/current_pose', 10)
        self.pub_curr_r = self.create_publisher(PoseStamped, '/openarm/right/current_pose', 10)
        self.pub_torques_l = self.create_publisher(Float64MultiArray, '/openarm/left/joint_torques', 1)
        self.pub_torques_r = self.create_publisher(Float64MultiArray, '/openarm/right/joint_torques', 1)
        self.pub_force_tip = self.create_publisher(Float64, '/openarm/right/measured_force_tip_z', 10)
        self.pub_force_calc = self.create_publisher(Float64, '/openarm/right/calculated_force_tip_z', 10)
        self.pub_contact_state = self.create_publisher(Bool, '/openarm/right/contact_mode', 10)

        # Subscribers
        self.create_subscription(PoseStamped, '/openarm/left/target_pose', self._cb_left, 10)
        self.create_subscription(PoseStamped, '/openarm/right/target_pose', self._cb_right, 10)
        self.create_subscription(Float64, '/openarm/right/wrist_cmd', self._cb_wrist, 10)
        self.create_subscription(Float64, '/openarm/left/gripper_cmd', self._cb_gripper, 10)
        self.create_subscription(Float64, '/openarm/right/target_force', self._cb_force, 10)
        self.create_subscription(Bool, '/openarm/right/contact_mode', self._cb_contact_mode, 10)

    def _cb_left(self, msg):
        """
        Get the left TCP target pose.
        """
        self.left_pose = msg.pose

    def _cb_right(self, msg):
        """
        Get the right TCP target pose.
        """
        self.right_pose = msg.pose

    def _cb_wrist(self, msg):
        """
        Get the wrist target angle.
        """
        self.target_wrist_angle = float(msg.data)

    def _cb_gripper(self, msg):
        """
        Get the gripper target pose.
        """
        val = float(np.clip(msg.data, 0.0, 0.044))
        self.target_gripper_pos = val

    def _cb_force(self, msg):
        """
        Get the target force.
        """
        self.target_force = float(max(0.0, msg.data))

    def _cb_contact_mode(self, msg):
        """
        Get if the contact mode is active.
        """
        self.contact_mode = bool(msg.data)

    def publish_torques(self, tau_l, tau_r):
        """
        Publish the left and right torques.
        """
        msg_l = Float64MultiArray()
        msg_l.data = [float(x) for x in tau_l]
        self.pub_torques_l.publish(msg_l)

        msg_r = Float64MultiArray()
        msg_r.data = [float(x) for x in tau_r]
        self.pub_torques_r.publish(msg_r)

    def publish_forces(self, f_tip_direct, f_tip_calc):
        """
        Publish the ground truth and calculated forces, and the mode.
        """
        msg_tip = Float64()
        msg_tip.data = float(f_tip_direct)
        self.pub_force_tip.publish(msg_tip)

        msg_calc = Float64()
        msg_calc.data = float(f_tip_calc)
        self.pub_force_calc.publish(msg_calc)

        msg_c = Bool()
        msg_c.data = bool(self.contact_mode)
        self.pub_contact_state.publish(msg_c)

    def publish_states(self, pos_l, R_l, pos_r, R_r, joint_names, q_vals):
        """
        Publish the current state of the robot.
        """
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

def main():
    # File paths
    pkg_path = get_package_share_directory('openarm_mujoco')
    xml_path = os.path.join(pkg_path, 'v1', 'openarm_bimanual_edited.xml')
    desc_pkg_path = get_package_share_directory('openarm_description')
    urdf_path = os.path.join(desc_pkg_path, 'assets', 'robot', 'openarm_v1.0', 'urdf', 'example', 'v1_edited.urdf')
    
    # Initialization
    rclpy.init()
    node = RVizBridgeNode()
    robot = MujocoRobotInterface(xml_path)
    controller = BimanualCartesianController(
        urdf_path,
        left_ee_frame="openarm_left_hand_tcp",
        right_ee_frame="openarm_right_hand_tcp"
    )
    
    # Get IDs
    wrist_site_id   = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "force_measure_point")
    tip_site_id     = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "ultrasound_tip")
    clarius_body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "clarius_c3_hd3")
    ee_adapter_id   = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_ee_adapter")
    subtree_body_ids = get_subtree_body_ids(robot.model, ee_adapter_id)

    # Fixed angles for the fingers
    fixed_fingers = {
        'right_p-abd': 0.0,  'right_p-mcp': 1.72, 'right_p-pip': 1.5,
        'right_r-abd': 0.0,  'right_r-mcp': 1.72, 'right_r-pip': 1.7,
        'right_m-abd': 0.0,  'right_m-mcp': 1.72, 'right_m-pip': 1.7,
        'right_i-abd': 0.0,  'right_i-mcp': 1.4,  'right_i-pip': 1.3,
        'right_t-cmc': -0.78,'right_t-abd': 0.95, 'right_t-mcp': 0.1,  'right_t-pip': 0.5
    }

    # Initial pose for the gripper
    j_adr_g1 = robot._get_joint_qpos_adr('openarm_left_finger_joint1')
    j_adr_g2 = robot._get_joint_qpos_adr('openarm_left_finger_joint2')
    if j_adr_g1 is not None: robot.data.qpos[j_adr_g1] = 0.035
    if j_adr_g2 is not None: robot.data.qpos[j_adr_g2] = 0.035

    # Initial pose for the wrist
    j_adr_w = robot._get_joint_qpos_adr('right_wrist')
    if j_adr_w is not None: robot.data.qpos[j_adr_w] = 0.0
    for j_name, angle in fixed_fingers.items():
        j_adr = robot._get_joint_qpos_adr(j_name)
        if j_adr is not None: robot.data.qpos[j_adr] = angle

    # Get joint names
    all_joint_names = [controller.model.names[i] for i in range(1, len(controller.model.joints))]

    # Alpha for the filter
    alpha_pos = 0.35
    alpha_rot = 0.35

    # Initial desired poses
    pos_d_l, R_d_l = None, None
    pos_d_r, R_d_r = None, None

    # Move a timestep forward
    mujoco.mj_forward(robot.model, robot.data)

    # Initial values
    q_pin = np.zeros(controller.model.nq)
    dq_pin = np.zeros(controller.model.nv)
    f_norm_calc_prev = 0.0
    zero_force_counter = 0

    # Simulation timestep
    dt = robot.model.opt.timestep

    # History of data
    t_contact_list = []
    f_target_list = []
    f_calc_list = []
    f_direct_list = []
    contact_time_counter = 0.0
    
    # Main loop
    try:
        while robot.is_running():
            # Update kinematics
            rclpy.spin_once(node, timeout_sec=0.0)
            robot.update_kinematics()

            # Wrist control
            j_adr_w, v_adr_w = robot._get_joint_adrs('right_wrist')
            q_w_curr = robot.data.qpos[j_adr_w] if j_adr_w is not None else 0.0
            dq_w_curr = robot.data.qvel[v_adr_w] if v_adr_w is not None else 0.0
            
            wrist_cmd = controller.compute_wrist_cmd(
                q_w_curr, dq_w_curr, node.target_wrist_angle, is_position_actuator=True
            )
            
            wrist_act_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_actuator")
            if wrist_act_id != -1:
                robot.data.ctrl[wrist_act_id] = wrist_cmd

            # Synchronize the state from MuJoCo to Pinocchio
            for i in range(1, len(controller.model.joints)):
                urdf_jname = controller.model.names[i]
                mj_jname = URDF_TO_MJ_JOINTS.get(urdf_jname, urdf_jname)
                j_qadr, j_vadr = robot._get_joint_adrs(mj_jname)
                if j_qadr is not None and j_vadr is not None:
                    q_pin[controller.model.joints[i].idx_q] = robot.data.qpos[j_qadr]
                    dq_pin[controller.model.joints[i].idx_v] = robot.data.qvel[j_vadr]
            pin.forwardKinematics(controller.model, controller.data, q_pin, dq_pin)
            pin.updateFramePlacements(controller.model, controller.data)

            # Initialize desired poses
            if pos_d_l is None:
                H_l = controller.data.oMf[controller.left_ee_id]
                pos_d_l = H_l.translation.copy()
                R_d_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_d_r = H_r.translation.copy()
                R_d_r = H_r.rotation.copy()

            # Filter for following the left arm
            if node.left_pose is not None:
                # Desired position filtered
                target_p = np.array([node.left_pose.position.x, node.left_pose.position.y, node.left_pose.position.z])
                q_l = node.left_pose.orientation
                pos_d_l = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_l

                # Desired rotation filtered
                q_curr = R_scipy.from_matrix(R_d_l).as_quat()
                q_targ = np.array([q_l.x, q_l.y, q_l.z, q_l.w])
                if np.dot(q_curr, q_targ) < 0.0: q_targ = -q_targ
                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_l = slerp([alpha_rot])[0].as_matrix()

            # State machine
            if node.target_force > 0.0:
                if not node.contact_mode and f_norm_calc_prev > 0.5:
                    node.contact_mode = True
                    zero_force_counter = 0
                    node.get_logger().info("Contact established: Contact mode activated.")
                elif node.contact_mode and f_norm_calc_prev < 0.1:
                    zero_force_counter += 1
                    if zero_force_counter > 300:
                        node.contact_mode = False
                        controller.reset_force_integral()
                        node.get_logger().info("Contact lost: Contact mode deactivated.")
                else:
                    zero_force_counter = 0
            else:
                node.contact_mode = False
                controller.reset_force_integral()

            # Filter for following the right arm
            if node.right_pose is not None:
                # Position desired filtered
                target_p = np.array([node.right_pose.position.x, node.right_pose.position.y, node.right_pose.position.z])
                pos_d_r = alpha_pos * target_p + (1.0 - alpha_pos) * pos_d_r

                # Rotation desired filtered (modified to normal to the surface if the contact mode is on)
                q_r = node.right_pose.orientation
                q_user = np.array([q_r.x, q_r.y, q_r.z, q_r.w])

                if node.contact_mode:
                    in_contact_geom, normal_world = get_abdomen_normal_and_contact(robot)
                    R_user_marker = R_scipy.from_quat(q_user).as_matrix()
                    R_d_r_auto = compute_tcp_rotation_from_normal(normal_world, R_user_marker)
                    
                    q_auto = R_scipy.from_matrix(R_d_r_auto).as_quat()
                    if np.dot(q_user, q_auto) < 0.0: q_auto = -q_auto
                    slerp_auto = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_user, q_auto])))
                    q_targ = slerp_auto([0.25])[0].as_quat()
                else:
                    q_targ = q_user

                q_curr = R_scipy.from_matrix(R_d_r).as_quat()
                if np.dot(q_curr, q_targ) < 0.0: q_targ = -q_targ
                slerp = Slerp([0.0, 1.0], R_scipy.from_quat(np.vstack([q_curr, q_targ])))
                R_d_r = slerp([alpha_rot])[0].as_matrix()

            # Compute the torques for both arms
            tau_l, tau_r = controller.compute_bimanual_torques(
                q_pin=q_pin,
                dq_pin=dq_pin,
                pos_d_l=pos_d_l, R_d_l=R_d_l, vel_d_l=np.zeros(3),
                pos_d_r=pos_d_r, R_d_r=R_d_r, vel_d_r=np.zeros(3),
                contact_mode=node.contact_mode,
                f_d=node.target_force,
                f_measured=f_norm_calc_prev,
                dt=dt
            )

            # Clip torques according to the motor limits
            tau_l_clamped = np.clip(tau_l, -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_r_clamped = np.clip(tau_r, -TORQUE_LIMITS, TORQUE_LIMITS)

            # Assign torques
            cmd_dict = {}
            for idx, act_id in enumerate(robot.left_act_ids):
                if act_id is not None: cmd_dict[act_id] = tau_l_clamped[idx]
            for idx, act_id in enumerate(robot.right_act_ids):
                if act_id is not None: cmd_dict[act_id] = tau_r_clamped[idx]

            # Keep the fingers fixed
            for j_name, angle in fixed_fingers.items():
                actuator_name = j_name + "_actuator"
                act_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if act_id != -1: cmd_dict[act_id] = angle

            # Gripper control
            j_adr_g1, v_adr_g1 = robot._get_joint_adrs('openarm_left_finger_joint1')
            q_g_curr = robot.data.qpos[j_adr_g1] if j_adr_g1 is not None else 0.035
            dq_g_curr = robot.data.qvel[v_adr_g1] if v_adr_g1 is not None else 0.0
            tau_gripper = controller.compute_gripper_torque(q_g_curr, dq_g_curr, node.target_gripper_pos)

            if robot.left_finger_act_id is not None: cmd_dict[robot.left_finger_act_id] = tau_gripper
            if robot.left_finger2_act_id is not None: cmd_dict[robot.left_finger2_act_id] = tau_gripper

            # Send the torques and take a step in the simulation
            robot.send_torques(cmd_dict)
            robot.step()

            # Direct force measurement
            f_norm_direct = get_tip_direct_contact_force(
                robot.model, robot.data, clarius_body_id, tip_site_id
            )

            # Calculated force measurement (sensor simulation)
            f_norm_calc = 0.0
            if ee_adapter_id != -1 and wrist_site_id != -1 and tip_site_id != -1:
                f_raw_w, tau_raw_w = read_sensone_raw_wrench(
                    robot.model, robot.data, ee_adapter_id, wrist_site_id, subtree_body_ids
                )

                R_w_world = robot.data.site_xmat[wrist_site_id].reshape(3, 3)
                p_w_world = robot.data.site_xpos[wrist_site_id]
                R_t_world = robot.data.site_xmat[tip_site_id].reshape(3, 3)
                p_t_world = robot.data.site_xpos[tip_site_id]

                # Gravity compensation
                m_sub = float(robot.model.body_subtreemass[ee_adapter_id])
                p_com_world = robot.data.subtree_com[ee_adapter_id]
                r_com_w = R_w_world.T @ (p_com_world - p_w_world)
                g_w = R_w_world.T @ np.array([0.0, 0.0, -G_ACC])
                f_grav_w = m_sub * g_w
                f_ext_w = f_raw_w - f_grav_w
                R_t_w = R_t_world.T @ R_w_world
                f_ext_t = R_t_w @ f_ext_w

                n_tip_local = np.array([0.0, 0.0, 1.0])
                f_norm_calc = max(0.0, float(np.dot(f_ext_t, n_tip_local)))

            f_norm_calc_prev = f_norm_calc

            # Sample contact forces
            if node.contact_mode or node.target_force > 0.0 or f_norm_calc > 0.1:
                contact_time_counter += dt
                t_contact_list.append(contact_time_counter)
                f_target_list.append(node.target_force)
                f_calc_list.append(f_norm_calc)
                f_direct_list.append(f_norm_direct)

            # Publish forces
            node.publish_forces(f_norm_direct, f_norm_calc)

            # Logger
            node.get_logger().info(
                f"[IMPEDANCE HYBRID] Mode: {'CONTACT' if node.contact_mode else 'FLIGHT'} | Force Target: {node.target_force:.1f} N | Ground Truth: {f_norm_direct:.2f} N | Sensor: {f_norm_calc:.2f} N",
                throttle_duration_sec=0.2
            )

            # Publish torques
            node.publish_torques(tau_l_clamped, tau_r_clamped)

            # Publish states
            q_list = [float(q_pin[controller.model.joints[i].idx_q]) for i in range(1, len(controller.model.joints))]
            node.publish_states(pos_d_l, R_d_l, pos_d_r, R_d_r, all_joint_names, np.array(q_list, dtype=np.float64))

    except KeyboardInterrupt:
        print("\n[INFO] Closing MuJoCo...")
    finally:
        robot.close()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    # Plot
    if len(t_contact_list) > 0:
        print(f"\n[PLOTTING] Opening plot with {len(t_contact_list)} samples...")
        p = multiprocessing.Process(
            target=plot_force_exerted,
            args=(list(t_contact_list), list(f_target_list), list(f_calc_list), list(f_direct_list))
        )
        p.start()
        p.join()
    else:
        print("[INFO] No samples available.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
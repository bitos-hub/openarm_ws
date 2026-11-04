import numpy as np
import pinocchio as pin

class BimanualCartesianController:
    """
    Bimanual Impedance Cartesian Controller tunned for normal force regulation
    and fast dynamic response durint X-Z movements.
    """
    def __init__(
        self, 
        urdf_path, 
        left_ee_frame="openarm_left_hand_tcp", 
        right_ee_frame="openarm_right_hand_tcp"
    ):
        # Get data from the model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Get end efector IDs
        self.left_ee_id = self.model.getFrameId(left_ee_frame)
        self.right_ee_id = self.model.getFrameId(right_ee_frame)

        # Nominal gains for the left arm
        self.Kp_trans_left = np.diag([600.0, 600.0, 600.0])
        self.Kd_trans_left = np.diag([35.0, 35.0, 35.0])
        self.Kp_rot_left   = np.diag([35.0, 35.0, 35.0])
        self.Kd_rot_left   = np.diag([4.0, 4.0, 4.0])

        # Nominal gains for the right arm
        self.Kp_trans_right = np.diag([400.0, 400.0, 400.0])
        self.Kd_trans_right = np.diag([40.0, 40.0, 40.0])
        self.Kp_rot_right   = np.diag([80.0, 80.0, 80.0])
        self.Kd_rot_right   = np.diag([8.0, 8.0, 8.0])

        # Null space
        self.Kp_null_vec = np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        self.Kd_null_vec = np.array([0.5, 0.5, 0.5, 1.0, 0.3, 0.3, 0.2])
        self.q4_target_elbow = 0.8 # Around 45°
        self.damping_lambda_sq = 1e-3
        self.max_force_n = 30.0
        self.max_torque_nm = 10.0

        # Force admittance
        self.Kp_force = 0.015       # Balanced for static and dynamic situations
        self.Ki_force = 0.0008      # Fix static offset
        self.Kd_force = 0.00008     # Damping
        self.force_integral_err = 0.0
        self.y_offset_admittance = 0.0
        self.f_filtered = 0.0
        self.f_prev_calc = 0.0

        # Gains for the wrist
        self.Kp_wrist = 150.0
        self.Kd_wrist = 10.0

        # Wrist joint limits
        self.wrist_limit = (-1.134, 0.610)

        # Get joint IDs
        self.left_arm_q_ids = [self.model.getJointId(f"openarm_left_joint{i}") for i in range(1, 8)]
        self.right_arm_q_ids = [self.model.getJointId(f"openarm_right_joint{i}") for i in range(1, 8)]
        
        # Get indices in the velocity vector of each joint
        self.left_arm_v_ids = [self.model.joints[j_id].idx_v for j_id in self.left_arm_q_ids]
        self.right_arm_v_ids = [self.model.joints[j_id].idx_v for j_id in self.right_arm_q_ids]

    def reset_force_integral(self):
        """
        Reset variables for the force integral.
        """
        self.force_integral_err = 0.0
        self.y_offset_admittance = 0.0
        self.f_filtered = 0.0
        self.f_prev_calc = 0.0

    def compute_arm_torques(
        self,
        q_pin, dq_pin,
        pos_d, R_d, vel_d,
        ee_id, arm_v_ids, arm_q_ids,
        is_right_arm=True,
        contact_mode=False,
        f_d=0.0,
        f_measured=0.0,
        dt=0.001
    ):
        """
        Compute the arm torques.
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q_pin, dq_pin)
        pin.updateFramePlacements(self.model, self.data)

        # Get the joint positions and velocities
        q_arm = q_pin[arm_q_ids]
        dq_arm = dq_pin[arm_v_ids]

        # Get the rotation and translation matrices
        H = self.data.oMf[ee_id]
        R = H.rotation
        pos = H.translation

        # Modified desired position for the contact mode
        pos_d_modified = pos_d.copy()

        if is_right_arm and contact_mode and f_d > 0.0:
            # --- Contact mode ---
            # Force filter
            self.f_filtered = 0.35 * f_measured + 0.65 * self.f_filtered
            df_dt = (self.f_filtered - self.f_prev_calc) / dt
            self.f_prev_calc = self.f_filtered
            
            # Thrust direction (-Y_tcp)
            probe_dir_world = -R[:, 1]
            
            # Force error
            err_f = f_d - self.f_filtered
            
            # Integral
            self.force_integral_err += err_f * dt
            self.force_integral_err = np.clip(self.force_integral_err, -2.0, 2.0)

            # Admittance force controll
            dy_admittance = (self.Kp_force * err_f + self.Ki_force * self.force_integral_err - self.Kd_force * df_dt) * dt
            dy_admittance = np.clip(dy_admittance, -0.0015, 0.0015)

            # Update admittance
            self.y_offset_admittance += dy_admittance
            self.y_offset_admittance = np.clip(self.y_offset_admittance, 0.0, 0.15)

            # Modify desired position based on the admittance
            pos_d_modified += probe_dir_world * self.y_offset_admittance
        else:
            # Back to flight mode
            if is_right_arm:
                self.reset_force_integral()

        # Get the jacobian
        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_pin, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, arm_v_ids]

        if is_right_arm:
            if contact_mode:
                # --- Contact mode ---
                # Cartesian rigidy adjusted
                Kp_trans = np.diag([550.0, 250.0, 550.0])
                Kd_trans = np.diag([32.0, 25.0, 32.0])
                Kp_rot   = np.diag([100.0, 100.0, 100.0])
                Kd_rot   = np.diag([10.0, 10.0, 10.0])
            else:
                # --- Flight mode ---
                Kp_trans, Kd_trans = self.Kp_trans_right, self.Kd_trans_right
                Kp_rot, Kd_rot     = self.Kp_rot_right, self.Kd_rot_right
        else:
            # Nominal gains for the left arm
            Kp_trans, Kd_trans = self.Kp_trans_left, self.Kd_trans_left
            Kp_rot, Kd_rot     = self.Kp_rot_left, self.Kd_rot_left

        # Local position and rotation error
        e_pos_local = R.T @ (pos_d_modified - pos)
        R_err_local = R.T @ R_d
        e_rot_local = pin.log3(R_err_local)

        # Local velocity error
        v_world = J[:3, :] @ dq_arm
        w_world = J[3:, :] @ dq_arm
        v_local = R.T @ v_world
        w_local = R.T @ w_world
        e_v_local = (R.T @ vel_d) - v_local
        e_w_local = -w_local

        # Cartesian impedance control
        F_trans_local = Kp_trans @ e_pos_local + Kd_trans @ e_v_local
        F_rot_local   = Kp_rot @ e_rot_local + Kd_rot @ e_w_local
        
        # Clip force and torque
        if self.max_force_n is not None:
            f_norm = np.linalg.norm(F_trans_local)
            if f_norm > self.max_force_n:
                F_trans_local = (F_trans_local / f_norm) * self.max_force_n

        if self.max_torque_nm is not None:
            t_norm = np.linalg.norm(F_rot_local)
            if t_norm > self.max_torque_nm:
                F_rot_local = (F_rot_local / t_norm) * self.max_torque_nm

        # Task torque
        F_6d_world = np.hstack([R @ F_trans_local, R @ F_rot_local])
        tau_task = J.T @ F_6d_world

        # Nullspace torque
        JJT = J @ J.T
        J_dls_pinv = J.T @ np.linalg.inv(JJT + self.damping_lambda_sq * np.eye(6))
        N = np.eye(7) - J.T @ J_dls_pinv.T
        q_target_null = q_arm.copy()
        q_target_null[3] = self.q4_target_elbow
        tau_null_raw = self.Kp_null_vec * (q_target_null - q_arm) - self.Kd_null_vec * dq_arm
        tau_null = N @ tau_null_raw

        # Gravity torque
        tau_gravity = pin.nonLinearEffects(self.model, self.data, q_pin, dq_pin)[arm_v_ids]
        
        # Total torque
        tau_total = tau_task + tau_null + tau_gravity
        return tau_total

    def compute_bimanual_torques(
        self,
        q_pin, dq_pin,
        pos_d_l, R_d_l, vel_d_l,
        pos_d_r, R_d_r, vel_d_r,
        contact_mode=False,
        f_d=0.0,
        f_measured=0.0,
        dt=0.001
    ):
        """
        Compute the torques of both arms at the same time.
        """
        # Left arm torques
        tau_l = self.compute_arm_torques(
            q_pin, dq_pin, pos_d_l, R_d_l, vel_d_l,
            self.left_ee_id, self.left_arm_v_ids, self.left_arm_q_ids,
            is_right_arm=False, contact_mode=False, dt=dt
        )

        # Right arm torques
        tau_r = self.compute_arm_torques(
            q_pin, dq_pin, pos_d_r, R_d_r, vel_d_r,
            self.right_ee_id, self.right_arm_v_ids, self.right_arm_q_ids,
            is_right_arm=True, contact_mode=contact_mode, f_d=f_d, f_measured=f_measured, dt=dt
        )

        return tau_l, tau_r

    def compute_wrist_cmd(self, q_curr, dq_curr, q_d, is_position_actuator=True):
        """
        Compute the wrist control.
        """
        q_d_clamped = float(np.clip(q_d, self.wrist_limit[0], self.wrist_limit[1]))
        if is_position_actuator:
            return q_d_clamped
        else:
            return self.Kp_wrist * (q_d_clamped - q_curr) - self.Kd_wrist * dq_curr

    def compute_gripper_torque(self, q_curr, dq_curr, q_d_init):
        """
        Compute the gripper controll.
        """
        return 200.0 * (q_d_init - q_curr) - 10.0 * dq_curr
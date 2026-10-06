import numpy as np
import pinocchio as pin

class PinocchioImpedanceController:
    def __init__(self, urdf_path, left_ee_frame="openarm_left_hand_tcp", right_ee_frame="openarm_right_hand_tcp"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.left_ee_id = self.model.getFrameId(left_ee_frame)
        self.right_ee_id = self.model.getFrameId(right_ee_frame)

        self.tcp_offset = np.zeros(3)

        # Posturas nominales de referencia
        self.q_nominal_left = np.array([0.0, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.q_nominal_right = np.array([0.0, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Rigidez y Amortiguamiento
        self.Kp_trans_left = np.diag([600.0, 600.0, 600.0])
        self.Kd_trans_left = np.diag([35.0, 35.0, 35.0])
        self.Kp_rot_left   = np.diag([35.0, 35.0, 35.0])
        self.Kd_rot_left   = np.diag([4.0, 4.0, 4.0])

        self.Kp_trans_right = np.diag([450.0, 450.0, 450.0])
        self.Kd_trans_right = np.diag([35.0, 35.0, 35.0])
        self.Kp_rot_right   = np.diag([20.0, 20.0, 20.0])
        self.Kd_rot_right   = np.diag([3.0, 3.0, 3.0])

        # Obtención de índices de velocidad/articulaciones en Pinocchio
        self.left_arm_v_ids = [self.model.joints[self.model.getJointId(f"openarm_left_joint{i}")].idx_v for i in range(1, 8)]
        self.right_arm_v_ids = [self.model.joints[self.model.getJointId(f"openarm_right_joint{i}")].idx_v for i in range(1, 8)]

    def compute_arm_torque(
        self, q_pin, dq_pin, frame_id, arm_v_ids, pos_d, vel_d, R_d, 
        q_nominal=None, is_right_arm=True, max_force_n=40.0, max_torque_nm=10.0, include_gravity=True
    ):
        pin.forwardKinematics(self.model, self.data, q_pin, dq_pin)
        pin.updateFramePlacements(self.model, self.data)

        q_arm = q_pin[arm_v_ids]
        dq_arm = dq_pin[arm_v_ids]

        H = self.data.oMf[frame_id]
        R = H.rotation      # Orientación actual
        pos = H.translation  # Posición actual

        # Jacobiano en marco LOCAL_WORLD_ALIGNED (6x7)
        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_pin, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, arm_v_ids]  # 6x7

        if is_right_arm:
            Kp_trans, Kd_trans = self.Kp_trans_right, self.Kd_trans_right
            Kp_rot, Kd_rot     = self.Kp_rot_right, self.Kd_rot_right
        else:
            Kp_trans, Kd_trans = self.Kp_trans_left, self.Kd_trans_left
            Kp_rot, Kd_rot     = self.Kp_rot_left, self.Kd_rot_left

        # --- 1. ERRORES EN MARCO LOCAL ---
        e_pos_local = R.T @ (pos_d - pos)
        R_err_local = R.T @ R_d
        e_rot_local = pin.log3(R_err_local)

        # Velocidades expresadas en marco local
        v_world = J[:3, :] @ dq_arm
        w_world = J[3:, :] @ dq_arm
        v_local = R.T @ v_world
        w_local = R.T @ w_world

        e_v_local = (R.T @ vel_d) - v_local
        e_w_local = -w_local

        # --- 2. FUERZAS DE IMPEDANCIA LOCAL (Control PD) ---
        F_trans_local = Kp_trans @ e_pos_local + Kd_trans @ e_v_local
        F_rot_local   = Kp_rot @ e_rot_local + Kd_rot @ e_w_local

        # Acotar la fuerza/torque cartesiano para evitar aceleraciones descontroladas
        if max_force_n is not None:
            f_norm = np.linalg.norm(F_trans_local)
            if f_norm > max_force_n:
                F_trans_local = (F_trans_local / f_norm) * max_force_n

        if max_torque_nm is not None:
            t_norm = np.linalg.norm(F_rot_local)
            if t_norm > max_torque_nm:
                F_rot_local = (F_rot_local / t_norm) * max_torque_nm

        F_6d_world = np.hstack([R @ F_trans_local, R @ F_rot_local])
        tau_task = J.T @ F_6d_world

        # --- 3. ESPACIO NULO (Estabilización 7-DOF) ---
        J_pinv = np.linalg.pinv(J)
        N = np.eye(7) - J.T @ J_pinv.T

        if q_nominal is None:
            q_nominal = q_arm

        Kp_null = 5.0 if is_right_arm else 3.0
        Kd_null = 0.8 if is_right_arm else 0.5
        tau_null_raw = Kp_null * (q_nominal - q_arm) - Kd_null * dq_arm
        tau_null = N @ tau_null_raw

        # --- 4. COMPENSACIÓN DE DINÁMICA / GRAVEDAD ---
        tau_total = tau_task + tau_null
        if include_gravity:
            tau_gravity = self.compute_nonlinear_effects(q_pin, dq_pin)[arm_v_ids]
            tau_total += tau_gravity

        return tau_total

    def compute_nonlinear_effects(self, q_pin, dq_pin):
        return pin.nonLinearEffects(self.model, self.data, q_pin, dq_pin)

    def compute_gripper_torque(self, q_curr, dq_curr, q_d_init):
        Kp_grip = 10.0
        Kd_grip = 2.0
        return Kp_grip * (q_d_init - q_curr) - Kd_grip * dq_curr
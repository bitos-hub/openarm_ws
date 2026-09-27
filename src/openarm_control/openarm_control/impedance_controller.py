import numpy as np
import pinocchio as pin

class PinocchioImpedanceController:
    def __init__(self, urdf_path, left_ee_frame="openarm_left_hand_tcp", right_ee_frame="openarm_right_hand_tcp"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.left_ee_id = self.model.getFrameId(left_ee_frame)
        self.right_ee_id = self.model.getFrameId(right_ee_frame)

        self.tcp_offset = np.zeros(3)

        # Rigidez local calibrada para sostener la masa del brazo
        self.Kp_trans_local = np.diag([120.0, 120.0, 120.0])
        self.Kd_trans_local = np.diag([50.0, 50.0, 50.0])

        self.Kp_rot_local = np.diag([40.0, 40.0, 40.0])
        self.Kd_rot_local = np.diag([20.0, 20.0, 20.0])

        # Obtención precisa de índices de velocidad en Pinocchio
        self.left_arm_v_ids = [self.model.joints[self.model.getJointId(f"openarm_left_joint{i}")].idx_v for i in range(1, 8)]
        self.right_arm_v_ids = [self.model.joints[self.model.getJointId(f"openarm_right_joint{i}")].idx_v for i in range(1, 8)]

    def compute_arm_torque(self, q_pin, dq_pin, frame_id, arm_v_ids, pos_d, vel_d, R_d, q_nominal=None):
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

        # --- 1. ERRORES EN MARCO LOCAL DE LA HERRAMIENTA ---
        e_pos_local = R.T @ (pos_d - pos)

        # Error de rotación correcto en marco local
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
        F_trans_local = self.Kp_trans_local @ e_pos_local + self.Kd_trans_local @ e_v_local
                         
        F_rot_local = self.Kp_rot_local @ e_rot_local + self.Kd_rot_local @ e_w_local

        F_6d_world = np.hstack([R @ F_trans_local, R @ F_rot_local])
        tau_task = J.T @ F_6d_world

        # --- 3. CONTROL EN EL ESPACIO NULO (Estabilización de 7-DOF) ---
        # Proyector de espacio nulo: N = I - J^T * J^+T
        J_pinv = np.linalg.pinv(J)
        N = np.eye(7) - J.T @ J_pinv.T

        # Postura nominal por defecto (si no se proporciona, usa la actual)
        if q_nominal is None:
            q_nominal = q_arm

        # Rigidez/Amortiguamiento articular secundario para eliminar la oscilación interna
        Kp_null = 1.0
        Kd_null = 0.5
        tau_null_raw = Kp_null * (q_nominal - q_arm) - Kd_null * dq_arm

        tau_null = N @ tau_null_raw

        return tau_task + tau_null

    def compute_nonlinear_effects(self, q_pin, dq_pin):
        return pin.nonLinearEffects(self.model, self.data, q_pin, dq_pin)

    def compute_gripper_torque(self, q_curr, dq_curr, q_d_init):
        """
        Mantiene los grippers fijados en su posición de spawn.
        Ganas reducidas para evitar oscilaciones contra topes físicos.
        """
        Kp_grip = 10.0  # Rigidez suave
        Kd_grip = 2.0   # Amortiguamiento proporcional
        
        return Kp_grip * (q_d_init - q_curr) - Kd_grip * dq_curr

    def compute_friction_compensation(self, dq_curr, v_ids, coulomb=0.1, damping=0.4):
        """
        Calcula la fricción estática (Coulomb) y viscosa (Damping) 
        para anular la resistencia interna de MuJoCo.
        """
        dq_arm = dq_curr[v_ids]
        #tau_fric = coulomb * np.sign(dq_arm) + damping * dq_arm
        tau_fric = coulomb * np.tanh(10.0 * dq_arm) + damping * dq_arm
        return tau_fric
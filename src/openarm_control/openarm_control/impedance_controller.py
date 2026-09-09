import numpy as np
import pinocchio as pin

class PinocchioImpedanceController:
    def __init__(self, urdf_path, left_ee_frame="openarm_left_hand_tcp", right_ee_frame="openarm_right_hand_tcp"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.left_ee_id = self.model.getFrameId(left_ee_frame)
        self.right_ee_id = self.model.getFrameId(right_ee_frame)

        self.tcp_offset = np.array([0.0, 0.0, 0.10])

        # Ganancias de rigidez cartesiana elevadas para mejor seguimiento
        self.Kp_cart = np.diag([500.0, 500.0, 500.0, 60.0, 60.0, 60.0])
        self.Kd_cart = np.diag([50.0, 50.0, 50.0, 6.0, 6.0, 6.0])

        # Ganancias incrementadas para vencer fricción en grippers
        # Ganancias optimizadas para vencer fricción mecánica en MuJoCo
        self.Kp_grip = 300.0
        self.Kd_grip = 15.0

        self.left_arm_v_ids = [self.model.getJointId(f"openarm_left_joint{i}") - 1 for i in range(1, 8)]
        self.right_arm_v_ids = [self.model.getJointId(f"openarm_right_joint{i}") - 1 for i in range(1, 8)]

    def compute_arm_torque(self, q_pin, dq_pin, frame_id, arm_v_ids, pos_d, vel_d, R_d):
        pin.forwardKinematics(self.model, self.data, q_pin, dq_pin)
        pin.updateFramePlacements(self.model, self.data)

        H = self.data.oMf[frame_id]
        R = H.rotation
        pos = H.translation + R @ self.tcp_offset

        J_full = pin.computeFrameJacobian(
            self.model, self.data, q_pin, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, arm_v_ids]

        e_pos = pos_d - pos
        e_rot = pin.log3(R_d @ R.T)
        e_6d = np.hstack([e_pos, e_rot])

        v = J @ dq_pin[arm_v_ids]
        v_d = np.hstack([vel_d, np.zeros(3)])
        e_v = v_d - v

        F_6d = self.Kp_cart @ e_6d + self.Kd_cart @ e_v
        return J.T @ F_6d

    def compute_nonlinear_effects(self, q_pin, dq_pin):
        # Compensación dinámica completa: Gravedad + Coriolis + Fuerza centrífuga
        return pin.nonLinearEffects(self.model, self.data, q_pin, dq_pin)

    def compute_gripper_torque(self, q_curr, dq_curr, q_d):
    # El signo menos invierte la polaridad para adaptarla a la convención de MuJoCo
        return (self.Kp_grip * (q_d - q_curr) - self.Kd_grip * dq_curr)
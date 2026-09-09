import time
import numpy as np
import mujoco
import mujoco.viewer

class MujocoRobotInterface:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.start_time = time.time()

        self.left_arm_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.right_arm_names = [f"openarm_right_joint{i}" for i in range(1, 8)]

        self.left_finger_q_id, self.left_finger_v_id = self._get_joint_adrs("openarm_left_finger_joint1")
        self.right_finger_q_id, self.right_finger_v_id = self._get_joint_adrs("openarm_right_finger_joint1")

        self.left_act_ids = [self._get_actuator_id_by_joint(name) for name in self.left_arm_names]
        self.right_act_ids = [self._get_actuator_id_by_joint(name) for name in self.right_arm_names]

        self.left_finger_act_id = self._get_actuator_id_by_joint("openarm_left_finger_joint1")
        self.right_finger_act_id = self._get_actuator_id_by_joint("openarm_right_finger_joint1")

        # MOVER AQUÍ AL FINAL (después de definir los IDs)
        if self.left_finger_act_id is not None:
            self.model.actuator_ctrllimited[self.left_finger_act_id] = False
            self.model.actuator_ctrlrange[self.left_finger_act_id] = [-100.0, 100.0]

        if self.right_finger_act_id is not None:
            self.model.actuator_ctrllimited[self.right_finger_act_id] = False
            self.model.actuator_ctrlrange[self.right_finger_act_id] = [-100.0, 100.0]

    def _get_joint_qpos_adr(self, name):
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_qposadr[j_id] if j_id != -1 else None

    def _get_joint_adrs(self, joint_name):
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id != -1:
            return self.model.jnt_qposadr[j_id], self.model.jnt_dofadr[j_id]
        return None, None

    def _get_actuator_id_by_joint(self, joint_name):
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id != -1:
            for i in range(self.model.nu):
                if self.model.actuator_trnid[i, 0] == j_id:
                    return i
        return None

    def is_running(self): return self.viewer.is_running()
    def get_time(self): return time.time() - self.start_time
    def update_kinematics(self): mujoco.mj_kinematics(self.model, self.data)

    def send_torques(self, tau_dict):
        for act_id, torque in tau_dict.items():
            if act_id is not None and 0 <= act_id < self.model.nu:
                if self.model.actuator_ctrllimited[act_id]:
                    c_min = self.model.actuator_ctrlrange[act_id, 0]
                    c_max = self.model.actuator_ctrlrange[act_id, 1]
                    torque = np.clip(torque, c_min, c_max)
                self.data.ctrl[act_id] = torque

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        time.sleep(self.model.opt.timestep)

    def close(self): self.viewer.close()
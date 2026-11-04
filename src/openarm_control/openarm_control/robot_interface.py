import time
import numpy as np
import mujoco
import mujoco.viewer

class MujocoRobotInterface:
    def __init__(self, xml_path):
        # Setup
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.start_time = time.time()

        # Generate arms joints names
        self.left_arm_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.right_arm_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    
        # Map joints names to actuators IDs
        self.left_act_ids = [self._get_actuator_id_by_joint(name) for name in self.left_arm_names]
        self.right_act_ids = [self._get_actuator_id_by_joint(name) for name in self.right_arm_names]

        # Get left fingers actuators IDs
        self.left_finger_act_id = self._get_actuator_id_by_joint("openarm_left_finger_joint1")
        self.left_finger2_act_id = self._get_actuator_id_by_joint("openarm_left_finger_joint2")

        # Deactivate limits
        if self.left_finger_act_id is not None:
            self.model.actuator_ctrllimited[self.left_finger_act_id] = False
            self.model.actuator_ctrlrange[self.left_finger_act_id] = [-333.0, 333.0]

        if self.left_finger2_act_id is not None:
            self.model.actuator_ctrllimited[self.left_finger2_act_id] = False
            self.model.actuator_ctrlrange[self.left_finger2_act_id] = [-333.0, 333.0]

    def _get_joint_qpos_adr(self, name):
        """
        Find the memory address of a joint in the position vector.
        """
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_qposadr[j_id] if j_id != -1 else None

    def _get_joint_adrs(self, name):
        """
        Find the memory address of a joint in the position and velocity/force vector.
        """
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            return self.model.jnt_qposadr[j_id], self.model.jnt_dofadr[j_id]
        return None, None

    def _get_actuator_id_by_joint(self, name):
        """
        Get the actuator ID that matches a joint ID.
        """
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            for i in range(self.model.nu):
                if self.model.actuator_trnid[i, 0] == j_id:
                    return i
        return None

    def is_running(self): 
        """
        Returns True while the 3D window is open.
        """
        return self.viewer.is_running()
    
    def get_time(self): 
        """
        Get the simulation time.
        """
        return time.time() - self.start_time 
    
    def update_kinematics(self): 
        """
        Do the kinematics calculation.
        """
        mujoco.mj_kinematics(self.model, self.data) 

    def send_torques(self, tau_dict):
        """
        Send torques to actuators.
        """
        for act_id, torque in tau_dict.items():
            if act_id is not None and 0 <= act_id < self.model.nu:
                if self.model.actuator_ctrllimited[act_id]:
                    c_min = self.model.actuator_ctrlrange[act_id, 0]
                    c_max = self.model.actuator_ctrlrange[act_id, 1]
                    torque = np.clip(torque, c_min, c_max)
                self.data.ctrl[act_id] = torque

    def step(self):
        """
        Execute a full physical integration step of the simulation.
        """
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        time.sleep(self.model.opt.timestep)

    def close(self): 
        """
        Close the 3D window
        """
        self.viewer.close()
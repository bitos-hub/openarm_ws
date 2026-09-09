import os
import numpy as np
import mujoco
import pinocchio as pin
from openarm_control.trajectory_generator import CircleTrajectory
from openarm_control.impedance_controller import PinocchioImpedanceController
from openarm_control.robot_interface import MujocoRobotInterface
from ament_index_python.packages import get_package_share_directory

def draw_debug_trajectory(robot, pos_init_l, pos_init_r, R_init_l, R_init_r, planner, num_samples=48):
    if not hasattr(robot.viewer, 'user_scn'):
        return

    robot.viewer.user_scn.ngeom = 0
    freq = getattr(planner, 'w', 2 * np.pi * 0.3) / (2 * np.pi)
    period = 1.0 / freq if freq > 0 else 3.0
    times = np.linspace(0, period, num_samples)

    for t in times:
        pos_l, _, _, _ = planner.get_ref(t, pos_init_l, R_init_l)
        pos_r, _, _, _ = planner.get_ref(t, pos_init_r, R_init_r)

        for pos, color in [(pos_l, [0.2, 0.5, 1.0, 0.8]), (pos_r, [0.2, 1.0, 0.5, 0.8])]:
            if robot.viewer.user_scn.ngeom < robot.viewer.user_scn.maxgeom:
                g = robot.viewer.user_scn.geoms[robot.viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.006, 0.0, 0.0]),
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=np.array(color, dtype=np.float32)
                )
                robot.viewer.user_scn.ngeom += 1

def main():
    pkg_path = get_package_share_directory('openarm_mujoco')
    xml_path = os.path.join(pkg_path, 'v1', 'openarm_bimanual.xml')
    urdf_path = os.path.expanduser(
        "~/openarm_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
    )

    robot = MujocoRobotInterface(xml_path)
    planner = CircleTrajectory()
    controller = PinocchioImpedanceController(
        urdf_path,
        left_ee_frame="openarm_left_hand_tcp",
        right_ee_frame="openarm_right_hand_tcp"
    )

    pos_init_l, R_init_l = None, None
    pos_init_r, R_init_r = None, None
    trajectory_drawn = False
    print("Límites Gripper L:", robot.model.actuator_ctrlrange[robot.left_finger_act_id])
    print(f"ID Actuador Gripper L: {robot.left_finger_act_id}")
    print(f"ID Actuador Gripper R: {robot.right_finger_act_id}")
    try:
        while robot.is_running():
            t = robot.get_time()
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
                pos_init_l = H_l.translation.copy() + H_l.rotation.copy() @ controller.tcp_offset
                R_init_l = H_l.rotation.copy()

                H_r = controller.data.oMf[controller.right_ee_id]
                pos_init_r = H_r.translation.copy() + H_r.rotation.copy() @ controller.tcp_offset
                R_init_r = H_r.rotation.copy()

            if not trajectory_drawn:
                draw_debug_trajectory(robot, pos_init_l, pos_init_r, R_init_l, R_init_r, planner)
                trajectory_drawn = True

            pos_l, vel_l, R_l, grip_l_d = planner.get_ref(t, pos_init_l, R_init_l)
            pos_r, vel_r, R_r, grip_r_d = planner.get_ref(t, pos_init_r, R_init_r)

            tau_l = controller.compute_arm_torque(
                q_pin, dq_pin, controller.left_ee_id, controller.left_arm_v_ids, pos_l, vel_l, R_l
            )
            tau_r = controller.compute_arm_torque(
                q_pin, dq_pin, controller.right_ee_id, controller.right_arm_v_ids, pos_r, vel_r, R_r
            )

            # Lectura de posiciones
            q_g_l = robot.data.qpos[robot.left_finger_q_id] if robot.left_finger_q_id is not None else 0.0
            dq_g_l = robot.data.qvel[robot.left_finger_v_id] if robot.left_finger_v_id is not None else 0.0

            q_g_r = robot.data.qpos[robot.right_finger_q_id] if robot.right_finger_q_id is not None else 0.0
            dq_g_r = robot.data.qvel[robot.right_finger_v_id] if robot.right_finger_v_id is not None else 0.0

            # Gripper Derecho (PD Estándar)
            # Gripper Derecho e Izquierdo usando el mismo controlador:
            tau_g_l = controller.compute_gripper_torque(q_g_l, dq_g_l, grip_l_d)
            tau_g_r = controller.compute_gripper_torque(q_g_r, dq_g_r, grip_r_d)

            # Cálculo de dinámica completa (Coriolis + Gravedad)
            tau_dyn = controller.compute_nonlinear_effects(q_pin, dq_pin)

            cmd_dict = {}
            for idx, act_id in enumerate(robot.left_act_ids):
                if act_id is not None:
                    v_id = controller.left_arm_v_ids[idx]
                    cmd_dict[act_id] = tau_l[idx] + tau_dyn[v_id]

            for idx, act_id in enumerate(robot.right_act_ids):
                if act_id is not None:
                    v_id = controller.right_arm_v_ids[idx]
                    cmd_dict[act_id] = tau_r[idx] + tau_dyn[v_id]

            if robot.left_finger_act_id is not None:
                cmd_dict[robot.left_finger_act_id] = tau_g_l
            if robot.right_finger_act_id is not None:
                cmd_dict[robot.right_finger_act_id] = tau_g_r

                        # Diagnóstico rápido en el bucle principal:
            if int(t * 10) % 20 == 0:  # Imprime cada 2 segundos
                print(f"[Gripper L] q: {q_g_l:.4f} | Target: {grip_l_d:.4f}")
                print(f"[Gripper R] q: {q_g_r:.4f} | Target: {grip_r_d:.4f}\n")
            robot.send_torques(cmd_dict)
            robot.step()
    finally:
        robot.close()

if __name__ == "__main__":
    main()
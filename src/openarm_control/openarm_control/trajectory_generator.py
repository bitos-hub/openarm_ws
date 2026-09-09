import numpy as np

class CircleTrajectory:
    def __init__(self, radius=0.08, freq=0.3):
        self.r = radius
        self.w = 2 * np.pi * freq

    def get_ref(self, t, pos_init, R_init):
        phase = t * self.w

        x_front = 0.28
        z_lift  = 0.20

        pos_d = np.zeros(3)
        pos_d[0] = pos_init[0] + x_front
        pos_d[1] = pos_init[1] + self.r * np.cos(phase)
        pos_d[2] = pos_init[2] + z_lift + self.r * np.sin(phase)

        vel_d = np.array([
            0.0,
            -self.r * self.w * np.sin(phase),
            self.r * self.w * np.cos(phase)
        ])

        # Pitch de -90 grados sobre Y para orientar hacia +X sin Roll
        theta = -np.pi / 2.0
        R_pitch_y = np.array([
            [ np.cos(theta), 0.0, np.sin(theta)],
            [ 0.0,           1.0, 0.0          ],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ])

        R_d = R_pitch_y @ R_init

        gripper_d = 0.04 if (int(t / 2.0) % 2 == 0) else 0.00

        return pos_d, vel_d, R_d, gripper_d
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

class CircleTrajectory:
    def __init__(self, radius=0.08, freq=0.3, blend_time=4.0):
        self.r = radius
        self.w = 2 * np.pi * freq
        self.blend_time = blend_time # Tiempo en segundos para alcanzar la trayectoria suavemente

    def get_ref(self, t, pos_init, R_init):
        phase = t * self.w

        # 1. Factor de transición suave (0 a 1) usando un polinomio de 3er orden (Min-Jerk)
        alpha = np.clip(t / self.blend_time, 0.0, 1.0)
        alpha_smooth = 3 * (alpha ** 2) - 2 * (alpha ** 3)

        x_front = 0.10
        z_lift  = 0.20

        # 2. Calcular la trayectoria final del círculo
        pos_circle = np.zeros(3)
        pos_circle[0] = pos_init[0] + x_front
        # Se resta self.r para que el inicio del coseno (1) no cause un salto en Y
        pos_circle[1] = pos_init[1] + self.r * (np.cos(phase) - 1.0) 
        pos_circle[2] = pos_init[2] + z_lift + self.r * np.sin(phase)

        # 3. Interpolar la posición: de pos_init al círculo
        pos_d = pos_init * (1.0 - alpha_smooth) + pos_circle * alpha_smooth

        # Interpolar velocidades (0 al inicio, exactas después del blending)
        if t < self.blend_time:
            vel_d = np.zeros(3)
        else:
            vel_d = np.array([
                0.0,
                -self.r * self.w * np.sin(phase),
                self.r * self.w * np.cos(phase)
            ])

        # 4. Orientación Deseada Objetivo (Pitch de -90)
        theta = -np.pi / 2.0
        R_pitch_y = np.array([
            [ np.cos(theta), 0.0, np.sin(theta)],
            [ 0.0,           1.0, 0.0          ],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ])
        R_target = R_pitch_y @ R_init

        # 5. Interpolar Orientación suavemente usando SciPy (Rotación logarítmica)
        if alpha_smooth >= 1.0:
            R_d = R_target
        else:
            # Encuentra el eje-ángulo de diferencia y lo escala por alpha_smooth
            r_err = R_scipy.from_matrix(R_target @ R_init.T).as_rotvec()
            R_d = (R_scipy.from_rotvec(alpha_smooth * r_err) * R_scipy.from_matrix(R_init)).as_matrix()

        # Abrir/cerrar gripper de forma intermitente
        gripper_d = 0.04 if (int(t / 2.0) % 2 == 0) else 0.00

        return pos_d, vel_d, R_d, gripper_d
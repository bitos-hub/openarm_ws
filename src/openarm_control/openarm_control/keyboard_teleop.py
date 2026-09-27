import sys
import select
import tty
import termios
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

class KeyboardTeleop:
    def __init__(self):
        self.active_arm = 'left'
        self.active_axis = 'x'
        
        self.trans_dir = 0.0
        self.rot_dir = 0.0
        
        self.linear_speed = 0.05   # m/s
        self.angular_speed = 0.4   # rad/s
        
        self.last_key_time = 0.0
        self.running = True

        self.thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.thread.start()

        print("\n=== CONTROLADOR DE TECLADO ACTIVO ===")
        print("• Brazo:      '1' (Izquierdo) | '2' (Derecho)")
        print("• Eje:        'x' | 'y' | 'z'")
        print("• Traslación: Mantén 'i' (+) o 'k' (-)")
        print("• Rotación:   Mantén 'u' (Antihorario) o 'o' (Horario)")
        print("• Frenar:     Barra 'ESPACIO'\n")

    def _listen_keyboard(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1).lower()
                    self._process_key(key)
                else:
                    # Tolerancia de 0.6s para cubrir el retraso de repetición del SO
                    if time.time() - self.last_key_time > 0.6:
                        self.trans_dir = 0.0
                        self.rot_dir = 0.0
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _process_key(self, key):
        self.last_key_time = time.time()

        if key == '1':
            self.active_arm = 'left'
            print("[TECLADO] Brazo: IZQUIERDO   ", end='\r')
        elif key == '2':
            self.active_arm = 'right'
            print("[TECLADO] Brazo: DERECHO     ", end='\r')
        elif key in ['x', 'y', 'z']:
            self.active_axis = key
            print(f"[TECLADO] Eje Activo: {self.active_axis.upper()}   ", end='\r')
        elif key == 'i':
            self.trans_dir = 1.0
        elif key == 'k':
            self.trans_dir = -1.0
        elif key == 'u':
            self.rot_dir = 1.0
        elif key == 'o':
            self.rot_dir = -1.0
        elif key == ' ':
            self.trans_dir = 0.0
            self.rot_dir = 0.0

    def get_ref(self, dt, pos_curr, R_curr):
        v_local = np.zeros(3)
        w_local = np.zeros(3)

        if self.active_axis == 'x':
            v_local[0] = self.trans_dir * self.linear_speed
            w_local[0] = self.rot_dir * self.angular_speed
        elif self.active_axis == 'y':
            v_local[1] = self.trans_dir * self.linear_speed
            w_local[1] = self.rot_dir * self.angular_speed
        elif self.active_axis == 'z':
            v_local[2] = self.trans_dir * self.linear_speed
            w_local[2] = self.rot_dir * self.angular_speed

        v_world = R_curr @ v_local
        pos_d = pos_curr + v_world * dt

        delta_R = R_scipy.from_rotvec(w_local * dt).as_matrix()
        R_d = R_curr @ delta_R

        return pos_d, v_world, R_d, 0.0
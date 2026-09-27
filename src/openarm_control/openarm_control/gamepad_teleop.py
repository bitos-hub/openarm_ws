import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from scipy.spatial.transform import Rotation as R_scipy

class GamepadTeleop(Node):
    def __init__(self):
        super().__init__('gamepad_teleop_node')
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.active_arm = 'left' # Inicia controlando el brazo izquierdo
        
        # Velocidades (Traslación y Rotación)
        self.v_x, self.v_y, self.v_z = 0.0, 0.0, 0.0
        self.w_x, self.w_y, self.w_z = 0.0, 0.0, 0.0
        
        self.linear_speed = 0.05   # metros por segundo
        self.angular_speed = 0.5   # radianes por segundo
        
    def joy_callback(self, msg):
        # 1. Selección de brazo (D-Pad Izquierda/Derecha -> Eje 6)
        if msg.axes[6] > 0.5:
            self.active_arm = 'left'
        elif msg.axes[6] < -0.5:
            self.active_arm = 'right'
            
        # 2. Traslación XY (Joystick Izquierdo -> Ejes 1 y 0)
        self.v_x = msg.axes[1] # Arriba/Abajo
        self.v_y = msg.axes[0] # Izq/Der
        
        # 3. Traslación Z (D-Pad Arriba/Abajo -> Eje 7)
        self.v_z = msg.axes[7] 

        # 4. Rotación X e Y (Joystick Derecho -> Ejes 3 y 4)
        self.w_x = msg.axes[3] # Izq/Der
        self.w_y = msg.axes[4] # Arriba/Abajo

        # 5. Rotación Z (Botón Triángulo y X -> Índices 2 y 0)
        # Triángulo suma rotación, X (Cruz) resta rotación
        triangle_pressed = msg.buttons[2]
        x_pressed = msg.buttons[0]
        self.w_z = float(triangle_pressed - x_pressed)

    def get_ref(self, dt, pos_curr, R_curr):
        # Escalar comandos por las velocidades máximas
        v_local = np.array([self.v_x, self.v_y, self.v_z]) * self.linear_speed
        w_local = np.array([self.w_x, self.w_y, self.w_z]) * self.angular_speed
        
        # Rotar la velocidad lineal al marco global
        v_world = R_curr @ v_local
        
        # Integrar posición
        pos_d = pos_curr + v_world * dt
        
        # Integrar orientación (rotación local aplicada sobre la orientación actual)
        delta_R = R_scipy.from_rotvec(w_local * dt).as_matrix()
        R_d = R_curr @ delta_R
        
        gripper_d = 0.0 # Placeholder para los dedos
        
        return pos_d, v_world, R_d, gripper_d
# openarm_ws — Sistema Robótico Antropomórfico Bimanual para Ecografía Teleoperada

Espacio de trabajo en **ROS 2** y **MuJoCo** que contiene los paquetes de descripción cinemática/dinámica, interfaz de comunicación, simulación y control desarrollados para el sistema robótico antropomórfico bimanual de ecografía teleoperada.

> **Rama principal de evaluación:** `development` (Tag de tesis: `v1.0-tesis`)

---

## 1. Requisitos del Sistema

* **Sistema Operativo:** Ubuntu 22.04 LTS
* **Middleware Robótico:** ROS 2 (Humble)
* **Motor de Simulación Física:** MuJoCo (`mujoco` Python bindings)
* **Lenguajes:** Python 3.10+ / C++
* **Dependencias principales:** `numpy`, `scipy`, `matplotlib`, `ros2_control`, `xacro`, `rviz2`

---

## 2. Estructura del Workspace

```text
openarm_ws/
├── src/
│   ├── openarm_can/              # Interfaz de comunicación por bus CAN para actuadores
│   ├── openarm_control/          # Controlador híbrido, teleoperación y simulación
│   │   ├── effectors_cli.py            # Interfaz CLI para accionamiento de efectores
│   │   ├── hybrid_controller.py       # Controlador
│   │   ├── interactive_teleop_node.py# Nodo ROS 2 de teleoperación interactiva
│   │   ├── plot_errors.py            # Cáluclo de errores de las gráficas
│   │   ├── robot_interface.py        # Abstracción de estados, cinemática y comandos del robot
│   │   └── run_simulation.py         # Script principal de ejecución de simulación en MuJoCo
│   ├── openarm_description/      # Modelos URDF/Xacro y mallas del manipulador bimanual
│   ├── openarm_mujoco/           # Escenas XML, modelos físicos y sensores para MuJoCo
│   ├── openarm_ros2/             # Nodos de integración, controladores y archivos launch ROS 2
│   └── orcahand_description/     # Modelo URDF/mallas del efector antropomórfico OrcaHand
├── force_data_raw*.csv           # Registros experimentales de fuerza de contacto en simulación
├── force_plot*.png               # Gráficas de validación de seguimiento de fuerza
└── README.md

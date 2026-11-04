#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

# Values for the open and close positions
OPEN_POSITION = 0.035
CLOSE_POSITION = 0.015

# Range of the wrist joint
WRIST_MIN = -1.134
WRIST_MAX = 0.610

class HandControlCliNode(Node):
    def __init__(self):
        super().__init__('hand_control_cli_node')

        # Publishers
        self.pub_gripper = self.create_publisher(Float64, '/openarm/left/gripper_cmd', 10)
        self.pub_wrist   = self.create_publisher(Float64, '/openarm/right/wrist_cmd', 10)

    def send_gripper_cmd(self, value: float):
        """
        Publish the gripper command.
        """
        msg = Float64()
        msg.data = float(value)
        for _ in range(5):
            self.pub_gripper.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(f"Gripper cmd sent: {value:.3f} m")

    def send_wrist_cmd(self, angle_rad: float):
        """
        Publish the wrist command.
        """
        clamped_val = float(max(WRIST_MIN, min(WRIST_MAX, angle_rad)))
        msg = Float64()
        msg.data = clamped_val
        for _ in range(5):
            self.pub_wrist.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(f"Wrist cmd sent: {clamped_val:.3f} rad ({clamped_val*180/3.14159:.1f}°)")

def main(args=None):
    rclpy.init(args=args)
    node = HandControlCliNode()

    print("=====================================================")
    print("           CLI CONTROL: GRIPPER AND WRIST            ")
    print("=====================================================")
    print(" Left gripper commands:")
    print("   'o'       : Open gripper (0.035 m)")
    print("   'c'       : Close gripper (0.015 m)")
    print("-----------------------------------------------------")
    print(" Right wrist commands:")
    print("   'w <rad>' : Angle in radians [-1.134 a 0.610]")
    print("   'wd <deg>': Angle in degrees [-65.0 a 35.0]")
    print("   'w0'      : Center wrist (0.0 rad)")
    print("-----------------------------------------------------")
    print("   'q'       : Exit")
    print("=====================================================\n")

    try:
        while True:
            userInput = input("Cmd > ").strip().lower()
            if not userInput:
                continue
            if userInput == 'q':
                break
            elif userInput == 'o':
                node.send_gripper_cmd(OPEN_POSITION)
            elif userInput == 'c':
                node.send_gripper_cmd(CLOSE_POSITION)
            elif userInput == 'w0':
                node.send_wrist_cmd(0.0)
            elif userInput.startswith('wd '):
                try:
                    deg = float(userInput.split()[1])
                    node.send_wrist_cmd(deg * 3.14159265 / 180.0)
                except ValueError:
                    print("Not valid value.")
            elif userInput.startswith('w '):
                try:
                    rad = float(userInput.split()[1])
                    node.send_wrist_cmd(rad)
                except ValueError:
                    print("Not valid value.")
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
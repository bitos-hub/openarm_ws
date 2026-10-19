#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

# Gripper possible positions
OPEN_POSITION = 0.035
CLOSE_POSITION = 0.015

class GripperCliNode(Node):
    def __init__(self):
        super().__init__('gripper_cli_node')
        # Publisher for the gripper command
        self.pub = self.create_publisher(Float64, '/openarm/left/gripper_cmd', 10)

    def send_cmd(self, value: float):
        # Send a gripper command
        msg = Float64()
        msg.data = float(value)
        for _ in range(5):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info("Command sent")

def main(args=None):
    # Initialization
    rclpy.init(args=args)
    node = GripperCliNode()

    # Command Line Interface
    print("=============================================")
    print("                LEFT GRIPPER                 ")
    print("=============================================")
    print(" Commands:")
    print("   'o' : Open gripper (0.035 m)")
    print("   'c' : Close gripper (0.015 m)")
    print("   'q' : Quit")
    print("=============================================\n")

    try:
        while True:
            # Get commands
            userInput = input("Gripper Cmd > ").strip().lower()
            if userInput == 'q':
                break
            elif userInput == 'o':
                node.send_cmd(OPEN_POSITION)
            elif userInput == 'c':
                node.send_cmd(CLOSE_POSITION)
            else:
                print("Comando no válido.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
LAPTOP = False

import rclpy
import time
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool

if not LAPTOP:
    import jetank_arm.TTLServo
    from jetank_arm.TTLServo import servoAngleCtrl

class ServoController(Node):
    """
        This class controls the "wheels" of the Jetank. This includes driving and turning
    """
    def __init__(self, node_name="servo_controller", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)        
        for i in range(1, 6):
            setattr(self, f"s{i}_gain", 4)

        self.SetDefault()
        self.sub = self.create_subscription(Twist, 'ik_output', self.callback, 10)
        self.sub = self.create_subscription(Bool, 'arm_gripper', self.callback1, 10)

    def callback(self, msg):

        if(msg.angular.x == 0 and msg.angular.y == 0 and msg.angular.z == 0):
            self.SetDefault()
            return

        distance = math.sqrt(math.sqrt(msg.linear.x**2 + msg.linear.y**2)**2 + msg.linear.z**2)

        if(0 < distance <= 0.13):
            self.setAngle(5, -40, 300)

        time.sleep(0.2)

        self.setAngle(2, msg.angular.x - 90, 400)   # Set arm 1 into position
        self.setAngle(3, msg.angular.y - 90, 400)   # Set arm 2 into position
        self.setAngle(1, msg.angular.z, 400)        # Set base into position

        time.sleep(0.2)

        if(0 < distance > 0.13):
            self.setAngle(5, -10, 400)

        if(distance == 0):
            self.setAngle(5, -30, 400)

    def callback1(self, msg: Bool):
        self.setAngle(4, 90 if msg.data else 40, 400)

    def callback1(self, msg: Bool):
        self.setAngle(4, 90 if msg.data else 0, 200)

    
    def setAngle(self, index: int, angle: float, speed: int):
        if not LAPTOP:
            servoAngleCtrl(index, angle, 1, speed)
        else:
            self.get_logger().info(f"Index: {index}, angle: {angle}, speed: {speed}")

    def SetDefault(self):
        for i in range(1, 4):
            self.setAngle(i, 0, 100)
        self.setAngle(5, -10, 200)
        self.setAngle(4, 0, 200)

    def destroy_node(self):
        self.SetDefault()
        self.get_logger().info(f"shutting down, stopping servo arm ...")
        self.stop()


def main(args=None):
    rclpy.init(args=args)
    servo_node = ServoController()
    servo_node.get_logger().info("Waiting for messages to come ...")
    rclpy.spin(servo_node)
    servo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

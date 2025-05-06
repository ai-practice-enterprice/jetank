LAPTOP = True

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult

if not LAPTOP:
    import jetank_arm.TTLServo
    from jetank_arm.TTLServo import servoAngleCtrl

class ServoController(Node):
    """
        This class controls the "wheels" of the Jetank. This includes driving and turning
    """

    def __init__(self):
        super().__init__('servo_controller')
        for i in range(1, 6):
            setattr(self, f"s{i}_gain", 4)

        self.SetDefault()
        self.sub = self.create_subscription(Twist, 'ik_output', self.callback, 10)

    def callback(self, msg):
        self.setAngle(2, msg.angular.x - 90, 300)   # Set arm 1 into position
        self.setAngle(3, msg.angular.y - 90, 300)   # Set arm 2 into position
        self.setAngle(1, msg.angular.z, 300)        # Set base into position

    
    def setAngle(self, index: int, angle: float, speed: int):
        if not LAPTOP:
            servoAngleCtrl(index, angle, 1, speed)
        else:
            self.get_logger().info(f"Index: {index}, angle: {angle}, speed: {speed}")

    def SetDefault(self):
        for i in range(1, 6):
            self.setAngle(i, 0, 100)

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

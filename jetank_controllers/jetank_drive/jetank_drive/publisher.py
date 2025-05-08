import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MotorPublisher(Node):

    def __init__(self):
        super().__init__('motordriver_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):

        msg = Twist()

        msg.linear.x = float(self.i % 20)
        msg.angular.z = float(self.i % 360)

        self.publisher_.publish(msg)
        self.get_logger().info(f'speed: {msg.linear.x}')
        self.get_logger().info(f'angle: {msg.angular.z}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MotorPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

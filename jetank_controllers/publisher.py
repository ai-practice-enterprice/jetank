import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class MotorPublisher(Node):

    def __init__(self):
        super().__init__('motordriver_publisher')
        self.publisher_ = self.create_publisher(Twist, 'ik_topic', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):

        msg = Twist()

        msg.linear.x = self.i % 10
        msg.linear.y = self.i % 20
        msg.linear.z = self.i % 30

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MotorPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
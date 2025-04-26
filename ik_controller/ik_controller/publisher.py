import rclpy
import random
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class MotorPublisher(Node):

    def __init__(self):
        super().__init__('motordriver_publisher')
        self.publisher_ = self.create_publisher(Twist, 'ik_topic', 10)
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):

        msg = Twist()

        msg.linear.x = float(3)
        msg.linear.y = float(4)
        msg.linear.z = float(0)

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: x={msg.linear.x} y={msg.linear.y} z={msg.linear.z}')
        self.i+=1


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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class test_publisher(Node):
    """
        This class it a publisher to manually test a Twist subscriber
    """

    def __init__(self):
        super().__init__('ik_test')                                             # Init node
        self.publisher_ = self.create_publisher(Twist, 'ik_input', 10)          # Create publisher
        timer_period = 5                                                        # Repeat every x seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)       # Call the send function at timer interval

    def timer_callback(self):
        msg = Twist()                                                           # Init message

        msg.linear.x = 0.12                                                     # x
        msg.linear.y = 0.04                                                     # y
        msg.linear.z = 0.20                                                     # z

        self.publisher_.publish(msg)                                            # Publish
        self.get_logger().info(f'Published')                                    # Debug


def main(args=None):
    rclpy.init(args=args)
    publisher = test_publisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

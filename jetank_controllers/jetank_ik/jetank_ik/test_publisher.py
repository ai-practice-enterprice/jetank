import rclpy
import random
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math
import time

class test_publisher(Node):
    """
        This class it a publisher to manually test a Twist subscriber
    """

    def __init__(self):
        super().__init__('ik_test')                                             # Init node
        self.publisher_ = self.create_publisher(Twist, 'ik_input', 10)          # Create publisher
        self.pub = self.create_publisher(Bool, 'arm_gripper', 10)
        
        self.demo = 1

        self.timer_period = 2 if (self.demo in [2, 1]) else 5                        # Repeat every x seconds

        self.i = 0
        self.j = -1

        self.prev_x = 0.135
        self.prev_y = 0.01
        self.prev_z = -0.10

        self.timer_callback()

        self.get_logger().info("Publishing demo ...")

    def timer_callback(self):
        msg = Twist()                                                           # Init message
        msg1 = Bool()

        if self.demo == 2:

            if(self.j % 2):
                self.i = self.i % 6
                msg.linear.x = 0.135 if (self.i < 3) else 0.08
                msg.linear.z = -0.09

                y = self.i % 3

                if y == 0:
                    msg.linear.y = 0.0
                elif y == 1:
                    msg.linear.y = -0.06
                elif y == 2:
                    msg.linear.y = 0.07

                self.i += 1

            else:
                msg.linear.x, msg.linear.y, msg.linear.z = 0.0, 0.0, 0.0

            self.j += 1


        if self.demo == 1:

            while True:

                self.j += 1
                self.j = self.j % 7

                if(self.j in [0, 4]):
                    msg.linear.x = 0.08
                    msg.linear.y = 0.0
                    msg.linear.z = 0.12

                    self.publisher_.publish(msg)                                            # Publish
                    time.sleep(1.5)

                elif(self.j in [1, 6]):
                    msg1.data = True
                    self.pub.publish(msg1)
                    time.sleep(0.3)

                elif(self.j == 2):
                    msg.linear.x = self.prev_x
                    msg.linear.y = self.prev_y
                    msg.linear.z = self.prev_z

                    self.publisher_.publish(msg)
                    time.sleep(1.5)


                elif(self.j == 3):
                    msg1.data = False
                    self.pub.publish(msg1)
                    time.sleep(0.3)

                elif(self.j == 5):

                    self.i = random.randint(0, 5)

                    msg.linear.x = 0.22 if (self.i < 3) else 0.08
                    msg.linear.z = 0.05 if (self.i < 3) else -0.10

                    y = self.i % 3

                    if y == 0:
                        msg.linear.y = 0.0
                    elif y == 1:
                        msg.linear.y = -0.06 if (self.i > 3) else -0.12
                    elif y == 2:
                        msg.linear.y = 0.07 if (self.i > 3) else 0.12

                    self.prev_x, self.prev_y, self.prev_z = msg.linear.x, msg.linear.y, msg.linear.z

                    self.publisher_.publish(msg)                                            # Publish
                    time.sleep(1.5)


def main(args=None):
    rclpy.init(args=args)
    publisher = test_publisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

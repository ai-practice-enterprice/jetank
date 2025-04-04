import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import sin, cos, acos, atan2, pi, sqrt

base = [0, 0]
arm1_length = 25
arm2_length = 10

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.d = sqrt((self.x - base[0])**2 + (self.y - base[1])**2)

class ArmStuk:                                          # Class for arm joint
    def __init__(self, length):
        self.x = [0, 0]                                 # base x, end x
        self.y = [0, 0]                                 # base y, end y
        self.l = length                                 # length of arm
        self.a = 0

    def setBase(self, x, y):
        """
            This function sets the base coordinates of the arm joint
        :param x:
            x-coordinate
        :param y:
            y-coordinate
        """
        self.x[0] = x                                   # Set x-base
        self.y[0] = y                                   # Set y-base

    def setAngle(self, angle):
        """
            This function updates the x and y coordinates of the joint
        :param angle:
            The angle that is wanted
        """
        self.x[1] = cos(angle) * self.l + self.x[0]     # Set x-end
        self.y[1] = sin(angle) * self.l + self.y[0]     # Set y-end
        self.a = angle / pi * 180


class ik_solver(Node):
    def __init__(self):
        super().__init__("ik_solver")
        self.arm = [ArmStuk(arm1_length), ArmStuk(arm2_length)]
        self.sub = self.create_subscription(Twist, 'ik_topic', self.callback_function, 10)
        self.pub = self.create_publisher(Twist, 'ik_output', 10)

    def callback_function(self, msg):
        self.get_logger().info(f"received inputs: \n\
                               \t{msg.linear.x} \n\
                               \t{msg.linear.y} \n\
                               \t{msg.linear.z} \n\
                               \t{msg.angular.x} \n\
                               \t{msg.angular.y} \n\
                               \t{msg.angular.z} \n")
        
        th = Target(msg.linear.x, msg.linear.y)                                                                 # Create target (horizontal plane)
        tv = Target(th.d, msg.linear.z)                                                                         # Create target (vertical plane)

        self.get_logger().info(f"debug: {th.x, th.y, th.d} \n\
                               debug: {tv.x, tv.y, tv.d} \n")
        
        if self.is_in_range(th.d, tv.y):
            alpha = atan2(tv.y - base[1], tv.x - base[0])                                                           # Tangens = y/x
            beta = acos((tv.d**2 + self.arm[0].l**2 - self.arm[1].l**2) / (2 * tv.d * self.arm[0].l))               # Calcualte primary leg

            self.arm[0].setAngle(alpha + beta)                                                                      # Set angle of primary arm
            self.arm[1].setBase(self.arm[0].x[1], self.arm[0].y[1])                                                 # Set base of second arm

            gamma = acos((self.arm[0].l**2 + self.arm[1].l**2 - tv.d**2) / (2 * self.arm[0].l * self.arm[1].l))     # Calculate second arm
            self.arm[1].setAngle(alpha + beta + gamma)                                                              # Set angle of second arm

            msg.angular.x = self.arm[0].a
            msg.angular.y = self.arm[1].a
            msg.angular.z = atan2(th.y, th.x) * 180 / pi

        self.get_logger().info(f"Output: \n\
                               \t{msg.linear.x} \n\
                               \t{msg.linear.y} \n\
                               \t{msg.linear.z} \n\
                               \t{msg.angular.x} \n\
                               \t{msg.angular.y} \n\
                               \t{msg.angular.z} \n")
        
    def is_in_range(self, x, y):
        max_distance = self.arm[0].l                      # Start with the first leg
        min_distance = self.arm[0].l                      # Start with the first leg
        for part in self.arm[1:]:                         # For each other leg
            max_distance += part.l                          # Add it for maximum
            min_distance -= part.l                          # Subtract it for minimum

        x = x - base[0]                                     # relative distance: target to base (x)
        y = y - base[1]                                     # relative distance: target to base (x)
        distance = sqrt(x*x + y*y)                          # relative distance: target to base

        if(min_distance < distance <= max_distance):
            self.get_logger().info(f"Distance: {distance}")
        elif distance <= min_distance:
            self.get_logger().info(f"Distance too short to reach, need to move backwards a bit")
        elif max_distance < distance:
            self.get_logger().info(f"Distance too long to reach, need to move forwards")

        return min_distance < distance <= max_distance      # True if in range, false if not
        

def main(args=None):
    rclpy.init(args=args)
    servo_node = ik_solver()
    servo_node.get_logger().info("Waiting for messages to come ...")
    rclpy.spin(servo_node)
    servo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

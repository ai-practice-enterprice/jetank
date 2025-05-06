import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import sin, cos, acos, atan2, pi, sqrt

base = [0, 0]
arm1_length = 10
arm2_length = 16

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
        angle = angle / pi * 180
        self.a = angle


class ik_solver(Node):
    def __init__(self):
        super().__init__("ik_solver")
        self.arm = [ArmStuk(arm1_length), ArmStuk(arm2_length)]
        self.sub = self.create_subscription(Twist, 'ik_topic', self.callback_function, 10)
        self.pub = self.create_publisher(Twist, 'ik_output', 10)
        
    def inverse_kinematics(self, tv : Target, base, l1, l2):
        dx = tv.x - base[0]
        dy = tv.y - base[1]
    	
        alpha = atan2(dy, dx)
        c = sqrt(dx**2 + dy**2)
        beta = acos((l1**2 + c**2 - l2**2) / (2 * l1 * c))
        gamma = acos((l1**2 + l2**2 - c**2) / (2 * l1 * l2))
    	
        s1 = alpha + beta
        s2 = gamma
    	
        # self.get_logger().info(f'\n\tArm 1: {s1} \n\t Arm 2: {s2}')
    	
        return s1, s2

    def callback_function(self, msg):
        '''
        self.get_logger().info(f"received inputs: \n\
            \t{msg.linear.x} \n\
            \t{msg.linear.y} \n\
            \t{msg.linear.z} \n\
            \t{msg.angular.x} \n\
            \t{msg.angular.y} \n\
            \t{msg.angular.z} \n")
        '''
        
        th = Target(msg.linear.x, msg.linear.y)                                                                 # Create target (horizontal plane)
        tv = Target(th.d, msg.linear.z)                                                                         # Create target (vertical plane)
        
        if self.is_in_range(tv.x, tv.y):
            base = (0, 0)
            s1, s2 = self.inverse_kinematics(tv, base, self.arm[0].l, self.arm[1].l)
            msg.angular.x = s1 * 180 / pi                       # First leg
            msg.angular.y = s2 * 180 / pi                       # Second leg
            msg.angular.z = atan2(th.y, th.x) * 180 / pi        # Rotation of base

            self.get_logger().info(f"s1: {msg.angular.x:.2f}\ts2: {msg.angular.y:.2f}\ts3: {msg.angular.z:.2f}")
        
        '''
        self.get_logger().info(f"Output: \n\
            \t{msg.linear.x} \n\
            \t{msg.linear.y} \n\
            \t{msg.linear.z} \n\
            \t{msg.angular.x} \n\
            \t{msg.angular.y} \n\
            \t{msg.angular.z} \n")
        '''

        self.pub.publish(msg)
        
    def is_in_range(self, x, y):
        max_distance = self.arm[0].l                      # Start with the first leg
        min_distance = self.arm[0].l                      # Start with the first leg
        for part in self.arm[1:]:                         # For each other leg
            max_distance += part.l                          # Add it for maximum
            min_distance -= part.l                          # Subtract it for minimum

        min_distance = max(min_distance, 6)

        x = x - base[0]                                     # relative distance: target to base (x)
        y = y - base[1]                                     # relative distance: target to base (x)
        distance = sqrt(x*x + y*y)                          # relative distance: target to base

        if distance <= min_distance:
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

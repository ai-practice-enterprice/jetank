import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import acos, atan2, pi, sqrt

base = [0, 0]                               # Define base position (view point to center of arm)
arm1_length = 0.100                         # Lenght of the primary leg (in meter)
arm2_length = 0.160                         # Length of the secondary leg (in meter)

class Target:
    '''
        This class is a wrapper for a point in space.
        - As input it has a x and y coordinate
        - Additionally the distance to that point is calculated
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.d = sqrt((self.x)**2 + (self.y)**2)


class ik_solver(Node):
    def __init__(self, node_name="ik_solver", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        
        """
            In the init we create:
            - a listener for ik messages
            - a publisher to the arm
            - a publisher to the controller
        """
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        self.sub = self.create_subscription(Twist, 'ik_input', self.callback_function, 10)
        self.pub = self.create_publisher(Twist, 'ik_output', 10)
        # self.control = self.create_publisher(Twist, 'ik_control', 10)

    def inverse_kinematics(self, th : Target, tv : Target, l1, l2):
        """
            This function does the inverse kinematics:
            - given a point in space
            - return the angles of the servo's
        """
    	
        alpha = atan2(tv.y, tv.x)
        c = sqrt(tv.x**2 + tv.y**2)
        beta = acos((l1**2 + c**2 - l2**2) / (2 * l1 * c))
        gamma = acos((l1**2 + l2**2 - c**2) / (2 * l1 * l2))

        s1 = alpha + beta
        s2 = gamma
        s3 = atan2(th.y, th.x)

        return s1, s2, s3
    
    def callback_function(self, msg):
        """
            This function is called when a publisher sends a message to the topic which is subscribed here.
        """
        th = Target(msg.linear.x - base[0], msg.linear.y)       # Make target in horizontal plance
        tv = Target(th.d, msg.linear.z - base[1])               # Make target in vertical plane

        if msg.linear.x == 0 and msg.linear.y == 0 and msg.linear.z == 0:
            msg.angular.x = 90.0
            msg.angular.y = 90.0
            msg.angular.z = 0.0

        elif self.is_in_range(tv):                                # Check if target is in range
            s1, s2, s3 = self.inverse_kinematics(th, tv, arm1_length, arm2_length)    # Do the calculations
            msg.angular.x = s1 * 180 / pi                                           # Radians to degrees
            msg.angular.y = s2 * 180 / pi                                           # Radians to degrees
            msg.angular.z = s3 * 180 / pi                            # Radians to degrees
        
        self.pub.publish(msg)                                                   # Publish message
            # self.get_logger().info(f'Output published')                             # Debug

    def is_in_range(self, p: Target):
        """
            This function checks if a given target in space can be reached using this arm.
            - If successful, the arm grabs the object
            - If not, it returns a signal to the main controller
        """
        max_distance = arm1_length + arm2_length                # Maximum length is two arms spread
        min_distance = max(0.08, arm1_length - arm2_length)     # Minimum lenght is arms closed, but with a minimal distance of

        if p.d <= min_distance:                                         # If distance is shorter than minimum
            self.get_logger().info(f"Distance too short to reach")      # Print warning
            # self.control.publish()                                      # Send to main controller

        elif max_distance < p.d:                                        # If distance is longer than maximum
            self.get_logger().info(f"Distance too long to reach")       # Print warning
            # self.control.publish()                                      # Send to main controller

        return min_distance < p.d <= max_distance                       # True if in range, false if not


def main(args=None):
    rclpy.init(args=args)
    servo_node = ik_solver()
    servo_node.get_logger().info("Waiting for ik requests ...")
    rclpy.spin(servo_node)
    servo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

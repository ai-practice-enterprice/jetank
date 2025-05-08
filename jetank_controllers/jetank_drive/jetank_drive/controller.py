import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from Adafruit_MotorHAT import Adafruit_MotorHAT
from rcl_interfaces.msg import SetParametersResult

LAPTOP = False

class MotorController(Node):
    """
    Abstract motor controller base node for supporting different JetBots.
    Can be extended to support any diff drive by overriding set_speed(),
    or any node that subscribes to the /jetbot/cmd_vel Twist message.
    """
    def __init__(self, node_name="jetank_drive", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        
        self.sub = self.create_subscription(Twist, 'cmd_vel', self.twist_listener, 10)
        
        self.declare_parameter('left_trim', 0.0)
        self.declare_parameter('right_trim', 0.0)
        self.declare_parameter('max_pwm', 255)
        self.declare_parameter('max_rpm', 200)              # https://www.adafruit.com/product/3777
        self.declare_parameter('wheel_separation', 0.1016)  # 4 inches
        self.declare_parameter('wheel_diameter', 0.060325)  # 2 3/8 inches
        
        self.left_trim = self.get_parameter('left_trim').value
        self.right_trim = self.get_parameter('right_trim').value
        self.max_pwm = self.get_parameter('max_pwm').value
        self.max_rpm = self.get_parameter('max_rpm').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.wheel_diameter = self.get_parameter('wheel_diameter').value
        
        self.add_on_set_parameters_callback(self.parameters_callback)
         
        self.last_x = -999
        self.last_rot = -999
        
    def destroy_node(self):
        self.get_logger().info(f"shutting down, stopping robot...")
        self.stop()
        
    def parameters_callback(self, params):
        for param in params:
            if param.name == 'left_trim':
                self.left_trim = param.value
            elif param.name == 'right_trim':
                self.right_trim = param.value
            elif param.name == 'max_pwm':
                self.max_pwm = param.value
            elif param.name == 'wheel_separation':
                self.wheel_separation = param.value
            else:
                raise ValueError(f'unknown parameter {param.name}')
                
        return SetParametersResult(successful=True)
        
    def set_speed(self, left, right):
        """
        Sets the motor speeds between [-1.0, 1.0]
        Override this function for other motor controller setups.
        Should take into account left_trim, right_trim, and max_pwm.
        """
        raise NotImplementedError('MotorController subclasses should implement set_speed()')

    def stop(self):
        self.set_speed(0,0)

    def twist_listener(self, msg):
        x = msg.linear.x
        rot = msg.angular.z
        
        if x == self.last_x and rot == self.last_rot:
            return
            
        self.last_x = x
        self.last_rot = rot
        
        # https://github.com/ros-simulation/gazebo_ros_pkgs/blob/231a7219b36b8a6cdd100b59f66a3df2955df787/gazebo_plugins/src/gazebo_ros_diff_drive.cpp#L331
        left = x - rot * self.wheel_separation / 2.0
        right = x + rot * self.wheel_separation / 2.0
        
        max_speed = (self.max_rpm / 60.0) * 2.0 * math.pi * (self.wheel_diameter * 0.5)

        # convert velocities to [-1,1]
        left = max(min(left, max_speed), -max_speed) / max_speed
        right = max(min(right, max_speed), -max_speed) / max_speed
        
        self.get_logger().info(f"x={x:.03f} rotation={rot:.03f} -> left={left:.03f} right={right:.03f}")
        self.set_speed(left, right)


class MotorControllerWaveshare(MotorController):
    """
    Motor controller node that supports the Waveshare JetBot.
    @see motors.py for the base class to implement different controllers.
    """
    MOTOR_LEFT = 1      # left motor ID
    MOTOR_RIGHT = 2     # right motor ID
    
    def __init__(self):
        super().__init__()
        
        if LAPTOP:
            self.motors = {
            self.MOTOR_LEFT : 41,
            self.MOTOR_RIGHT : 38
            }

            self.pwm_channels = {
                self.MOTOR_LEFT : (1, 0),
                self.MOTOR_RIGHT : (2, 3)
            }

        else:
            # open Adafruit MotorHAT driver
            self.driver = Adafruit_MotorHAT(i2c_bus=1)
            
            # get motor objects from driver
            self.motors = {
                self.MOTOR_LEFT : self.driver.getMotor(self.MOTOR_LEFT),
                self.MOTOR_RIGHT : self.driver.getMotor(self.MOTOR_RIGHT)
            }
            
            self.pwm_channels = {
                self.MOTOR_LEFT : (1, 0),
                self.MOTOR_RIGHT : (2, 3)
            }


        self.get_logger().info("Initializing motors")


        
    def set_speed(self, left, right):
        """
        Sets the motor speeds between [-1.0, 1.0]
        """
        self._set_pwm(self.MOTOR_LEFT, left, self.left_trim)
        self._set_pwm(self.MOTOR_RIGHT, right, self.right_trim)
      
    def _set_pwm(self, motor, value, trim):
        # apply trim and convert [-1,1] to PWM value
        pwm = int(min(max((abs(value) + trim) * self.max_pwm, 0), self.max_pwm))

        if not LAPTOP:
            self.motors[motor].setSpeed(pwm)

        # set the motor direction
        ina, inb = self.pwm_channels[motor]

        if not LAPTOP:
            if value > 0:
                self.motors[motor].run(Adafruit_MotorHAT.FORWARD)
                self.driver._pwm.setPWM(ina, 0, pwm * 16)
                self.driver._pwm.setPWM(inb, 0, 0)
            elif value < 0:
                self.motors[motor].run(Adafruit_MotorHAT.BACKWARD)
                self.driver._pwm.setPWM(ina, 0, 0)
                self.driver._pwm.setPWM(inb, 0, pwm * 16)
            else:
                self.motors[motor].run(Adafruit_MotorHAT.RELEASE)
                self.driver._pwm.setPWM(ina, 0, 0)
                self.driver._pwm.setPWM(inb, 0, 0)

        self.get_logger().info("writing motor: " + str(motor) + "\tvalue: " + str(value) + "\ttrim: " + str(trim))
 

def main(args=None):
    rclpy.init(args=args)
    
    node = MotorControllerWaveshare()
    node.get_logger().info("listening for velocity messages...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
    
    node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()
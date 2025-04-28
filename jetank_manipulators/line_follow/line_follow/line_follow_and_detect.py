import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

# credits to:
# -----------
# https://github.com/gabrielnhn/ros2-line-follower/blob/main/follower/follower/follower_node.py
# https://github.com/Tinker-Twins/Autonomy-Science-And-Systems/blob/main/Assignment%203-B/assignment_3b/assignment_3b/assignment_3b/lane_following.py
# TJ's original LineDetectorNode

class LineFollower(Node):
    def __init__(self, node_name="LineFollower", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        
        self.bridge = CvBridge()

        # Robot state
        self.is_aligned = False
        self.is_following_line = False
        self.have_detected_red_dot = False
        self.have_detected_blue_dot = False

        # https://en.wikipedia.org/wiki/HSL_and_HSV
        # RED is a bit tricky due to range of the HSV color space 
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([80, 255, 255])

        self.lower_blue = np.array([100, 150, 0])
        self.upper_blue = np.array([140, 255, 255])

        self.lower_red1 = np.array([0, 150, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 150, 50])
        self.upper_red2 = np.array([180, 255, 255])
 
        ## User-defined parameters: (Update these values to your liking)
        # Minimum size for a contour to be considered anything
        self.MIN_AREA = 500 
        # Minimum size for a contour to be considered part of the track
        self.MIN_AREA_TRACK = 5000
        # Robot's speed when following the line
        self.LINEAR_SPEED = 0.2
        # Proportional constant to be applied on speed when turning 
        # (Multiplied by the error value)
        self.KP = 1.5/100 
        # If the line is completely lost, the error value shall be compensated by:
        self.LOSS_FACTOR = 1.2
        # Send messages every $TIMER_PERIOD seconds
        self.TIMER_PERIOD = 0.06
        # When about to end the track, move for ~$FINALIZATION_PERIOD more seconds
        self.FINALIZATION_PERIOD = 4
        # The maximum error value for which the robot is still in a straight line
        self.MAX_ERROR = 30
        # Linear velocity (linear.x in Twist) 
        self.LIN_VEL = 0.06
        # Angular velocity (angular.z in Twist)
        # self.ANG_VEL = 

        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            msg_type=Image,
            topic='camera/image_raw',
            callback=self.read_image_callback,
            qos_profile=10
        )
        
        # Publisher for processed image
        self.image_publisher = self.create_publisher(
            msg_type=Image,
            topic='detected_lines/result',
            qos_profile=10
        )
        
        # Publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(
            msg_type=Twist,
            topic='diff_drive_controller/cmd_vel_unstamped',
            qos_profile=10
        )

        # Callback to execute every X seconds to process a image/frame
        self.timer = self.create_timer(
            timer_period_sec=1/60,
            callback=self.process_image
        )

    # ---------------------- Helper functions ----------------- #
        
    def create_masks(self,hsv):
        # Masks
        # GREEN
        self.green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        # BLUE
        self.blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        # RED
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        self.red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    def crop_size(self,width,height):
        # (Height_upper_boundary, Height_lower_boundary,Width_left_boundary, Width_right_boundary)
        return (1*height//3, height, width//4, 3*width//4)

    # ---------------------- Callback functions ----------------- #
    def read_image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

    def process_image(self):
        width, height, channels = self.cv_image.shape
        crop = self.crop_size(height,width)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        self.create_masks(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mark = {}
        line = {}

        m = cv2.moments(mask, False) 
        try:
            # Calculate centroid of the blob using moments
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00'] 
        except ZeroDivisionError:
            cx, cy = width/2, height/2
        # Calculate error (deviation) from lane center
        error = (width/2 - cx + 10)/175
        try:
            # Process the image and get navigation command
            nav_command = self.detector.process(self.cv_image)
        except ValueError as e:
            self.get_logger().error(f'Detection Error: {e}')
            return
        
        # Publish the processed image
        output_image = self.detector.output_img
        if output_image is not None:
            output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
            self.image_publisher.publish(output_msg)

            cv2.imshow('Detected Lines', output_image)
            cv2.waitKey(1)
        self.navigate(nav_command)


    def navigate(self, nav_command):
        """Control the robot's movement based on line detection results"""
        # Initialize Twist message
        twist = Twist()
        
        # If no navigation command or no vertical line detected
        if nav_command is None or self.detector.chosen_vertical_line is None:
            self.get_logger().warn('No valid navigation command - stopping robot')
            self.cmd_vel_publisher.publish(twist)  # All zeros = stop
            self.is_aligned = False
            self.is_following_line = False
            return
        
        # Check alignment status
        self.is_aligned = nav_command['aligned']
        
        if not self.is_aligned:
            # Robot needs to align with the vertical line
            twist.angular.z = nav_command['angular_z']
            self.get_logger().info(f'Aligning to line: angular.z = {twist.angular.z}')
            self.is_following_line = False
        else:
            # Robot is aligned, start line following
            self.is_following_line = True
            
            # Simple line following - move forward at constant speed
            twist.linear.x = -self.detector.line_follow_speed
            self.get_logger().info(f'Following line: linear.x = {twist.linear.x}')
            
            # Check if we need to make slight adjustments to stay on the line
            x_diff = self.detector.chosen_vertical_line['center_x'] - self.detector.image_center_x
            
            # Apply a proportional correction to stay centered on the line
            kp = 0.005  # Proportional gain - adjust as needed
            twist.angular.z = -kp * x_diff
        
        # Publish command
        self.cmd_vel_publisher.publish(twist)



def main(args=None):
    rclpy.init(args=args)
    line_detector_node = LineFollower()
    try:
        rclpy.spin(line_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutting down
        stop_cmd = Twist()
        line_detector_node.cmd_vel_publisher.publish(stop_cmd)
        
        line_detector_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
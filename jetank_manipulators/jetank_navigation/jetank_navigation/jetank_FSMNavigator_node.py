import cv2
import numpy as np
import rclpy
from rclpy.node import Node , ParameterDescriptor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory
from cv_bridge import CvBridge, CvBridgeError
from enum import Enum
import random
import heapq
import time

# references from:
# ----------------
# https://automaticaddison.com/how-to-detect-and-draw-contours-in-images-using-opencv/
# https://github.com/gabrielnhn/ros2-line-follower/blob/main/follower/follower/follower_node.py
# https://github.com/Tinker-Twins/Autonomy-Science-And-Systems/blob/main/Assignment%203-B/assignment_3b/assignment_3b/assignment_3b/lane_following.py
# https://www.instructables.com/OpenCV-Based-Line-Following-Robot/
# https://const-toporov.medium.com/line-following-robot-with-opencv-and-contour-based-approach-417b90f2c298
# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
# https://www.waveshare.com/wiki/21_Line_Following_Autonomous_Driving_with_OpenCV

# credits to:
# -----------
# TJ's original LineDetectorNode

class JetankState(Enum):
    INITIALIZE = 1	
    FOLLOW_LINE = 2	
    DOT_DETECTED = 3
    TURN_ON_SPOT = 4
    DRIVE_FORWARD = 5
    DECIDE_DIRECTION = 6
    PICK_UP_PACKAGE = 7
    DESTINATION_REACHED = 8
    IDLE = 9

class DotType(Enum):
    RED = 1
    BLUE = 2

class Turning(Enum):
    CLOCK_WISE = 1
    ANTI_CLOCK_WISE = -1

class Directions(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    # https://stackoverflow.com/questions/37183612/how-to-define-a-mapping-of-enum-members-in-the-enum-type
    __MAPPING__ = {
        0 : NORTH,
        1 : EAST,
        2 : SOUTH,
        3 : WEST,
    }

    @classmethod
    def from_value(cls, value):
        if value in cls.__MAPPING__:
            return cls.__MAPPING__[value]
        else:
            raise ValueError(f"No Direction found for value: {value}")

class ZoneTypes(Enum):
    VOID = 1
    ROBOT_STATION = 2
    STORAGE = 3
    ZONE_IN = 4
    ERROR_ZONE = 5
    NORMAL = 6
    ZONE_OUT = 7
    ADD_ZONE = 8
    # additional zones
    REAL_ZONE = 9
    BLOCKED = 10


class FSMNavigator(Node):
    def __init__(self, node_name="FSMNavigator", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        self.get_logger().info(f"--- booting up {self.get_name()} ---")
        
        self.bridge = CvBridge()
        self.cv_image = None

        # --------------------------- -------------------- --------------------------- #
        # --------------------------- state send by Server --------------------------- # 
        # --------------------------- -------------------- --------------------------- #
        self.goal_position = (1, 2)
        self.start_position = (0,0) 
        self.current_position = (0, 0)
        self.direction = Directions.SOUTH
        self.map = [
            [9,9,9,9,9,],
            [9,9,9,9,9,],
            [9,9,9,9,9,],
            [9,9,9,9,9,],
            [9,9,9,9,9,],
        ]

        # --------------------------- ----------- --------------------------- #
        # --------------------------- Robot state --------------------------- # 
        # --------------------------- ----------- --------------------------- #
        self.path_plan = []
        self.prev_cx = None
        self.prev_cy = None
        self.prev_direction = None

        self.jetank_state = JetankState.INITIALIZE
        self.prev_jetank_state = JetankState.INITIALIZE

        self.dot_color_detected = None
        self.next_dot_color = None
        self.dot_disseappered_at_time = 0
        self.dead_reckoning_active = False
        self.DEAD_RECKONING_THRESHOLD = 4.0

        self.next_position = self.current_position

        self.direction_turn = Turning.CLOCK_WISE

        self.alpha_smoother = 0.3 
        
        ## user-defined parameters:
        # Minimum size for a contour to be considered anything
        self.MIN_AREA = 150 
        # Proportional constant to be applied on speed when turning 
        # (Multiplied by the error value)
        self.KP = 1.1/100 
        # Send messages every X seconds
        self.TIMER_PERIOD = 1/30
        # The maximum error value for which the robot is still in a straight line
        self.MAX_ERROR = 30
        # The maximum error value for which the robot is aligned when turning
        self.MAX_ALIGNMENT_ERROR = 15
        # Linear velocity (linear.x in Twist) 
        self.declare_parameter(
            name="LIN_VEL",
            value=0.15,
            ignore_override=False,
            descriptor=ParameterDescriptor(
                type="float",
                description="Sets the std linear velocity of the Jetank",
                name="LIN_VEL"
            )
        )
        self.LIN_VEL = self.get_parameter("LIN_VEL").value
        self.current_lin_vel = 0
        # Angular velocity (angular.z in Twist)
        self.ANG_VEL = 0.9
        self.current_ang_vel = 0 

        self.COUNTER_THRESHOLD = self.TIMER_PERIOD*400
        self.STATE_COUNTER = 0
        
        self.CONFIDENCE_THRESHOLD = 5
        self.CONFIDENCE_COUNTER = 0

        # https://en.wikipedia.org/wiki/HSL_and_HSV
        # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        # RED is a bit tricky due to range of the HSV color space 
        # also HSV is defined as range from (0deg, 0%, 0%) to (360deg , 100% , 100%)
        # but because OpenCV wants to keep 8bit unsigned integers 
        # 360deg is mapped to 180 (x deg => x/2) 
        # 100% is mapped to 255   (x %   => x/100 * 255)
        self.lower_green = np.array([40, 70, 50])
        self.upper_green = np.array([80, 255, 255])

        self.lower_blue = np.array([110, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([5, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([179, 255, 255])
 
        # --------------------------- ------------------------------------- --------------------------- #
        # --------------------------- subscriptions , publishers and timers --------------------------- # 
        # --------------------------- ------------------------------------- --------------------------- #

        # subscribeer to the camera feed 
        self.subscription = self.create_subscription(
            msg_type=Image,
            topic='camera/image_raw',
            callback=self.read_image_callback,
            qos_profile=10
        )

        # publisher for steering
        self.cmd_vel_publisher = self.create_publisher(
            msg_type=Twist,
            topic='diff_drive_controller/cmd_vel_unstamped',
            qos_profile=10
        )

        # publisher for processed image for visualization purposes
        self.image_publisher = self.create_publisher(
            msg_type=Image,
            topic='detection/result',
            qos_profile=10
        )

        # for debugging
        self.mask_publisher_green = self.create_publisher(
            msg_type=Image,
            topic='detection/result/green_mask',
            qos_profile=10
        )

        self.mask_publisher_red = self.create_publisher(
            msg_type=Image,
            topic='detection/result/red_mask',
            qos_profile=10
        )

        self.mask_publisher_blue = self.create_publisher(
            msg_type=Image,
            topic='detection/result/blue_mask',
            qos_profile=10
        )
        
        # publisher for robot movement commands => ros2_control controller
        self.cmd_vel_publisher = self.create_publisher(
            msg_type=Twist,
            topic='diff_drive_controller/cmd_vel_unstamped',
            qos_profile=10
        )

        self.arm_traj_publisher = self.create_publisher(
            msg_type=JointTrajectory,
            topic='arm_controller/joint_trajectory',
            qos_profile=10
        )

        self.ik_arm_publisher = self.create_publisher(
            msg_type=Twist,
            topic='ik_topic',
            qos_profile=10
        )

        self.main_loop = self.create_timer(
            timer_period_sec=self.TIMER_PERIOD,
            callback=self.FSMloop
        )

        self.cmd_vel_loop = self.create_timer(
            timer_period_sec=self.TIMER_PERIOD,
            callback=self.publish_cmd_vel
        )

        self.get_logger().info(f"--- booting up complete ---")

    # ---------------------- ------------------ ----------------- #
    # ---------------------- Steering functions ----------------- #
    # ---------------------- ------------------ ----------------- #
    
    def turn_on_spot(self,direction):
        self.current_lin_vel = 0
        self.current_ang_vel = -1 * direction * self.ANG_VEL

    def drive_straight(self):
        self.current_lin_vel = self.LIN_VEL
        self.current_ang_vel = 0

    def stop_moving(self):
        self.current_lin_vel = 0
        self.current_ang_vel = 0

    def drive_towards_center(self, error):
        self.current_lin_vel = self.LIN_VEL
        self.current_ang_vel = self.KP * error

    def publish_cmd_vel(self):
        msg = Twist()
        msg.linear.x = float(-1 * self.current_lin_vel)
        msg.angular.z = float(-1 * self.current_ang_vel)
        self.cmd_vel_publisher.publish(msg)

    # ---------------------- ------------- ----------------- #
    # ---------------------- Arm functions ----------------- #
    # ---------------------- ------------- ----------------- #

    def publish_arm_trajectory(self,points = "pickup"):
        joints = ['turn_ARM', 'SERVO_LOWER_', 'SERVO_UPPER_']

        if points == "pickup": points_traj = [0,1,0]
        elif points == "normal": points_traj = [0,1,0]
        elif points == "pickup": points_traj = [0,1,0]
        else :  points_traj = [0,1,0]
        
        msg = JointTrajectory()

        for joint in joints:
            msg.joint_names.append(joint)

        for point in points_traj:
            msg.points.append(point)

        self.arm_traj_publisher.publish(msg)

    def publish_arm_ik(self):        
        msg = Twist()
        self.ik_arm_publisher.publish(msg)

    # ---------------------- ---------------- ----------------- #
    # ---------------------- Helper functions ----------------- #
    # ---------------------- ---------------- ----------------- #

    def notify_server(self):
        # we can add more logic in here to allow distress signals 
        # or a new map request or smt
        # or pictures (QR codes)
        self.get_logger().info(f"{self.get_namespace()} Arrived at destination")
        self.jetank_state = JetankState.INITIALIZE

    def update_position(self,init: bool=False):
        current_index = self.path.index(self.current_position)
        if current_index + 1 >= len(self.path): 
            self.jetank_state = JetankState.DESTINATION_REACHED
        else: 
            if init:
                self.next_position = self.path[current_index + 1]
            else: 
                self.current_position = self.path[current_index + 1]

                if self.current_position == self.goal_position: 
                    self.get_logger().info(f"{self.get_namespace()} Arrived at destination")
                    self.jetank_state = JetankState.DESTINATION_REACHED
                else:
                    self.next_position = self.path[current_index + 2]
                
            self.get_logger().info(f'Updating : current position {self.current_position}')
            self.get_logger().info(f'Updating : next position {self.next_position}')

    def update_direction(self):
        if self.direction_turn == Turning.CLOCK_WISE:
            new_direction_value = (self.direction.value + self.direction_turn.value) % len(Directions.__MAPPING__.values())
                        
        elif self.direction_turn == Turning.ANTI_CLOCK_WISE:
            new_direction_value = (self.direction.value + self.direction_turn.value) % len(Directions.__MAPPING__.values())

        if   new_direction_value == Directions.NORTH.value : self.direction = Directions.NORTH
        elif new_direction_value == Directions.EAST.value  : self.direction = Directions.EAST
        elif new_direction_value == Directions.SOUTH.value : self.direction = Directions.SOUTH
        elif new_direction_value == Directions.WEST.value  : self.direction = Directions.WEST

    def calculate_direction(self):
        # this is just a function to see where you have to go to from the map's perspective
        a = self.current_position
        b = self.next_position
        if a[0] < b[0]:
            return Directions.EAST
        elif a[0] > b[0]:
            return Directions.WEST
        elif a[1] < b[1]:
            return Directions.SOUTH 
        elif a[1] > b[1]:
            return Directions.NORTH
        return -1
    
    def calculate_turn(self,target_direction: Directions):
        total_directions = len(Directions.__MAPPING__.values())  # 4 (N, E, S, W)
        current = self.direction.value
        target_direction = target_direction.value
        # turning from the robots perspective involves representing the 4 possible directions
        # we can move to, as numbers ([N E S W] => [0 1 2 3])and then calculate the distance between the numbers
        # compute the shortest turn direction
        clockwise_steps = (target_direction - current) % total_directions
        anti_clockwise_steps = (current - target_direction) % total_directions

        # 180 deg or -180 deg till next direction (doesn't matter if clock or anti-clock wise turn)
        if clockwise_steps == anti_clockwise_steps:  # 180° turn
            if random.random() > 0.5:
                return Turning.CLOCK_WISE
            else:
                return Turning.ANTI_CLOCK_WISE

        # 90 deg or -90 deg till next direction (does matter if clock or anti-clock wise turn if we want the most optimal turn)
        elif clockwise_steps < anti_clockwise_steps:
            return Turning.CLOCK_WISE
        else:
            return Turning.ANTI_CLOCK_WISE
     
    def heuristic(self,a, b):
        # Manhattan distance => sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def neighbors(self,pos):
        x, y = pos
        # E, S, W, N
        moves = [
            (1,0),
            (0,1), 
            (-1,0), 
            (0,-1)
        ]  
        results = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.map[0]) and 0 <= ny < len(self.map):
                if self.map[ny][nx] != ZoneTypes.VOID and self.map[ny][nx] != ZoneTypes.BLOCKED:
                    results.append((nx, ny))
        return results
    
    # https://www.geeksforgeeks.org/a-search-algorithm-in-python/
    # https://www.datacamp.com/tutorial/a-star-algorithm
    def a_star(self,recalculating = False):
        # if a recalculation occurs during moving from 1 point to a another point
        # we can't forget to reset the starting position
        if recalculating:
            self.start_position = self.current_position

        # open list     // Nodes to be evaluated
        open_list = []

        # closed list   // Nodes already evaluated
        cost_so_far = {self.start_position: 0}

        # for faster retrievel of the next tile with the lowest => priority queue
        # https://www.geeksforgeeks.org/priority-queue-set-1-introduction/
        heapq.heappush(open_list, (0, self.start_position))

        # for reconstruction purposes
        came_from = {self.start_position: None}

        while open_list:
            # selects the most promising position (that is why we use a priority queue)
            _, current = heapq.heappop(open_list)
            if current == self.goal_position:
                break

            # examining all neighboring positions
            for next_pos in self.neighbors(current):
                # the cost of the next position to be evaluated
                # is equal to the previous cost from the last tile visted +1 
                new_cost = cost_so_far[current] + 1
                # skip positions already in the closed list (do not revist) 
                # or if it is visted it must be at a better
                # cost the previously set
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    # update (or insert into) the closed list 
                    cost_so_far[next_pos] = new_cost
                    # the next position needs to be added to the open list ready to be evaluated
                    priority = new_cost + self.heuristic(self.goal_position, next_pos)
                    heapq.heappush(open_list, (priority, next_pos))

                    came_from[next_pos] = current

        # Reconstruct path only if a path was found and valid
        if len(came_from) == 1 or self.start_position not in came_from.values() or self.goal_position not in came_from.keys():
            return False

        # once the goal_pos is reached, the algorithm works backward 
        # through the parent references to construct the optimal path from start_pos to goal_pos.
        path = []
        current = self.goal_position
        while current != self.start_position:
            path.append(current)
            current = came_from[current]
        path.append(self.start_position)
        path.reverse()

        self.path = path

        return True
    
    def listen_to_server(self):
        # TODO => request from Zenoh topic instead from the server
        self.goal_position = (0,0)
        while self.goal_position == self.current_position: 
            self.goal_position = (random.randint(0,len(self.map[0])),random.randint(0,len(self.map)))
        
        self.get_logger().info(f"SERVER sending {self.get_namespace()} to destination {self.goal_position}")

    def smoother(self,cx,cy): 
        # https://en.wikipedia.org/wiki/Moving_average
        # smoothing factor (0 = very smooth, 1 = instant)
        if self.prev_cx is None or self.prev_cy is None:
            self.prev_cx = cx
            self.prev_cy = cy
        else:
            self.prev_cy = int((1 - self.alpha_smoother) * self.prev_cy + self.alpha_smoother * cy)
            self.prev_cx = int((1 - self.alpha_smoother) * self.prev_cx + self.alpha_smoother * cx)

        return self.prev_cx , self.prev_cy
    
    # ---------------------- ---------------------------- ----------------- #
    # ---------------------- Image manipulation functions ----------------- #
    # ---------------------- ---------------------------- ----------------- #

    def crop_to_roi(self):
        # cropping the image to the ROI (region of intrest)
        # (Height_upper_boundary, Height_lower_boundary,Width_left_boundary, Width_right_boundary)
        height , width , channels = self.cv_image.shape
        roi = self.cv_image[
            int(height/2 + 100):int(height),
            int(width/5):int(4*width/5)
        ]
        return roi

    def morph_filter(self,mask):
        # https://cyrillugod.medium.com/filtering-and-morphological-operations-990662c5bd59
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        # this basically a convulution but without any complicated kernel 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel,iterations=1)
        mask = cv2.dilate(mask, kernel,iterations=1)
        return mask

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

    def try_detect_dots(self):
        to_examine = [ 
            DotType.RED,
            DotType.BLUE,
        ]
        for dot_type in to_examine:
            mask = self.red_mask 
            if dot_type == DotType.RED:
                mask = self.red_mask                 
            elif dot_type == DotType.BLUE:
                mask = self.blue_mask
            else:
                return False

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > self.MIN_AREA:
                    # if a dot is detected we keep track what dot color it 
                    # was and switch to a dot detected state as following a line
                    # is not our priority anymore 
                    self.dot_color_detected = dot_type
                    return True
        return False

    def detect_and_center(self,roi):
        # the image is processed according to the state we're in. changing from state to state 
        # is based upon a FSM.
        # (https://en.wikipedia.org/wiki/Finite-state_machine)
        # (https://www.spiceworks.com/tech/tech-general/articles/what-is-fsm/)
        # (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
        
        if self.jetank_state == JetankState.DOT_DETECTED:
            if self.dot_color_detected == DotType.RED:
                contours, _ = cv2.findContours(self.red_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour_clr = (0,0,255)
            elif self.dot_color_detected == DotType.BLUE:
                contours, _ = cv2.findContours(self.blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contour_clr = (255,0,0)
            else:
                return False
            
        elif self.jetank_state == JetankState.FOLLOW_LINE:
            contours, _ = cv2.findContours(self.green_mask,1, cv2.CHAIN_APPROX_NONE)
            contour_clr = (0,255,0)

        else:
            return False

        if len(contours) > 0 :
            cnt = max(contours,key=cv2.contourArea)

            M = cv2.moments(cnt)
            height , width , _ = roi.shape
            
            try:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            except ZeroDivisionError:
                cx, cy = int(width/2), int(height/2)

            smoothed_cx , smoothed_cy = self.smoother(cx,cy)      
            
            # direction logic
            image_center_x = width // 2
            self.error = 0
            
            # the smoothed_cx has to fall inside the interval : [image_center_x - self.MAX_ERROR ;image_center_x - self.MAX_ERROR]
            # if not we calculate the error margin from the center
            if smoothed_cx < image_center_x - self.MAX_ERROR or image_center_x + self.MAX_ERROR < smoothed_cx:
                self.error = image_center_x - smoothed_cx

            # annotations
            cv2.circle(roi,(smoothed_cx,smoothed_cy),5,contour_clr,2)
            cv2.drawContours(roi, [cnt], -1, contour_clr, 1)
            return True
        return False

    def try_realign_after_turn(self, roi):
        contours, _ = cv2.findContours(self.green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > self.MIN_AREA:
                M = cv2.moments(cnt)
                height, width, _ = roi.shape
                try:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                except ZeroDivisionError:
                    # can't calculate center
                    return False  

                smoothed_cx , smoothed_cy = self.smoother(cx,cy)  
                image_center_x = width // 2

                # check if green line is centered (within margin) on the right side
                # depending on the turn direction
                if self.direction_turn == Turning.CLOCK_WISE:
                    if image_center_x < smoothed_cx and abs(smoothed_cx - image_center_x) < self.MAX_ALIGNMENT_ERROR:
                        cv2.circle(roi, (smoothed_cx, smoothed_cy), 5, (0, 255, 0), 2)
                        # ready to go  
                        return True  
                    
                elif self.direction_turn == Turning.ANTI_CLOCK_WISE:
                    if image_center_x > smoothed_cx and abs(smoothed_cx - image_center_x) < self.MAX_ALIGNMENT_ERROR:
                        cv2.circle(roi, (smoothed_cx, smoothed_cy), 5, (0, 255, 0), 2)
                        # ready to go  
                        return True 
                    
                return False 
        # can't calculate center
        return False  
    
    def process_image(self,roi):
        if self.cv_image is None:
            return

        # we change the format of the image and create 
        # "masks" which are copies of the cropped images but filtered based upon a predefined color range
        # and then converted to black white image
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.create_masks(roi_hsv)
        

        # once we have created our different mask we can process these masks
        # based upon our state we're in.
        if self.jetank_state == JetankState.TURN_ON_SPOT:
            realigned = self.try_realign_after_turn(roi)
            if realigned:
                self.jetank_state = JetankState.DECIDE_DIRECTION
                self.get_logger().info("Realigned with green line. Switching to DECIDE_DIRECTION.")
        
        elif self.jetank_state == JetankState.FOLLOW_LINE:
            self.detect_and_center(roi)
            dot_detected = self.try_detect_dots()
            if dot_detected:
                self.jetank_state = JetankState.DOT_DETECTED
                self.get_logger().info("Dot detected. Switching to DOT_DETECTED.")

        elif self.jetank_state == JetankState.DOT_DETECTED:
            self.detect_and_center(roi)
            dot_detected = self.try_detect_dots()

            if not dot_detected and not self.dead_reckoning_active:
                self.dot_disseappered_at_time = time.time()
                self.dead_reckoning_active = True

            elif self.dead_reckoning_active:
                time_diff = time.time() - self.dot_disseappered_at_time
                # using a dead reckoning approach (found no other solution)
                # https://www.cavliwireless.com/blog/not-mini/what-is-dead-reckoning
                if time_diff < self.DEAD_RECKONING_THRESHOLD:
                    self.dead_reckoning_active = False
                    self.dot_disseappered_at_time = 0
                    self.jetank_state = JetankState.DECIDE_DIRECTION
                    self.get_logger().info("Dot disappeared. Switching to DECIDE_DIRECTION.")
                else:
                    self.get_logger().info(f"Waiting... {time_diff:.2f}s")

        # for debugging purposes (can be removed)
        self.mask_publisher_green.publish(
            self.bridge.cv2_to_imgmsg(self.green_mask)
        )

        self.mask_publisher_blue.publish(
            self.bridge.cv2_to_imgmsg(self.blue_mask)
        )
        self.mask_publisher_red.publish(
            self.bridge.cv2_to_imgmsg(self.red_mask)
        )

        # publish the raw image with some anottations on it if any 
        # mostly for debugging purposes (can be removed but not recommended)
        self.image_publisher.publish(
            self.bridge.cv2_to_imgmsg(roi,encoding='rgb8')
        )


    # ---------------------- ------------------ ----------------- #
    # ---------------------- Callback functions ----------------- #
    # ---------------------- ------------------ ----------------- #

    def read_image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg)
            # 1) we crop our image to the desired space
            roi = self.crop_to_roi()
            self.process_image(roi)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return
        
    def FSMloop(self):
        # 1)
        # [INITIALIZE] → [FOLLOW_LINE]

        # 2)
        # [FOLLOW_LINE] --> dot detected --> [DOT_DETECTED]
        # [DOT_DETECTED] --> [DECIDE_DIRECTION]
        # [DECIDE_DIRECTION] --> [TURNING] or [DRIVE_FORWARD]

        # 3)
        # [TURNING] --> [FOLLOW_LINE]
        # [DRIVE_FORWARD] --> [FOLLOW_LINE]

        # 4) 
        # [FOLLOW_LINE] --> destination matched --> [DESTINATION_REACHED]

        # TODO : Obstacle detected => set point as BLOCKED or VOID => recalculate route
        # TODO : Map saturated => request new map from the server
        # TODO : Map saturated => request new map from the server

        # ----------- state to state conditions ----------- #
        if self.prev_jetank_state == JetankState.DOT_DETECTED and self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.update_position()
            self.stop_moving()

        if self.prev_jetank_state == JetankState.TURN_ON_SPOT and self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.update_direction()
            self.stop_moving()

        # ----------- state logger ----------- #
        if self.prev_jetank_state is not self.jetank_state:
            self.get_logger().info(f'--- STATE : {self.jetank_state} ---')
            self.prev_jetank_state = self.jetank_state

        # ----------- state condition and logic ----------- #
        if self.jetank_state == JetankState.INITIALIZE:
            self.listen_to_server()
            if not self.a_star():
                self.get_logger().error(f'NO available path')
                self.jetank_state = JetankState.INITIALIZE
            else:
                self.update_position(init=True)
                self.get_logger().info(f'PATH : {self.path}')
                self.jetank_state = JetankState.DECIDE_DIRECTION

        elif self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.goal_direction = self.calculate_direction()
            self.get_logger().info(f'Goal direction : {self.goal_direction}')
            self.get_logger().info(f'Current direction : {self.direction}')
            if self.goal_direction == -1: 
                self.get_logger().error(f'Invalid direction {self.goal_direction}')

            elif self.direction != self.goal_direction:
                self.direction_turn = self.calculate_turn(self.goal_direction)
                self.get_logger().info(f'Turning : {self.direction_turn}')
                self.jetank_state = JetankState.TURN_ON_SPOT
            
            elif self.direction == self.goal_direction:
                self.jetank_state = JetankState.DRIVE_FORWARD

        elif self.jetank_state == JetankState.FOLLOW_LINE:
            self.drive_towards_center(self.error)

        elif self.jetank_state == JetankState.DOT_DETECTED:
            self.drive_towards_center(self.error)
            
        elif self.jetank_state == JetankState.TURN_ON_SPOT:
            self.turn_on_spot(self.direction_turn.value)

        elif self.jetank_state == JetankState.DRIVE_FORWARD:
            self.drive_straight()
            self.jetank_state = JetankState.FOLLOW_LINE

        elif self.jetank_state == JetankState.DESTINATION_REACHED:
            self.notify_server()
            self.stop_moving()


def main(args=None):
    rclpy.init(args=args)
    navigator_node = FSMNavigator()
    try:
        rclpy.spin(navigator_node)
    except KeyboardInterrupt:
        pass        
        navigator_node.stop_moving()
    navigator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
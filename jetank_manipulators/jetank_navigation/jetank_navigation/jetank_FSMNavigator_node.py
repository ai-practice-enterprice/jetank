import cv2
import numpy as np
import rclpy
from rclpy.node import Node , ParameterDescriptor
from cv_bridge import CvBridge, CvBridgeError
from enum import Enum
import random
import heapq
import time
import copy
import json
import typing

# from jetank_custom_msgs.msg import NotificationServer
from std_msgs.msg import Int16 , Bool , Int16MultiArray , String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist , Point
from trajectory_msgs.msg import JointTrajectory



# references from:
# ----------------
# https://automaticaddison.com/how-to-detect-and-draw-contours-in-images-using-opencv/
# https://github.com/gabrielnhn/ros2-line-follower/blob/main/follower/follower/follower_node.py
# https://github.com/Tinker-Twins/Autonomy-Science-And-Systems/blob/main/Assignment%203-B/assignment_3b/assignment_3b/assignment_3b/lane_following.py
# https://www.instructables.com/OpenCV-Based-Line-Following-Robot/
# https://const-toporov.medium.com/line-following-robot-with-opencv-and-contour-based-approach-417b90f2c298
# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
# https://www.waveshare.com/wiki/21_Line_Following_Autonomous_Driving_with_OpenCV
# https://www.sciencedirect.com/science/article/pii/S095741742300756X

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
    DANGER = 10
    PUT_DOWN_PACKAGE = 11

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

        self.DEBUG = False
        self.SIMULATION = False

        # --------------------------- ----------- --------------------------- #
        # --------------------------- Robot state --------------------------- # 
        # --------------------------- ----------- --------------------------- #
        self.goal_position = (0,0)
        self.start_position = (0,0)
        self.goal_storage_position = (0,0) 
        self.current_position = copy.deepcopy(self.start_position)
        self.next_position = self.current_position
        self.package_id = 0
        
        self.path_plan = []
        self.map = [
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
        ]

        self.direction = Directions.SOUTH
        self.direction_turn = Turning.CLOCK_WISE
        self.jetank_state = JetankState.IDLE
        self.prev_jetank_state = JetankState.IDLE
        self.is_arm_published = False
        self.recalculating_route = False
        
        ## user-defined parameters:
        self.dot_detected = False
        self.dead_reckoning_active = False
        self.line_disseappered_at_time = 0.0
        self.dot_disseappered_at_time = 0.0
        self.last_dot_time_since_FOLLOW_LINE_state = 0.0

        # Send messages every X seconds
        # The maximum error value for which the robot is still in a straight line
        self.MAX_ERROR = 30
        # The maximum error value for which the robot is aligned when turning
        self.MAX_ALIGNMENT_ERROR = 50

        self.error = 0
        self.prev_error = None
        self.prev_cx = None
        self.prev_cy = None
        self.alpha_smoother = 0.3 

        # Proportional constants to be applied on speed when turning (angular vel) or driving (linear vel)
        # https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller
        # https://support.haltech.com/portal/en/kb/articles/proportional-integral-and-derivative-gain
        # https://softinery.com/blog/implementation-of-pid-controller-in-python/
        # (Multiplied by the error value)
        self.KP = 1.1/100 
        self.KI = 1.1/100 
        self.KD = 1.1/100 
        self.integral = 0
        self.last_time = time.time()

        self.current_lin_vel = 0
        self.prev_lin_vel = 0
        self.current_ang_vel = 0 
        self.prev_ang_vel = 0

        self.TIMER_PERIOD = 1/30

        if self.SIMULATION:
            # Linear velocity (linear.x in Twist) 
            self.LIN_VEL = 0.3
            # Angular velocity (angular.z in Twist)
            self.ANG_VEL = 2.0
            # Amount of time driving forwards (since dots are quite close it would be impossible that the Jetank drives more then +/- 10sec)
            self.DRIVE_FORWARD_THRESHOLD = 30.0
            # Amount of time between a observation (dot or line disappeared) and action (stop , move forwards, realign)
            self.DEAD_RECKONING_THRESHOLD = 0.8
            # Minimum size for a contour to be considered anything
            self.MIN_AREA = 6000

            self.to_examine = [ 
                DotType.BLUE,
                DotType.RED,
            ]

            # https://en.wikipedia.org/wiki/HSL_and_HSV
            # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
            # RED is a bit tricky due to range of the HSV color space 
            # also HSV is defined as range from (0deg, 0%, 0%) to (360deg , 100% , 100%)
            # but because OpenCV wants to keep 8bit unsigned integers 
            # 360deg is mapped to 180 (x deg => x/2) 
            # 100% is mapped to 255   (x %   => x/100 * 255)
            self.lower_green = np.array([30, 50, 50])
            self.upper_green = np.array([60, 255, 255])
    
            self.lower_blue = np.array([70, 125, 125])
            self.upper_blue = np.array([120, 255, 255])
    
            self.lower_red1 = np.array([0, 125, 125])
            self.upper_red1 = np.array([15, 255, 255])
            self.lower_red2 = np.array([170, 125, 125])
            self.upper_red2 = np.array([179, 255, 255])
        else:
            self.LIN_VEL = 0.6
            self.ANG_VEL = 4
            self.DRIVE_FORWARD_THRESHOLD = 30.0
            self.DEAD_RECKONING_THRESHOLD = 0.8
            self.MIN_AREA = 6000
            
            self.to_examine = [ 
                DotType.RED,
            ]

            self.lower_green = np.array([60, 50, 130])
            self.upper_green = np.array([100, 255, 255])

            self.lower_blue = np.array([110, 50, 130])
            self.upper_blue = np.array([130, 255, 255])

            self.lower_red1 = np.array([0, 50, 130])
            self.upper_red1 = np.array([20, 255, 255])
            self.lower_red2 = np.array([175, 50, 130])
            self.upper_red2 = np.array([179, 255, 255])
 
        # --------------------------- ------------------------------------- --------------------------- #
        # --------------------------- subscriptions , publishers and timers --------------------------- # 
        # --------------------------- ------------------------------------- --------------------------- #

        # current subscriptions:
        # ----------------------
        # /<ns>/camera/image_raw/compressed
        # /<ns>/goal_position
        # /<ns>/start_position
        # /<ns>/FSM_state
        # /<ns>/direction
        # /server/reset
        # /server/map
         
        # current publishers:
        # -------------------
        # /to_server
        # /<ns>/detection_result
        # /<ns>/ik_input
        # /<ns>/arm_gripper
        # /<ns>/cmd_vel
        
        # if DEBUG
        # /<ns>/detection/result/red_mask
        # /<ns>/detection/result/green_mask
        # /<ns>/detection/result/blue_mask

        # --------------------------- ------------------------------------- --------------------------- #
        # subscribeer to the camera feed 
        self.image_feed = self.create_subscription(
            msg_type=CompressedImage,
            topic='camera/image_raw/compressed',
            callback=self.read_image_callback,
            qos_profile=10
        )

        # --------------------------- ------------------------------------- --------------------------- #
        # subscribers for the different robot params
        # e.g.: ros2 topic pub /jetank_1/goal_position --once geometry_msgs/msg/Point "{x: 1.0,y: 1.0,z: 1.0}"
        self.to_server_pub = self.create_publisher(
            msg_type=String,
            topic="/to_server",
            qos_profile=10,
        )
        # subscribers for the different robot params
        # e.g.: ros2 topic pub /jetank_1/goal_position --once geometry_msgs/msg/Point "{x: 1.0,y: 1.0,z: 1.0}"
        if self.SIMULATION:
            self.goal_pos_sub = self.create_subscription(
                msg_type=Point,
                topic='goal_position',
                callback=self.listen_to_server_goal_position,
                qos_profile=10
            )
        else:
            self.goal_pos_sub = self.create_subscription(
                msg_type=String,
                topic='goal_position',
                callback=self.listen_to_server_goal_position,
                qos_profile=10
            )
            self.goal_pos_sub = self.create_subscription(
                msg_type=String,
                topic='/server/reset',
                callback=self.listen_to_server_reset,
                qos_profile=10
            )

        # e.g.: ros2 topic pub /jetank_1/start_position --once geometry_msgs/msg/Point "{x: 1.0,y: 1.0,z: 1.0}"
        self.start_pos_sub = self.create_subscription(
            msg_type=Point,
            topic='start_position',
            callback=self.listen_to_server_start_position,
            qos_profile=10
        )

        # e.g.: ros2 topic pub /jetank_1/state_FSM --once std_msgs/msg/nt16 "{data: 1}"
        self.state_sub = self.create_subscription(
            msg_type=Int16,
            topic='state_FSM',
            callback=self.listen_to_server_state,
            qos_profile=10
        )

        # e.g.: ros2 topic pub /jetank_1/direction --once std_msgs/msg/Int16 "{data: 1}"
        self.direction_sub = self.create_subscription(
            msg_type=Int16,
            topic='direction',
            callback=self.listen_to_server_direction,
            qos_profile=10
        )

        self.map_sub = self.create_subscription(
            msg_type=String,
            topic='/server/map',
            callback=self.listen_to_server_map,
            qos_profile=10
        )

        # --------------------------- ------------------------------------- --------------------------- #
        # subscriber for midas
        self.midas_detect_result = self.create_subscription(
            msg_type=Int16,
            topic='midas_detection',
            callback=self.listen_to_midas_node,
            qos_profile=10
        ) 

        # --------------------------- ------------------------------------- --------------------------- #
        # publisher for processed image for visualization purposes
        self.image_publisher = self.create_publisher(
            msg_type=Image,
            topic='detection/result',
            qos_profile=10
        )

        # for debugging
        if self.DEBUG:
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
            
        # --------------------------- ------------------------------------- --------------------------- #
        # publisher for robot movement commands 
        # => ros2_control controller topics or just the specified controller topics

        if self.SIMULATION:
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
        else:
            self.cmd_vel_publisher = self.create_publisher(
                msg_type=Twist,
                topic='cmd_vel',
                qos_profile=10
            )
            self.ik_arm_publisher = self.create_publisher(
                msg_type=Twist,
                topic='ik_input',
                qos_profile=10
            )
            self.gripper_publisher = self.create_publisher(
                msg_type=Bool,
                topic="arm_gripper",
                qos_profile=10,
            )

        # --------------------------- ------------------------------------- --------------------------- #
        self.main_loop = self.create_timer(
            timer_period_sec=self.TIMER_PERIOD,
            callback=self.FSMloop
        )

        self.cmd_vel_loop = self.create_timer(
            timer_period_sec=self.TIMER_PERIOD,
            callback=self.publish_cmd_vel
        )

        self.get_logger().info(f"start position: {self.start_position}")
        self.get_logger().info(f"current position: {self.current_position}")
        self.get_logger().info(f"direction : {self.direction}")
        self.get_logger().info(f"Notifying server that i am available")
        self.notify_server()

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
        self.prev_lin_vel = 0
        self.prev_ang_vel = 0
        self.current_ang_vel = 0

    def drive_towards_center(self):
        try:
            proportion = (self.MAX_ERROR) / abs(self.error)
        except ZeroDivisionError:
            proportion = 1

        self.current_lin_vel = proportion * self.LIN_VEL
        self.current_ang_vel = self.KP * self.error

    def publish_cmd_vel(self):
        msg = Twist()
        msg.linear.x = float(-1 * self.current_lin_vel)
        msg.angular.z = float(-1 * self.current_ang_vel)
        self.cmd_vel_publisher.publish(msg)

    # ---------------------- ------------- ----------------- #
    # ---------------------- Arm functions ----------------- #
    # ---------------------- ------------- ----------------- #

    def publish_arm_ik(self,points: str):    
        if not self.is_arm_published and not self.SIMULATION: 
            self.is_arm_published = True    
            msg = Twist()

            if points == "pickup": 
                msg.linear.x = 0.12
                msg.linear.y = 0.04
                msg.linear.z = 0.20

            elif points == "rest_pos":
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0

            elif points == "putdown":
                msg.linear.x = 0.14
                msg.linear.y = 0.0
                msg.linear.z = -0.14

            else :  
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0

            self.ik_arm_publisher.publish(msg)

    def publish_gripper(self,open: bool):
        msg = Bool()
        msg.data = open
        self.gripper_publisher.publish(msg)

    # ---------------------- ---------------- ----------------- #
    # ---------------------- Helper functions ----------------- #
    # ---------------------- ---------------- ----------------- #

    def send_to_server(self,message : str,message_type: str):
        
        msg = String() 

        if message_type.upper() == "REQUEST":
            status = False
        elif message_type.upper() == "CONFIRMATION":
            status = True
        elif message_type.upper() == "INFO":
            status = False
        elif message_type.upper() == "WARNING":
            status = False

        
        msg.data = json.dumps({
            "robot_namespace" : self.get_namespace(),
            "robot_message" : self.get_namespace() + " " + message,
            "message_type" : message_type.upper(),
            "package_id": self.package_id,
            "status" : status,            
        })

        self.get_logger().info(f"sending to server : {self.get_namespace()} \n\n {msg.data} \n\n")

        self.to_server_pub.publish(msg)

    def notify_server(self):
        if self.jetank_state == JetankState.DESTINATION_REACHED:
            self.start_position = self.path[len(self.path) - 1] 

            if self.map[self.goal_position[1]][self.goal_position[0]] == ZoneTypes.ZONE_IN.value:
                self.jetank_state = JetankState.PICK_UP_PACKAGE
                
                self.send_to_server(f"""
                {self.get_namespace()} Picking up package in zone: {self.current_position}
                Moving to {self.goal_storage_position}
                """,
                message_type="Info")
                
            elif self.map[self.goal_position[1]][self.goal_position[0]] == ZoneTypes.STORAGE.value:
                self.jetank_state = JetankState.PUT_DOWN_PACKAGE
                
                self.send_to_server(f"""
                {self.get_namespace()} Arrived at destination: {self.current_position}
                
                """,
                message_type="Confirmation")

            
            else: 
                self.jetank_state = JetankState.IDLE

        elif self.jetank_state == JetankState.IDLE:
            self.send_to_server(f"{self.get_namespace()} available",message_type="Confirmation")

        elif self.jetank_state == JetankState.DANGER:
            self.send_to_server(f"{self.get_namespace()} danger detected",message_type="Warning")
            
        else:
            self.send_to_server("Goal position is current position",message_type="Info")
            # we can add more logic in here to allow distress signals 
            # or a new map request or smt
            # or pictures (QR codes)
            self.jetank_state = JetankState.IDLE

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
                    self.goal_position = self.goal_storage_position
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
            self.get_logger().warning(f"Recalculating route...")
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
        # so a backwards pass 
        # e.g.: 
        # prev pos <- next pos
        # start (0,0) <- (0,1)
        # (0,1) <- (1,1)
        # (1,1) <- (1,2) end
        # yields -> start -> (0,0) -> (0,1) -> (1,1) -> (1,2) -> end
        path = []
        current = self.goal_position
        while current != self.start_position:
            path.append(current)
            current = came_from[current]
        path.append(self.start_position)
        path.reverse()

        self.path = path

        return True
    
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
            int(width/10):int(9*width/10)
        ]
        return roi

    def morph_filter(self,mask):
        # https://cyrillugod.medium.com/filtering-and-morphological-operations-990662c5bd59
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        # this basically a convulution but without any complicated kernel 
        kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.erode(mask, kernel,iterations=1)
        # mask = cv2.dilate(mask, kernel,iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)

        return mask

    def create_masks(self,hsv):
        # Masks
        # GREEN
        self.green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        self.green_mask = self.morph_filter(self.green_mask)
        # BLUE
        self.blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        self.blue_mask = self.morph_filter(self.blue_mask)
        # RED
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        self.red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        self.red_mask = self.morph_filter(self.red_mask)

    def try_detect_dots(self):
        for dot_type in self.to_examine:
            if dot_type == DotType.RED:
                mask = self.red_mask                 
            elif dot_type == DotType.BLUE:
                mask = self.blue_mask
            else:
                return False

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area_detected = cv2.contourArea(c)
                if area_detected > self.MIN_AREA:
                    if self.DEBUG:
                        self.get_logger().info(f"Area : {area_detected}")
                    # if a dot is detected we keep track what dot color it 
                    # was and switch to a dot detected state as following a line
                    # is not our priority anymore 
                    self.dot_color_detected = dot_type
                    self.dot_detected = True
                    return True

            self.dot_detected = False

    def detect_and_center(self,roi):
        # the image is processed according to the state we're in. changing from state to state 
        # is based upon a FSM.
        # (https://en.wikipedia.org/wiki/Finite-state_machine)
        # (https://www.spiceworks.com/tech/tech-general/articles/what-is-fsm/)
        # (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
        
        if self.jetank_state == JetankState.DOT_DETECTED:
            if self.dot_color_detected == DotType.RED:
                contours, _ = cv2.findContours(self.red_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour_clr = (0,255,255)
            elif self.dot_color_detected == DotType.BLUE:
                contours, _ = cv2.findContours(self.blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contour_clr = (255,255,255)
            else:
                return False
            
        elif self.jetank_state == JetankState.FOLLOW_LINE or self.jetank_state == JetankState.IDLE:
            contours, _ = cv2.findContours(self.green_mask,1, cv2.CHAIN_APPROX_NONE)
            contour_clr = (0,0,0)

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
                    if image_center_x < cx  < image_center_x + self.MAX_ALIGNMENT_ERROR:
                        cv2.circle(roi, (smoothed_cx, smoothed_cy), 5, (0, 255, 0), 2)
                        # ready to go  
                        return True  
                    
                elif self.direction_turn == Turning.ANTI_CLOCK_WISE:
                    if image_center_x > cx  > image_center_x - self.MAX_ALIGNMENT_ERROR:
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

        # ================================================= #
        if self.jetank_state == JetankState.TURN_ON_SPOT:
            self.last_dot_time_since_FOLLOW_LINE_state = time.perf_counter()

            realigned = False
            if not self.dead_reckoning_active:
                self.dead_reckoning_active = True
                self.line_disseappered_at_time = time.perf_counter() 
            
            elif self.dead_reckoning_active:
                time_diff = time.perf_counter() - self.line_disseappered_at_time
                if time_diff > self.DEAD_RECKONING_THRESHOLD:
                    realigned = self.try_realign_after_turn(roi)
                    if realigned:
                        self.dead_reckoning_active = False
                        self.line_disseappered_at_time = 0
                        self.jetank_state = JetankState.DECIDE_DIRECTION
                        self.get_logger().info("Realigned with green line. Switching to DECIDE_DIRECTION.")
                else:
                    self.get_logger().info(f"Waiting... {time_diff:.2f}s")
            
        # ================================================= #
        elif self.jetank_state == JetankState.FOLLOW_LINE:
            self.detect_and_center(roi)
            self.try_detect_dots()
            

            time_diff = time.perf_counter() - self.last_dot_time_since_FOLLOW_LINE_state
            if  time_diff > self.DRIVE_FORWARD_THRESHOLD:
                self.jetank_state = JetankState.IDLE
                self.get_logger().info("Timer exceeded. Switching to IDLE.")
                self.last_dot_time_since_FOLLOW_LINE_state = time.perf_counter()

            # self.get_logger().info(f"{self.dot_detected} && {self.dead_reckoning_active}")
            
            if self.dot_detected:
                self.jetank_state = JetankState.DOT_DETECTED
                self.get_logger().info("Dot detected. Switching to DOT_DETECTED.")

        # ================================================= #
        elif self.jetank_state == JetankState.DOT_DETECTED:
            self.detect_and_center(roi)
            self.try_detect_dots()

            # self.get_logger().info(f"{self.dot_detected} && {self.dead_reckoning_active}")

            if not self.dot_detected and not self.dead_reckoning_active:
                self.dot_disseappered_at_time = time.perf_counter()
                self.get_logger().info(f"Dot disappeared at time {self.dot_disseappered_at_time:.2f}s")
                self.dead_reckoning_active = True
                self.last_dot_time_since_FOLLOW_LINE_state = time.perf_counter()

            elif self.dead_reckoning_active:
                time_diff = time.perf_counter() - self.dot_disseappered_at_time
                # using a dead reckoning approach (found no other solution)
                # https://www.cavliwireless.com/blog/not-mini/what-is-dead-reckoning
                if time_diff > self.DEAD_RECKONING_THRESHOLD:
                    self.dead_reckoning_active = False
                    self.dot_disseappered_at_time = 0
                    self.jetank_state = JetankState.DECIDE_DIRECTION
                    self.get_logger().info(f"Dot disappeared {time_diff:.2f}. Switching to DECIDE_DIRECTION.")
                else:
                    self.get_logger().info(f"Waiting... {time_diff:.2f}s")

        # ================================================= #
        elif self.jetank_state == JetankState.IDLE:
            self.detect_and_center(roi)
            self.try_detect_dots()
            self.last_dot_time_since_FOLLOW_LINE_state = time.perf_counter()



        # for debugging purposes (can be removed)
        if self.DEBUG:
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
        if self.SIMULATION:
            img_msg = self.bridge.cv2_to_imgmsg(roi,encoding="rgb8")
        else:
            img_msg = self.bridge.cv2_to_imgmsg(roi)
        self.image_publisher.publish(img_msg)

    # ---------------------- ------------------ ----------------- #
    # ---------------------- Callback functions ----------------- #
    # ---------------------- ------------------ ----------------- #

    def listen_to_server_goal_position(self,msg: typing.Union[Point , String]):

        if self.SIMULATION:
            self.goal_position = (int(msg.x),int(msg.y))      
            self.get_logger().info(f"SERVER sending to {self.get_namespace()} goal position: {self.goal_position}")
            if self.goal_position == self.current_position:
                self.jetank_state = JetankState.IDLE
                self.notify_server()
            else:
                self.jetank_state = JetankState.INITIALIZE
        else:
            # TODO => request from Zenoh topic instead which comes from the server
                        
            msg_data_from_server = json.loads(msg.data)
            self.get_logger().info(f"FROM SERVER: {msg_data_from_server}")

            if self.get_namespace() == msg_data_from_server["robot_namespace"]:
                self.get_logger().info(f"{self.get_namespace()} : THIS MESSAGE IS FOR ME")
                self.goal_position = (int(msg_data_from_server["x"]),int(msg_data_from_server["y"]))
                self.package_id = int(msg_data_from_server["package_id"])
                self.goal_storage_position = (int(msg_data_from_server["final_x"]),int(msg_data_from_server["final_y"]))

                self.jetank_state = JetankState.INITIALIZE

            else:
                self.get_logger().warning(f"{self.get_namespace()} : THIS MESSAGE IS NOT FOR ME")

    def listen_to_server_start_position(self,msg: Point):
        self.start_position = (int(msg.x),int(msg.y))      
        self.current_position = copy.deepcopy(self.start_position)
        self.get_logger().info(f"SERVER sending to {self.get_namespace()} start position: {self.start_position}")

    def listen_to_server_state(self,msg: Int16):
        state = int(msg.data)
        new_state = JetankState.IDLE
        if   state == JetankState.INITIALIZE.value : new_state = JetankState.INITIALIZE	
        elif state == JetankState.FOLLOW_LINE.value : new_state = JetankState.FOLLOW_LINE	
        elif state == JetankState.DOT_DETECTED.value : new_state = JetankState.DOT_DETECTED
        elif state == JetankState.TURN_ON_SPOT.value : new_state = JetankState.TURN_ON_SPOT
        elif state == JetankState.DRIVE_FORWARD.value : new_state = JetankState.DRIVE_FORWARD
        elif state == JetankState.DECIDE_DIRECTION.value : new_state = JetankState.DECIDE_DIRECTION
        elif state == JetankState.PICK_UP_PACKAGE.value : new_state = JetankState.PICK_UP_PACKAGE
        elif state == JetankState.PUT_DOWN_PACKAGE.value : new_state = JetankState.PUT_DOWN_PACKAGE
        elif state == JetankState.DESTINATION_REACHED.value : new_state = JetankState.DESTINATION_REACHED
        elif state == JetankState.IDLE.value : new_state = JetankState.IDLE
        elif state == JetankState.IDLE.value : new_state = JetankState.DANGER
        self.jetank_state = new_state
        self.get_logger().info(f"SERVER sending to {self.get_namespace()} state: {self.jetank_state}")

    def listen_to_server_direction(self,msg: Int16):
        direction = int(msg.data)
        new_direction = Directions.NORTH
        if   direction == Directions.NORTH.value : new_direction = Directions.NORTH	
        elif direction == Directions.EAST.value : new_direction = Directions.EAST	
        elif direction == Directions.SOUTH.value : new_direction = Directions.SOUTH
        elif direction == Directions.WEST.value : new_direction = Directions.WEST
        self.direction = new_direction
        self.get_logger().info(f"SERVER sending to {self.get_namespace()} direction: {self.direction}")

    def listen_to_midas_node(self,msg: Int16):
        danger = msg.data
        if danger == 1 and self.jetank_state == JetankState.DANGER:
            pass
        elif danger == 0: 
            self.jetank_state = JetankState.DANGER

    def listen_to_server_map(self,msg: String):

        msg_data_from_server = json.loads(msg.data)
        self.get_logger().info(f"FROM SERVER: {msg_data_from_server}")

        max_y = -1
        max_x = -1
        for zone in msg_data_from_server:
            x , y = zone["coords"]
            if max_y < y:
                max_y = y
            if max_x < x:
                max_x = x

        self.map = [
            [0 for x in range(max_x + 1)]
            for y in range(max_y + 1)
        ]

        for zone in msg_data_from_server:
            x , y = zone["coords"]
            self.map[y][x] = int(zone["zone_id"])

    def listen_to_server_reset(self,msg: String):
        msg_data_from_server = json.loads(msg.data)
        self.get_logger().info(f"FROM SERVER: {msg_data_from_server}")
        
        if self.get_namespace() == msg_data_from_server["robot_namespace"]:
            self.get_logger().info(f"{self.get_namespace()} : THIS MESSAGE IS FOR ME")
            
            if msg_data_from_server["reset"]:
                self.jetank_state == JetankState.INITIALIZE

            if msg_data_from_server["direction"]:
                pass
            if msg_data_from_server["start_position"]:
                pass
            if msg_data_from_server["current_position"]:
                pass
            if msg_data_from_server["state"]:
                pass

    def read_image_callback(self, msg: CompressedImage):
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            # 1) we crop our image to the desired space
            roi = self.crop_to_roi()
            self.process_image(roi)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return
        
    def FSMloop(self):
        # (1)
        # [IDLE] --> listen to server --> /<ns>/goal_position --> [INITIALIZE] 
        # [INITIALIZE] --> [FOLLOW_LINE]

        # (2)
        # [FOLLOW_LINE] --> dot detected --> [DOT_DETECTED]
        # [DOT_DETECTED] --> update position --> [DECIDE_DIRECTION] or [DESTINATION_REACHED]
        # [DECIDE_DIRECTION] --> [TURN_ON_SPOT] or [DRIVE_FORWARD]

        # (3)
        # [TURN_ON_SPOT] --> update direction --> [DECIDE_DIRECTION] --> (2)
        # [DRIVE_FORWARD] --> (2)

        # (4)
        # [DESTINATION_REACHED] --> notify server --> [IDLE] --> (1)

        # TODO : Obstacle detected => set point as BLOCKED or VOID => recalculate route
        # TODO : Map saturated => request new map from the server

        # ----------- state to state conditions ----------- #
        if self.prev_jetank_state == JetankState.DOT_DETECTED and self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.update_position()
            self.stop_moving()

        if self.prev_jetank_state == JetankState.TURN_ON_SPOT and self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.update_direction()
            self.stop_moving()

        if self.prev_jetank_state != JetankState.IDLE and self.jetank_state == JetankState.IDLE:
            self.stop_moving()

        # ----------- state logger ----------- #
        if self.prev_jetank_state is not self.jetank_state:
            self.is_arm_published = False
            self.get_logger().info(f'--- STATE : {self.jetank_state} ---')
            self.prev_jetank_state = self.jetank_state

        # ----------- state condition and logic ----------- #
        # ================================================= #
        if self.jetank_state == JetankState.INITIALIZE:
            if not self.a_star(recalculating=self.recalculating_route):
                self.get_logger().error(f'NO available path')
                self.jetank_state = JetankState.IDLE
            else:
                self.recalculating_route = False
                self.update_position(init=True)
                self.get_logger().info(f'PATH : {self.path}')
                self.jetank_state = JetankState.DECIDE_DIRECTION

        # ================================================= #
        elif self.jetank_state == JetankState.DECIDE_DIRECTION:
            self.goal_direction = self.calculate_direction()
            self.get_logger().info(f'Goal direction : {self.goal_direction}')
            self.get_logger().info(f'Current direction : {self.direction}')
            if self.goal_direction == -1: 
                self.get_logger().error(f'Invalid direction {self.goal_direction}')
                self.jetank_state = JetankState.IDLE

            elif self.direction != self.goal_direction:
                self.direction_turn = self.calculate_turn(self.goal_direction)
                self.get_logger().info(f'Turning : {self.direction_turn}')
                self.jetank_state = JetankState.TURN_ON_SPOT
            
            elif self.direction == self.goal_direction:
                self.jetank_state = JetankState.DRIVE_FORWARD

        # ================================================= #
        elif self.jetank_state == JetankState.FOLLOW_LINE:
            self.drive_towards_center()

        # ================================================= #
        elif self.jetank_state == JetankState.DOT_DETECTED:
            self.drive_towards_center()
            
        # ================================================= #
        elif self.jetank_state == JetankState.TURN_ON_SPOT:
            self.turn_on_spot(self.direction_turn.value)

        # ================================================= #
        elif self.jetank_state == JetankState.DRIVE_FORWARD:
            self.drive_straight()
            self.jetank_state = JetankState.FOLLOW_LINE

        # ================================================= #
        elif self.jetank_state == JetankState.DESTINATION_REACHED:
            self.notify_server()

        # ================================================= #
        elif self.jetank_state == JetankState.IDLE:
            self.publish_arm_ik(points="rest_pos")

        # ================================================= #
        # https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
        elif self.jetank_state == JetankState.PICK_UP_PACKAGE:
            self.publish_gripper(open=True)
            time.sleep(4)
            self.publish_arm_ik(points="pickup")
            time.sleep(4)
            self.publish_gripper(open=False)
            time.sleep(4)
            self.jetank_state = JetankState.IDLE


        # ================================================= #
        elif self.jetank_state == JetankState.PUT_DOWN_PACKAGE:
            self.publish_arm_ik(points="putdown")
            time.sleep(4)
            self.publish_gripper(open=True)
            time.sleep(4)
            self.jetank_state = JetankState.IDLE

        # ================================================= #
        elif self.jetank_state == JetankState.DANGER:
            self.stop_moving()
            self.get_logger().warning(f"Object detected...")
            self.notify_server()
            self.map[self.next_position[1]][self.next_position[0]] = ZoneTypes.VOID.value
            self.recalculating_route = True
            self.jetank_state = JetankState.INITIALIZE
            
    def send_before_dead(self):
        self.send_to_server("shutting down...power off","REQUEST")


def main(args=None):
    rclpy.init(args=args)
    navigator_node = FSMNavigator()
    try:
        rclpy.spin(navigator_node)
    except KeyboardInterrupt:
        navigator_node.stop_moving()

    navigator_node.send_before_dead()
    navigator_node.stop_moving()
    navigator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

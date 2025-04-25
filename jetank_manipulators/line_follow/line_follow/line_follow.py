import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class LineDetector:
    def __init__(self):
        self.img = None
        self.hsv = None
        self.yellow_mask = None
        self.green_mask = None
        self.yellow_lines = []
        self.green_lines = []
        self.yellow_merged = []
        self.green_merged = []
        self.horizontal_lines = []
        self.vertical_lines = []
        self.intersection_points = []
        self.output_img = None
        self.chosen_vertical_line = None
        self.image_center_x = 0
        
        # Color ranges (HSV format)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_green = np.array([40, 70, 50])
        self.upper_green = np.array([80, 255, 255])
        
        # Parameters
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.hough_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 80,
            'minLineLength': 50,
            'maxLineGap': 20
        }
        self.filter_params = {
            'angle_threshold': 10,
            'vertical_threshold': 50,
            'slope_threshold': 0.2
        }
        
        # Navigation parameters
        self.alignment_threshold = 20  # pixels
        self.line_follow_speed = 0.2   # m/s
        self.rotation_speed = 0.1      # rad/s
    
    def create_masks(self):
        # Create color masks
        self.yellow_mask = cv2.inRange(self.hsv, self.lower_yellow, self.upper_yellow)
        self.green_mask = cv2.inRange(self.hsv, self.lower_green, self.upper_green)
        
        # Apply morphological operations
        self.yellow_mask = cv2.morphologyEx(self.yellow_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        self.green_mask = cv2.morphologyEx(self.green_mask, cv2.MORPH_CLOSE, self.morph_kernel)
    
    def process_lines(self, mask):
        edges = cv2.Canny(mask, 50, 150)
        raw_lines = cv2.HoughLinesP(edges, **self.hough_params)
        
        processed_lines = []
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                angle = (angle + 180) % 180
                
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                else:
                    slope = float('inf')
                    intercept = x1
                
                processed_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'slope': slope,
                    'intercept': intercept,
                    'center_y': (y1 + y2) // 2,
                    'center_x': (x1 + x2) // 2,
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
        return processed_lines
    
    def filter_outlier_lines(self, lines):
        if not lines:
            return []
        
        angles = [line['angle'] for line in lines]
        normalized_angles = [angle if angle < 90 else 180 - angle for angle in angles]
        
        horizontal_count = sum(1 for angle in normalized_angles if angle < 45)
        vertical_count = len(normalized_angles) - horizontal_count
        predominantly_horizontal = horizontal_count >= vertical_count
        
        filtered_lines = []
        for line in lines:
            angle = line['angle']
            normalized_angle = angle if angle < 90 else 180 - angle
            
            if predominantly_horizontal:
                if normalized_angle < self.filter_params['angle_threshold']:
                    filtered_lines.append(line)
            else:
                if abs(normalized_angle - 90) < self.filter_params['angle_threshold']:
                    filtered_lines.append(line)
        
        return filtered_lines
    
    def group_thick_lines(self, lines):
        if not lines:
            return []
        
        # Determine primary orientation
        horizontal_count = sum(1 for line in lines 
                             if line['angle'] < 45 or line['angle'] > 135)
        primarily_horizontal = horizontal_count >= len(lines)/2
        
        # Sort lines
        sort_key = 'center_y' if primarily_horizontal else 'center_x'
        lines_sorted = sorted(lines, key=lambda x: x[sort_key])
        
        # Grouping logic
        line_groups = []
        current_group = [lines_sorted[0]]
        
        for line in lines_sorted[1:]:
            last = current_group[-1]
            angle_diff = abs(line['angle'] - last['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            slope_diff = abs(line['slope'] - last['slope']) if \
                        line['slope'] != float('inf') and last['slope'] != float('inf') else 0
            dist = abs(line[sort_key] - last[sort_key])
            
            if (dist < self.filter_params['vertical_threshold'] and
                angle_diff < self.filter_params['angle_threshold'] and
                slope_diff < self.filter_params['slope_threshold']):
                current_group.append(line)
            else:
                line_groups.append(current_group)
                current_group = [line]
        
        if current_group:
            line_groups.append(current_group)
        
        # Merge groups
        merged_lines = []
        for group in line_groups:
            avg_angle = sum(l['angle'] for l in group) / len(group)
            all_points = [p for l in group for p in [(l['coords'][0], l['coords'][1]),
                                                    (l['coords'][2], l['coords'][3])]]
            
            if primarily_horizontal:
                all_points.sort(key=lambda x: x[0])
                x1, y1 = all_points[0]
                x2, y2 = all_points[-1]
                avg_y = sum(p[1] for p in all_points) // len(all_points)
                merged_coords = (x1, avg_y, x2, avg_y)
            else:
                all_points.sort(key=lambda x: x[1])
                x1, y1 = all_points[0]
                x2, y2 = all_points[-1]
                avg_x = sum(p[0] for p in all_points) // len(all_points)
                merged_coords = (avg_x, y1, avg_x, y2)
            
            # Calculate center coordinates
            x1, y1, x2, y2 = merged_coords
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            merged_lines.append({
                'coords': merged_coords,
                'angle': avg_angle,
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                'center_x': center_x,
                'center_y': center_y
            })
        
        return [l for l in merged_lines if l['length'] > 50]
    
    def determine_horizontal_set(self):
        if not self.yellow_merged or not self.green_merged:
            return
        
        yellow_avg = sum(l['angle'] for l in self.yellow_merged) / len(self.yellow_merged)
        green_avg = sum(l['angle'] for l in self.green_merged) / len(self.green_merged)
        
        yellow_score = min(yellow_avg, 180 - yellow_avg)
        green_score = min(green_avg, 180 - green_avg)
        
        if yellow_score < green_score:
            self.horizontal_lines = self.yellow_merged
            self.vertical_lines = self.green_merged
        else:
            self.horizontal_lines = self.green_merged
            self.vertical_lines = self.yellow_merged
    
    def find_intersections(self):
        intersections = []
        for hl in self.horizontal_lines:
            for vl in self.vertical_lines:
                x1, y1, x2, y2 = hl['coords']
                x3, y3, x4, y4 = vl['coords']
                
                # Line intersection logic
                denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
                if denom == 0:
                    continue
                
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                
                if (min(x1, x2)-10 <= px <= max(x1, x2)+10 and 
                    min(y1, y2)-10 <= py <= max(y1, y2)+10 and
                    min(x3, x4)-10 <= px <= max(x3, x4)+10 and
                    min(y3, y4)-10 <= py <= max(y3, y4)+10):
                    intersections.append((int(px), int(py)))
        
        self.intersection_points = intersections
    
    def find_closest_vertical_line(self):
        if not self.vertical_lines:
            return None
        
        # Find the image center x-coordinate
        height, width = self.img.shape[:2]
        self.image_center_x = width // 2
        
        # Sort vertical lines by their center_x coordinate
        self.vertical_lines.sort(key=lambda x: x['center_x'])
        
        # Find the vertical line closest to the center
        closest_line = min(self.vertical_lines, 
                          key=lambda line: abs(line['center_x'] - self.image_center_x))
        
        self.chosen_vertical_line = closest_line
        return closest_line
    
    def calculate_alignment_command(self):
        """Calculate alignment command to center the robot on the chosen vertical line"""
        if self.chosen_vertical_line is None:
            return None
        
        x_diff = self.chosen_vertical_line['center_x'] - self.image_center_x
        
        # Check if we're close enough to the line
        if abs(x_diff) < self.alignment_threshold:
            # We're aligned, can proceed to follow the line
            return {'aligned': True, 'angular_z': 0.0}
        else:
            # Need to rotate to align with the line
            # Positive difference means line is to the right, so rotate clockwise (negative z)
            # Negative difference means line is to the left, so rotate counter-clockwise (positive z)
            angular_z = -np.sign(x_diff) * self.rotation_speed
            return {'aligned': False, 'angular_z': angular_z}
    
    def draw_results(self):
        self.output_img = self.img.copy()
        height, width = self.img.shape[:2]
        
        # Draw image center line
        cv2.line(self.output_img, (self.image_center_x, 0), 
                (self.image_center_x, height), (255, 255, 255), 1)
        
        # Draw horizontal lines with numbers
        self.horizontal_lines.sort(key=lambda x: x['center_y'], reverse=True)
        for i, line in enumerate(self.horizontal_lines, 1):
            x1, y1, x2, y2 = line['coords']
            cx = line['center_x']
            cy = line['center_y']
            cv2.line(self.output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(self.output_img, str(i), (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Draw vertical lines with letters (A, B, C, ...)
        self.vertical_lines.sort(key=lambda x: x['center_x'])
        for i, line in enumerate(self.vertical_lines):
            x1, y1, x2, y2 = line['coords']
            cx = line['center_x']
            cy = line['center_y']
            # Use green color for all vertical lines
            cv2.line(self.output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Use A-Z labeling for vertical lines
            letter = chr(65 + (i % 26))  # 65 is ASCII for 'A'
            cv2.putText(self.output_img, letter, (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Highlight the chosen vertical line if any
        if self.chosen_vertical_line:
            x1, y1, x2, y2 = self.chosen_vertical_line['coords']
            cv2.line(self.output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red color
            
            # Calculate and display alignment status
            alignment = self.calculate_alignment_command()
            if alignment:
                aligned_text = "ALIGNED" if alignment['aligned'] else "ALIGNING..."
                cv2.putText(self.output_img, aligned_text, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw intersections
        for p in self.intersection_points:
            cv2.circle(self.output_img, p, 8, (255, 0, 0), -1)

    def process(self, image):
        self.img = image
        if self.img is None:
            raise ValueError("Error: Image is None")
        height, width = self.img.shape[:2]
        self.image_center_x = width // 2
        
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.create_masks()
        self.yellow_lines = self.process_lines(self.yellow_mask)
        self.green_lines = self.process_lines(self.green_mask)
        self.yellow_lines = self.filter_outlier_lines(self.yellow_lines)
        self.green_lines = self.filter_outlier_lines(self.green_lines)
        self.yellow_merged = self.group_thick_lines(self.yellow_lines)
        self.green_merged = self.group_thick_lines(self.green_lines)
        self.determine_horizontal_set()
        self.find_intersections()
        self.find_closest_vertical_line()
        self.draw_results()
        
        # Return navigation command
        return self.calculate_alignment_command()

class LineDetectorNode(Node):
    def __init__(self):
        super().__init__('line_detector_node')
        self.bridge = CvBridge()
        self.detector = LineDetector()
        
        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        
        # Publisher for processed image
        self.image_publisher = self.create_publisher(
            Image,
            'detected_lines/result',
            10)
        
        # Publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'diff_drive_controller/cmd_vel_unstamped',
            10)
        
        # Robot state
        self.is_aligned = False
        self.is_following_line = False
        self.twist_cmd = Twist()
        
        self.get_logger().info("Line Detection initialized")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return
        
        try:
            # Process the image and get navigation command
            nav_command = self.detector.process(cv_image)
        except ValueError as e:
            self.get_logger().error(f'Detection Error: {e}')
            return
        
        # Publish the processed image
        output_image = self.detector.output_img
        if output_image is not None:
            try:
                output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
                self.image_publisher.publish(output_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge Output Error: {e}')
            
            cv2.imshow('Detected Lines', output_image)
            cv2.waitKey(1)
        else:
            self.get_logger().warn('No output image to display or publish')
        
        # Handle navigation based on the processed image
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
    line_detector_node = LineDetectorNode()
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
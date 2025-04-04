#!/usr/bin/env python3
# for ultralytics => because we're running on Ubuntu 22.04  
# we have to carefully select a numpy version that can be used 
# by ultralytics and other libs that ultralytics depends on. Depending on how you installed ultralytics:
# => run => sudo pip install -U numpy==1.26.4
# => or run => python3 -m pip install numpy==1.26.4
from ultralytics import YOLO

import rclpy
import os
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

from ament_index_python.packages import get_package_share_directory

class Camera_subscriber(Node):

    def __init__(self,node_name='camera_subscriber',namespace="",cli_args=""):
        super().__init__(node_name=node_name,namespace=namespace,cli_args=cli_args)

        # retrieve the models weights which is located in the share directory when we build this package with colcon
        package_share_dir = get_package_share_directory('yolo_node')
        best_pt_path = os.path.join(package_share_dir, 'weights', 'best.pt')
        # same for the classes
        classes_path = os.path.join(package_share_dir, 'classes', 'classes.txt')
        
        # we init the YOLO model
        self.model = YOLO(best_pt_path)
        # and the famous CV_Bridge to convert the images to a image format for OpenCV (see below) 
        self.bridge = CvBridge()

        #Load classes
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.yolov8_inference = Yolov8Inference()

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)
        
        # NOTE: from somebody else I believe he trained it live  "Inference" as learned in AI fund
        # self.yolov8_pub = self.create_publisher(Yolov8Inference, "Yolov8_Inference", 1)

        self.img_pub = self.create_publisher(Image, "yolo_result", 1)

    def camera_callback(self, data):

        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.model(img)

        # NOTE: from somebody else I believe he trained it live  "Inference" as learned in AI fund
        # self.yolov8_inference.header.frame_id = "inference"
        # self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()
        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         self.inference_result = InferenceResult()
        #         b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
        #         c = box.cls
        #         self.inference_result.class_name = self.model.names[int(c)]
        #         self.inference_result.top = int(b[0])
        #         self.inference_result.left = int(b[1])
        #         self.inference_result.bottom = int(b[2])
        #         self.inference_result.right = int(b[3])
        #         self.yolov8_inference.yolov8_inference.append(self.inference_result)
            #camera_subscriber.get_logger().info(f"{self.yolov8_inference}")


        #Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.classes[class_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        annotated_frame = results[0].plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame)  

        self.img_pub.publish(img_msg)

        # NOTE: from somebody else I believe he trained it live  "Inference" as learned in AI fund
        # self.yolov8_pub.publish(self.yolov8_inference)

        self.yolov8_inference.yolov8_inference.clear()


def main():
    rclpy.init(args=None)
    
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
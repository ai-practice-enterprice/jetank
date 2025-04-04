# most important library
import rclpy
from rclpy.node import Node

# messages (NOTE: CompressedImage is also a option)
from sensor_msgs.msg import CompressedImage , Image
from cv_bridge import CvBridge

import numpy as np
import cv2
from cv2 import VideoWriter

# this node is based upon https://answers.ros.org/question/416726/

class ImageToVideoConverter(Node):
    def __init__(self, node_name = "image_to_video_converter", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)

        self.bridge = CvBridge()
        self.video_writer = None
        self.image_capture = self.create_subscription(
            msg_type=Image,
            topic='camera/image_raw',
            callback=self.image_callback,
            qos_profile=10
        )

    def image_callback(self, msg):
        try:
            # np_arr = np.frombuffer(msg.data, np.uint8)
            # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = self.bridge.imgmsg_to_cv2(msg,'rgb8')
            
            if self.video_writer is None:
                self.init_VideoWriter(cv_image)

            self.video_writer.write(cv_image)
            self.get_logger().info('writing...')
        except Exception as e:
            self.get_logger().error('Error processing image: %s' % str(e))

    def init_VideoWriter(self,cv_image: cv2.typing.MatLike):
        try:
            height, width,_ = cv_image.shape
            video_format = 'mp4'  # or any other video format supported by OpenCV
            video_filename = 'recorded_footage.' + video_format
            # https://github.com/opencv/opencv/issues/24818
            fourcc = VideoWriter.fourcc(*'mp4v')
            fps = 30  # Frames per second
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            self.get_logger().info('initialized video writer')
        except Exception as e:
            self.get_logger().error('Error initializing video writer: %s' % str(e))
    


    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        super().destroy_node()    

def main(args=None):
    rclpy.init(args=args)

    video_recorder = ImageToVideoConverter()
    rclpy.spin(video_recorder)

    video_recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


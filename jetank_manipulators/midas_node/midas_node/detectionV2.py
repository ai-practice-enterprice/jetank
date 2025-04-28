import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image , CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from MidasDetector import MidasDetector , ModelType


class MidasNode(Node):
    def __init__(self, node_name = "MidasNode", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        # we init the Midas detector
        self.midas = MidasDetector(ModelType.MIDAS_SMALL)
        # and the famous CV_Bridge
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            msg_type=CompressedImage,
            topic='camera/image_raw/compressed',
            callback=self.camera_callback,
            qos_profile=10
        )

        self.img_pub = self.create_publisher(
            msg_type=Image, 
            topic="midas_result", 
            qos_profile=1
        )

    def camera_callback(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")

        depth_colored, depth_raw = self.midas.predict_depth(img)
        
        img_msg = self.bridge.cv2_to_imgmsg(depth_colored)  

        self.img_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    
    midas_node = MidasNode()
    rclpy.spin(midas_node)

    midas_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
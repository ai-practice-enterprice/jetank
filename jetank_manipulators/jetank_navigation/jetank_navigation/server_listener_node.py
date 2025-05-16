import rclpy
from rclpy.node import Node , ParameterDescriptor
from cv_bridge import CvBridge, CvBridgeError
import random
import time
import json

from std_msgs.msg import Int16 , Bool , Int16MultiArray , String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist , Point


class ServerNode(Node):
    def __init__(self, node_name = "ServerNode", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        self.get_logger().info(f"--- Booting up server ---")

        self.SIMULATION = False

        self.to_server_pub = self.create_subscription(
            msg_type=String,
            callback=self.to_server_callback,
            topic="/to_server",
            qos_profile=10,
        )

        self.OG_map = [
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
            [9,9,9,9,9],
        ]
        self.get_logger().info(f"--- Booting up server : Complete ---")

    def to_server_callback(self,msg):

        ns = msg.robot_namespace
        message = msg.robot_message
        message_type = msg.message_type
        self.get_logger().info(f"Received message type: {message_type} \n\tfrom: {ns} \n\tsaying: {message}")

        if message_type == "INFO":
            if self.SIMULATION:
                publisher = self.create_publisher(
                    msg_type=Point,
                    topic=ns + "/goal_position",
                    qos_profile=10,
                )
                new_point = Point()
                new_point.x = float(random.randint(0,len(self.OG_map[0]) - 1))
                new_point.y = float(random.randint(0,len(self.OG_map) - 1))
                for i in range(5):
                    self.get_logger().info(f"waiting to send...")
                    time.sleep(1)
                self.get_logger().info(f"Sending new goal to {ns} => ({new_point.x},{new_point.y})")
                publisher.publish(new_point)
                self.destroy_publisher(publisher)
            else:
                publisher = self.create_publisher(
                    msg_type=String,
                    topic=ns + "/goal_position",
                    qos_profile=10,
                )
                new_point = String()
                new_point.data = json.dumps({
                    "robot_namespace": "/" + r1.robotNamespace, 
                    "package_id": package_id, 
                    "x": random.randint(0,len(self.OG_map[0]) - 1), 
                    "y": random.randint(0,len(self.OG_map) - 1)
                })
                for i in range(5):
                    self.get_logger().info(f"waiting to send...")
                    time.sleep(1)
                self.get_logger().info(f"Sending new goal to {ns} => ({new_point.data})")
                publisher.publish(new_point)
                self.destroy_publisher(publisher)


def main(args=None):
    rclpy.init(args=args)
    server_node = ServerNode()
    try:
        rclpy.spin(server_node)
    except KeyboardInterrupt:
        pass

    server_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node

import json
from enum import Enum

from std_msgs.msg import String


class MessageTypes(Enum):
    INFO = 1
    WARNING = 1
    CONFIRMATION = 1


class ClientJetankNode(Node):
    def __init__(self, node_name = "ClientJetankNode", *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)

        self.get_logger().info(f"----- Booting up client node {self.get_name()}{self.get_namespace()} -----")

        # --------------------------- ------------------------------------- --------------------------- #
        self.goal_pos_sub = self.create_subscription(
            msg_type=String,
            topic='goal_position',
            callback=self.listen_to_server_goal_position,
            qos_profile=10
        )
        # --------------------------- ------------------------------------- --------------------------- #
        self.to_server_pub = self.create_publisher(
            msg_type=String,
            topic="/to_server",
            qos_profile=10,
        )

        self.timer = self.create_timer(
            timer_period_sec=30,
            callback=self.test_message_to_server_callback
        )

        self.timer_message_feedback = self.create_timer(
            timer_period_sec=2,
            callback=self.test_message_type_callback
        )

        self.next_message = None

        self.get_logger().info(f"----- Booting up client node {self.get_name()}{self.get_namespace()} : COMPLETED -----")


    def send_to_server(self,message : str,message_type: str):
        
        msg = String() # NotificationServer()
        # msg.robot_namespace = self.get_namespace()
        # msg.robot_message = f"{self.get_namespace()} {message}"
        # msg.message_type = message_type.upper()

        msg.data = json.dumps({
            "robot_namespace" : self.get_namespace(),
            "robot_message" : self.get_namespace() + message,
            "message_type" : message_type.upper(),            
        })

        self.get_logger().info(f"sending to server : {self.get_namespace()} \n\n {msg.data} \n\n")

        self.to_server_pub.publish(msg)

    def listen_to_server_goal_position(self,msg: String):

        msg_data_from_server = json.loads(msg.data)
        self.get_logger().info(f"FROM SERVER: {msg_data_from_server}")

        if self.get_namespace() == msg_data_from_server["robot_namespace"]:
            self.get_logger().info(f"{self.get_namespace()} : THIS MESSAGE IS FOR ME")

            self.goal_position = (int(msg_data_from_server["x"]),int(msg_data_from_server["y"]))
        else:
            self.get_logger().warning(f"{self.get_namespace()} : THIS MESSAGE IS NOT FOR ME")

    def test_message_to_server_callback(self):


        if self.next_message is None:
            self.next_message = MessageTypes.INFO

        if self.next_message == MessageTypes.INFO:
            self.next_message = MessageTypes.WARNING
            self.send_to_server("Hello from test node. I am not a robot!","info")

        if self.next_message == MessageTypes.WARNING:
            self.next_message = MessageTypes.CONFIRMATION
            self.send_to_server("Hello from test node. I am not a robot! This is a warning :( ","warning")
            
        if self.next_message == MessageTypes.CONFIRMATION:
            self.next_message = MessageTypes.INFO
            self.send_to_server("Hello from test node. I am not a robot!  This is a confirmation :) ","confirmation")


    def test_message_type_callback(self):
        self.get_logger().info(f"Current sending message {self.next_message}")


def main(args=None):
    
    rclpy.init(args=args)
    client_node = ClientJetankNode()

    try:
        rclpy.spin(client_node)
    except KeyboardInterrupt:
        pass

    client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
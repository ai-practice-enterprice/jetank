import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node


from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription , ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution , FindExecutable 
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition , UnlessCondition

# This launch file starts every node that is required for the Jetank
# both for the real deal and the simulation
def generate_launch_description():
    
    # these are all our params they are quite straightforward
    # but you can see explanation in the "DeclareLaunchArguments" 
    midas_on = LaunchConfiguration('midas_on')
    camera_gst_on = LaunchConfiguration('camera_gst_on')
    FSM_nav_on = LaunchConfiguration('FSM_nav_on')
    ik_controller_on = LaunchConfiguration('ik_controller_on')
    arm_controller_on = LaunchConfiguration('arm_controller_on')
    drive_controller_on = LaunchConfiguration('drive_controller_on')
    robot_namespace = LaunchConfiguration('ns')


    midas_arg = DeclareLaunchArgument(
        name='midas_on',
        default_value='false',
        description='[ARG] turn on/off the midas node on startup'
    )
    camera_gst_arg = DeclareLaunchArgument(
        name='camera_gst_on',
        default_value='true',
        description='[ARG] turn on/off the gstreamer node on startup (ONLY REQUIRED WHEN USING THE JETSON0)'
    )
    FSM_nav_arg = DeclareLaunchArgument(
        name='FSM_nav_on',
        default_value='true',
        description='[ARG] turn on/off the FSMNavigator node on startup'
    )
    robot_namespace_arg = DeclareLaunchArgument(
        name='ns',
        description='[ARG] required namespace to keep entity (simulation), nodes and topics separate when running multiple robots in the same simulation'
    )
    ik_controller_on_arg = DeclareLaunchArgument(
        name='ik_controller_on',
        default_value='true',
        description='[ARG] controller inverse kinematics controller on'
    )
    arm_controller_on_arg = DeclareLaunchArgument(
        name='arm_controller_on',
        default_value='true',
        description='[ARG] controller arm controller on'
    )
    drive_controller_on_arg = DeclareLaunchArgument(
        name='drive_controller_on',
        default_value='true',
        description='[ARG] controller diffdrive controller on'
    )




    FSM_navigator_node = Node(
        package="jetank_navigation",
        executable="navigate",
        namespace=robot_namespace,
        condition=IfCondition(FSM_nav_on)
    )
    camera_gst_node = Node(
        package="camera_image_publisher",
        executable="img_jetson_publisher",
        namespace=robot_namespace,
        condition=IfCondition(camera_gst_on)
    )
    midas_node = Node(
        package="midas_node",
        executable="depthV2",
        namespace=robot_namespace,
        condition=IfCondition(midas_on)
    )
    ik_node = Node(
        package="jetank_ik",
        executable="controller",
        namespace=robot_namespace,
        condition=IfCondition(ik_controller_on)
    )
    arm_node = Node(
        package="jetank_arm",
        executable="controller",
        namespace=robot_namespace,
        condition=IfCondition(arm_controller_on)
    )
    drive_node = Node(
        package="jetank_drive",
        executable="controller",
        namespace=robot_namespace,
        condition=IfCondition(drive_controller_on)
    )

    return LaunchDescription([
        midas_arg,
        camera_gst_arg,
        FSM_nav_arg,
        robot_namespace_arg,
        ik_controller_on_arg,
        arm_controller_on_arg,
        drive_controller_on_arg,

        FSM_navigator_node,
        camera_gst_node,
        midas_node,
        ik_node,
        arm_node,
        drive_node,
    ])
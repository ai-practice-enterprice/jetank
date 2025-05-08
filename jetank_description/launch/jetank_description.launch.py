import os
import xacro

from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument , IncludeLaunchDescription , ExecuteProcess
from launch.substitutions import LaunchConfiguration , PythonExpression , Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    # get this package's location in share directory
    jetank_package = get_package_share_directory('jetank_description')
    
    # here we just declare and define a variable that will be evaluated at runtime
    # to see whether we use the simulation's clock or unix clock
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='false',
        description='[ARG] tells the robot_state_publisher to use the simulation time or just unix timestamp'
    )

    robot_namespace = LaunchConfiguration('ns')
    robot_namespace_arg = DeclareLaunchArgument(
        name='ns',
        description='[ARG] required namespace to keep nodes and topics separate when running multiple robots in the same simulation'
    )

    # gather the main xacro file's location in order to feed it to the 
    # Xacro program
    main_xacro_file = os.path.join(
        jetank_package,'urdf','jetank_main.xacro'
    )

    if os.path.exists(main_xacro_file):
        jetank_description = Command(
            # spaces are important here since this is if we entered it on the CLI
            ["xacro"," ",main_xacro_file," ","namespace:=",robot_namespace]
        )
    else:
        print(f'failed to open {main_xacro_file}')
        exit()

    
    return LaunchDescription([
        use_sim_time_arg,
        robot_namespace_arg,
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            namespace=robot_namespace,
            parameters=[{
                # NOTE: if you switch to gazebo ignition the parser might cause some problems and therefore some special characters are not allowed in the URDF files
                # see => https://github.com/ros-controls/gazebo_ros2_control/issues/295 
                'robot_description': ParameterValue(jetank_description,value_type=str),
                'use_sim_time': use_sim_time,
            }],
            # multiple robots can be handled in different ways but i believe the best approach is to use namespaces and remap certain topics
            # https://discourse.ros.org/t/tf-tree-in-a-multi-robot-setup-in-ros2/41426/3
            # https://github.com/ros2/geometry2/issues/433
            remappings=[
                ('/tf','tf'),
                ('/tf_static','tf_static'),
            ]
        ),
    ])
        


import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('vision_input')

    # LAUNCH ARGUMENTS

    use_ur_driver_arg = DeclareLaunchArgument(
        'use_ur_driver',
        default_value='false',
        description='Set true when UR driver is running (lab/real robot)'
    )
    use_ur_driver = LaunchConfiguration('use_ur_driver')

    fatigue_ramp_arg = DeclareLaunchArgument(
        'fatigue_ramp_minutes',
        default_value='10.0',
        description='Time in minutes for fatigue to ramp from fresh to severe'
    )
    fatigue_ramp = LaunchConfiguration('fatigue_ramp_minutes')


    # 1. RViz + Robot Description (HOME MODE ONLY)

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'ur3_rviz.launch.py')
        ),
        condition=UnlessCondition(use_ur_driver)
    )


    # 2. Static Transform: camera_link relative to base_link

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.6', '0.5', '0.4', '0', '0', '0', 'base_link', 'camera_link'],
        output='screen'
    )

    # 3. Camera Node (The Eyes)

    camera_node = Node(
        package='vision_input',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    # 4. Robot Controller (The Brain)

    controller_node = Node(
        package='vision_input',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[{
            'use_ur_driver': use_ur_driver
        }]
    )

    # 5. Tobii Eye Tracking (Simulation Mode)

    tobii_node = Node(
        package='vision_input',
        executable='tobii_node',
        name='tobii_node',
        output='screen',
        parameters=[{
            'simulation_mode': True,
            'fatigue_ramp_minutes': fatigue_ramp,
        }]
    )

    # 6. Fatigue Monitor (Composite Score Engine)

    fatigue_monitor = Node(
        package='vision_input',
        executable='fatigue_monitor',
        name='fatigue_monitor',
        output='screen'
    )

    return LaunchDescription([
        use_ur_driver_arg,
        fatigue_ramp_arg,
        rviz_launch,
        camera_tf,
        camera_node,
        controller_node,
        tobii_node,
        fatigue_monitor,
    ])
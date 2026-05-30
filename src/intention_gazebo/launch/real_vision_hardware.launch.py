"""
real_vision_hardware.launch.py — Real UR3 hardware launch (Scenario 2: webcam)

Launches the application layer for webcam-driven fetch-and-deliver on the
real UR3. Assumes the UR driver is already running in another terminal:

    ros2 launch ur_robot_driver ur_control.launch.py \\
        ur_type:=ur3 robot_ip:=192.168.1.10 \\
        launch_rviz:=false use_fake_hardware:=false \\
        description_package:=intention_gazebo \\
        description_file:=ur3_robotiq_real.urdf.xacro

The UR driver owns /robot_description, robot_state_publisher, and the
controller_manager. This launch file must NOT duplicate any of those.
Unlike real_ur3.launch.py (gamepad), this launch uses the webcam path:
camera_node + hand_mirror feed bimanual hand data, torch_override stays
False so landmark_callback drives the arm.

Usage:
    # Terminal 1: UR driver (as above): press Play on pendant
    # Terminal 2:
    ros2 launch intention_gazebo real_vision_hardware.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_gazebo = get_package_share_directory('intention_gazebo')
    rviz_config = os.path.join(pkg_gazebo, 'config', 'gazebo_view.rviz')

    fatigue_ramp_arg = DeclareLaunchArgument(
        'fatigue_ramp_minutes',
        default_value='10.0',
        description='Minutes for simulated fatigue ramp'
    )
    fatigue_ramp = LaunchConfiguration('fatigue_ramp_minutes')

    camera_id_arg = DeclareLaunchArgument(
        'camera_id', default_value='0',
        description='Video device index (e.g. 3 for /dev/video3)'
    )
    camera_id = LaunchConfiguration('camera_id')

    use_sim_tobii_arg = DeclareLaunchArgument(
        'use_sim_tobii',
        default_value='true',
        description='True = simulated pupil+gaze; False = live Tobii Pro 3 via g3pylib'
    )
    use_sim_tobii = LaunchConfiguration('use_sim_tobii')

    # Shared mapping params: explicitly exposed so they can be tuned at
    # the tripod without rebuilding. Matches gazebo_sim.launch.py webcam
    # defaults. map_scale_y/_z widen or narrow the operator's reachable
    # workspace; map_offset_x/_z shift the mapped origin.
    map_offset_x_arg = DeclareLaunchArgument('map_offset_x', default_value='0.30')
    map_scale_x_arg  = DeclareLaunchArgument(
        'map_scale_x',  default_value='1.2',
        description='Overhead X-axis (forward/back) scale — 16:9 aspect fix'
    )
    map_scale_y_arg  = DeclareLaunchArgument('map_scale_y',  default_value='0.9')
    map_scale_z_arg  = DeclareLaunchArgument('map_scale_z',  default_value='0.10')
    map_offset_z_arg = DeclareLaunchArgument('map_offset_z', default_value='0.45')
    camera_placement_arg = DeclareLaunchArgument(
        'camera_placement', default_value='overhead',
        description='Camera angle: overhead (tripod top-down) or front_facing'
    )

    map_offset_x = LaunchConfiguration('map_offset_x')
    map_scale_x  = LaunchConfiguration('map_scale_x')
    map_scale_y  = LaunchConfiguration('map_scale_y')
    map_scale_z  = LaunchConfiguration('map_scale_z')
    map_offset_z = LaunchConfiguration('map_offset_z')
    camera_placement = LaunchConfiguration('camera_placement')

    # robot_controller params for webcam + real UR3.
    # - use_gazebo_sim=False: no LinkAttacher, no /set_entity_state calls
    # - use_virtual_operator=False: bimanual (right hand Y,Z + left hand X depth)
    # - use_gamepad=False: torch_override defaults False → landmark_callback drives arm
    robot_ctrl_params = [{
        'use_sim_time': False,
        'use_ur_driver': True,
        'use_gazebo_sim': False,
        'use_virtual_operator': False,
        'use_gamepad': False,
        'use_tf_tracking': False,
        'use_depth_camera': False,
        'camera_placement': camera_placement,
        'map_offset_x': map_offset_x,
        'map_scale_x': map_scale_x,
        'map_scale_y': map_scale_y,
        'map_scale_z': map_scale_z,
        'map_offset_z': map_offset_z,
    }]

    # Static TF for the workstation camera frame. Not strictly required
    # for the 2D mapping math (robot_controller and hand_mirror do not
    # reference camera_link: they project normalized image coords
    # directly into base_link via map_offset/scale), but included for
    # parity with gazebo_sim.launch.py and real_ur3.launch.py so RViz
    # displays the camera frame consistently across all launches.
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.6', '0.5', '0.4', '0', '0', '0',
                   'base_link', 'camera_link'],
        output='screen',
    )

    # Vision input chain
    camera_node = Node(
        package='vision_input',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{'camera_id': camera_id}],
    )

    hand_mirror = Node(
        package='vision_input',
        executable='hand_mirror',
        name='hand_mirror',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'map_offset_x': map_offset_x,
            'map_scale_x': map_scale_x,
            'map_scale_y': map_scale_y,
            'map_scale_z': map_scale_z,
            'map_offset_z': map_offset_z,
            'camera_placement': camera_placement,
        }],
    )

    # Application nodes
    robot_controller = Node(
        package='vision_input',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=robot_ctrl_params,
    )

    tobii_node = Node(
        package='vision_input',
        executable='tobii_node',
        name='tobii_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'simulation_mode': use_sim_tobii,
            'fatigue_ramp_minutes': fatigue_ramp,
        }],
    )

    fatigue_monitor = Node(
        package='vision_input',
        executable='fatigue_monitor',
        name='fatigue_monitor',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    fatigue_override = Node(
        package='vision_input',
        executable='fatigue_override',
        name='fatigue_override',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    # Delay app nodes by 2s so the UR driver's RSP has /robot_description
    # and /joint_states up. Same pattern as real_ur3.launch.py.
    delayed_app = TimerAction(
        period=2.0,
        actions=[
            camera_node,
            hand_mirror,
            robot_controller,
            tobii_node,
            fatigue_monitor,
            fatigue_override,
            rviz,
        ],
    )

    return LaunchDescription([
        fatigue_ramp_arg,
        camera_id_arg,
        use_sim_tobii_arg,
        map_offset_x_arg,
        map_scale_x_arg,
        map_scale_y_arg,
        map_scale_z_arg,
        map_offset_z_arg,
        camera_placement_arg,
        camera_tf,
        delayed_app,
    ])

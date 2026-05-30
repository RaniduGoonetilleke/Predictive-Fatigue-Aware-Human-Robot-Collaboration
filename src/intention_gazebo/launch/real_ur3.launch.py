"""
real_ur3.launch.py, Real UR3 hardware launch (Scenario 1: gamepad)

Launches ONLY the application layer, gamepad input, robot_controller,
fatigue stack, fatigue override, and RViz. Assumes the UR driver is
already running separately in another terminal:

    ros2 launch ur_robot_driver ur_control.launch.py \\
        ur_type:=ur3 robot_ip:=192.168.1.10 \\
        launch_rviz:=false use_fake_hardware:=false \\
        description_package:=intention_gazebo \\
        description_file:=urdf/ur3_robotiq_real.urdf.xacro

The UR driver owns /robot_description, robot_state_publisher, and the
controller_manager. This launch file must NOT duplicate any of those.
It also does NOT spawn Gazebo, the virtual hand, or any Gazebo-specific
controllers, those are sim-only.

Usage:
    # Terminal 1: UR driver (as above): press Play on pendant to start robot
    # Terminal 2:
    ros2 launch intention_gazebo real_ur3.launch.py
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

    use_sim_tobii_arg = DeclareLaunchArgument(
        'use_sim_tobii',
        default_value='true',
        description='True = simulated pupil+gaze; False = live Tobii Pro 3 via g3pylib'
    )
    use_sim_tobii = LaunchConfiguration('use_sim_tobii')

    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='1',
        description='Webcam device index for camera_node (default 1 for external USB cam)'
    )
    camera_id = LaunchConfiguration('camera_id')

    # Shared robot_controller params. Matches gazebo_sim.launch.py mapping
    # so the operator feels the same motion scaling on real hardware.
    # use_gazebo_sim=False bypasses LinkAttacher, /set_entity_state, and
    # all cube teleport calls: grasping on the real robot is handled by
    # the physical Robotiq gripper via command_gripper().
    robot_ctrl_params = {
        'use_sim_time': False,
        'use_ur_driver': True,
        'use_gazebo_sim': False,
        'use_virtual_operator': True,  # all 3 axes from one synthetic hand (gamepad)
        'use_gamepad': True,           # gamepad torch is the input source
        'map_offset_x': 0.30,
        'map_scale_y': 0.9,
        'map_scale_z': 0.10,
        'map_offset_z': 0.45,
        'use_depth_camera': False,
        'camera_placement': 'front_facing',
    }

    # Static TF for the workstation camera frame (same as sim).
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.6', '0.5', '0.4', '0', '0', '0',
                   'base_link', 'camera_link'],
        output='screen',
    )

    # Gamepad input chain
    # Use joy_linux (legacy /dev/input/js0 reader) instead of the
    # default 'joy' package's SDL2-based joy_node. SDL2 silently
    # drops the stick axes for the Logitech F310 in X mode while
    # still reading buttons/triggers correctly: confirmed by jstest
    # showing full stick range that joy_node never published.
    joy_node = Node(
        package='joy_linux',
        executable='joy_linux_node',
        name='joy_node',
        output='screen',
    )

    gamepad_operator = Node(
        package='vision_input',
        executable='gamepad_operator',
        name='gamepad_operator',
        output='screen',
    )

    # Overhead/front-facing webcam for MediaPipe hand tracking. Default
    # camera_id=1 (external USB cam). Override with camera_id:=0 if the
    # laptop's built-in webcam is the intended source.
    camera_node = Node(
        package='vision_input',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{'camera_id': camera_id}],
    )

    # Application nodes
    robot_controller = Node(
        package='vision_input',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[robot_ctrl_params],
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

    # fatigue_override runs a global pynput keyboard hook, so it needs
    # a real terminal. Separate process, same as sim launch.
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

    # Startup ordering 
    # Delay app nodes by 2s to give the UR driver's robot_state_publisher
    # time to publish /robot_description and /joint_states. Nothing here
    # is chained to OnProcessExit because there's no spawn_gripper event
    # to hang off — the UR driver's controllers are already up before
    # this launch is run.
    delayed_app = TimerAction(
        period=2.0,
        actions=[
            joy_node,
            gamepad_operator,
            camera_node,
            robot_controller,
            tobii_node,
            fatigue_monitor,
            fatigue_override,
            rviz,
        ],
    )

    return LaunchDescription([
        fatigue_ramp_arg,
        use_sim_tobii_arg,
        camera_id_arg,
        camera_tf,
        delayed_app,
    ])

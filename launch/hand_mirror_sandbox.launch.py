"""
hand_mirror_sandbox.launch.py, Virtual hand sandbox

UR3 + virtual hand in Gazebo, controlled via keyboard or gamepad.
Camera positioned as operator sitting in front of the robot.
No fatigue, no fetch logic, no object detection, just hand + robot.

Usage:
  colcon build --packages-select vision_input intention_gazebo
  source install/setup.bash
  ros2 launch intention_gazebo hand_mirror_sandbox.launch.py
  # Keyboard+mouse by default, hold LB on gamepad to switch to gamepad
"""

import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction,
    SetEnvironmentVariable, OpaqueFunction, RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue


def setup_gazebo_model_path(context, *args, **kwargs):
    model_dir = '/tmp/intention_gazebo_models'
    os.makedirs(model_dir, exist_ok=True)
    for pkg_name in ['robotiq_description', 'ur_description', 'intention_gazebo']:
        pkg_dir = get_package_share_directory(pkg_name)
        link_path = os.path.join(model_dir, pkg_name)
        if os.path.islink(link_path):
            os.unlink(link_path)
        if not os.path.exists(link_path):
            os.symlink(pkg_dir, link_path)
    return []


def generate_launch_description():

    pkg_gazebo = get_package_share_directory('intention_gazebo')

    urdf_file = os.path.join(pkg_gazebo, 'urdf', 'ur3_robotiq_gazebo.urdf.xacro')
    controllers_yaml = os.path.join(pkg_gazebo, 'config', 'gazebo_controllers.yaml')
    world_file = os.path.join(pkg_gazebo, 'worlds', 'workstation.sdf')
    rviz_config = os.path.join(pkg_gazebo, 'config', 'hand_sandbox_view.rviz')
    virtual_hand_xacro = os.path.join(pkg_gazebo, 'urdf', 'virtual_hand.urdf.xacro')

    robot_description = Command([
        'xacro ', urdf_file,
        ' simulation_controllers:=', controllers_yaml,
    ])
    virtual_hand_description = Command(['xacro ', virtual_hand_xacro])

    # Gazebo
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # UR3 robot state publisher 
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str),
            'use_sim_time': True,
        }],
    )

    # Virtual hand RSP
    virtual_hand_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='virtual_hand_state_publisher',
        namespace='virtual_hand',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(virtual_hand_description, value_type=str),
            'use_sim_time': True,
            'frame_prefix': '',
        }],
    )

    # Camera TF
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.6', '0.5', '0.4', '0', '0', '0',
                   'base_link', 'camera_link'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Spawn UR3 robot
    spawn_robot = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_ur3',
                output='screen',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'ur3_robotiq',
                    '-x', '0.0', '-y', '0.0', '-z', '0.0',
                ],
            ),
        ]
    )

    # Spawn virtual hand
    spawn_virtual_hand = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_virtual_hand',
                output='screen',
                arguments=[
                    '-topic', '/virtual_hand/robot_description',
                    '-entity', 'virtual_hand',
                    '-x', '0.80', '-y', '0.00', '-z', '0.60',
                ],
            ),
        ],
    )

    # Controller spawners (chained)
    spawn_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    spawn_jtc = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    spawn_gripper = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    chain_jtc_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_jtc])
    )
    chain_gripper_after_jtc = RegisterEventHandler(
        OnProcessExit(target_action=spawn_jtc, on_exit=[spawn_gripper])
    )

    # Send robot to home position
    send_home = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_gripper,
            on_exit=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'topic', 'pub', '--once',
                        '/scaled_joint_trajectory_controller/joint_trajectory',
                        'trajectory_msgs/msg/JointTrajectory',
                        '{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint], points: [{positions: [0.324, -1.240, 0.793, -1.135, -1.595, 0.217], time_from_start: {sec: 3}}]}'
                    ],
                    output='screen'
                ),
            ],
        )
    )

    # RViz (after controllers ready)
    rviz = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_gripper,
            on_exit=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    output='screen',
                    arguments=['-d', rviz_config],
                    parameters=[{'use_sim_time': True}],
                ),
            ],
        )
    )

    # Input: BOTH keyboard + gamepad always run
    # LB-based handoff: virtual_operator pauses when LB is held,
    # gamepad_operator pauses when LB is released. No conflicts.
    input_nodes = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_gripper,
            on_exit=[
                # Keyboard+mouse (pynput needs its own process)
                ExecuteProcess(
                    cmd=[
                        'bash', '-c',
                        'source ~/ros2_ws/install/setup.bash && '
                        'ros2 run vision_input virtual_operator '
                        '--ros-args -p use_sim_time:=true'
                    ],
                    output='screen',
                ),
                # Gamepad driver
                Node(
                    package='joy',
                    executable='joy_node',
                    name='joy_node',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Gamepad operator
                Node(
                    package='vision_input',
                    executable='gamepad_operator',
                    name='gamepad_operator',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Robot controller: follows virtual hand fingertip via TF
                Node(
                    package='vision_input',
                    executable='robot_controller',
                    name='robot_controller',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'use_gazebo_sim': True,
                        'use_ur_driver': True,
                        'use_virtual_operator': True,
                        'use_tf_tracking': True,
                    }],
                ),
                # Fatigue monitoring
                Node(
                    package='vision_input',
                    executable='tobii_node',
                    name='tobii_node',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'simulation_mode': True,
                        'fatigue_ramp_minutes': 3.0,
                    }],
                ),
                Node(
                    package='vision_input',
                    executable='fatigue_monitor',
                    name='fatigue_monitor',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Numpad fatigue override (1-4 to force levels)
                ExecuteProcess(
                    cmd=[
                        'bash', '-c',
                        'source ~/ros2_ws/install/setup.bash && '
                        'ros2 run vision_input fatigue_override'
                    ],
                    output='screen',
                ),
            ],
        )
    )

    return LaunchDescription([
        OpaqueFunction(function=setup_gazebo_model_path),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', '/tmp/intention_gazebo_models'),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',
            '/home/ranidu/ros2_ws/install/ros2_linkattacher/lib'
            + ':' + os.path.join(pkg_gazebo, '..', '..', 'lib')),

        gazebo,
        robot_state_publisher,
        virtual_hand_rsp,
        camera_tf,
        spawn_robot,

        # Controller chain
        TimerAction(period=7.0, actions=[spawn_jsb]),
        chain_jtc_after_jsb,
        chain_gripper_after_jtc,

        send_home,
        rviz,
        input_nodes,
        spawn_virtual_hand,
    ])

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, TimerAction,
    SetEnvironmentVariable, OpaqueFunction, RegisterEventHandler
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue


def setup_gazebo_model_path(context, *args, **kwargs):
    model_dir = '/tmp/intention_gazebo_models'
    os.makedirs(model_dir, exist_ok=True)
    # intention_gazebo added so package://intention_gazebo/meshes/... resolves
    # (needed for the LinkerHand L25 virtual hand meshes)
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
    rviz_config = os.path.join(pkg_gazebo, 'config', 'gazebo_view.rviz')
    virtual_hand_xacro = os.path.join(pkg_gazebo, 'urdf', 'virtual_hand.urdf.xacro')

    # REMOVED: initial_positions_file reference
    robot_description = Command([
        'xacro ', urdf_file,
        ' simulation_controllers:=', controllers_yaml,
    ])

    # Virtual hand URDF (LinkerHand L25 meshes): only used when
    # use_virtual_operator:=true
    virtual_hand_description = Command(['xacro ', virtual_hand_xacro])

    fatigue_ramp_arg = DeclareLaunchArgument(
        'fatigue_ramp_minutes',
        default_value='10.0',
        description='Minutes for simulated fatigue ramp'
    )
    fatigue_ramp = LaunchConfiguration('fatigue_ramp_minutes')

    virtual_op_arg = DeclareLaunchArgument(
        'use_virtual_operator',
        default_value='false',
        description='true = virtual_operator_node (mouse+KB), false = camera_node (webcam)'
    )
    use_virtual_op = LaunchConfiguration('use_virtual_operator')

    gamepad_arg = DeclareLaunchArgument(
        'use_gamepad',
        default_value='false',
        description='true = gamepad_operator_node (Logitech F310 + joy_node)'
    )
    use_gamepad = LaunchConfiguration('use_gamepad')

    # Gazebo

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        # Note: do NOT load libgazebo_ros_state.so here. It is a *world*
        # plugin, not a system plugin, and loading it via -s logs
        # "incorrect plugin type" and never publishes the service.
        # Virtual hand pose is owned by the custom virtual_hand_joint_driver
        # plugin (intention_gazebo) which subscribes to /virtual_hand/base_pose.
        output='screen'
    )


    # Robot state publisher
  
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


    # Spawn robot

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


    # Controller spawners: chained

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
        OnProcessExit(
            target_action=spawn_jsb,
            on_exit=[spawn_jtc],
        )
    )

    chain_gripper_after_jtc = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_jtc,
            on_exit=[spawn_gripper],
        )
    )


    # Send robot to home position after controllers load
    # (replaces initial_positions_file)

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


    # Virtual hand (LinkerHand L25): only when use_virtual_operator:=true

    # 1. Dedicated RSP in its own namespace so it doesn't clash with the UR3 RSP.
    #    Subscribes to /virtual_hand/joint_states and publishes TFs for all
    #    22 hand links.  Active in ALL input modes, the hand is the system's
    #    3D perception of the operator, regardless of sensor.
    virtual_hand_rsp = Node(
        condition=IfCondition(use_virtual_op),
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

    # 2. Pose driver selection:
    #    - virtual_operator mode -> virtual_operator_node owns base_pose + joint_states
    #    - webcam mode           -> hand_mirror_node fuses left+right hand -> base_pose
    #    - gamepad mode          -> (no hand mirror, gamepad has no hand data)

    # 3. Spawn the hand in Gazebo from /virtual_hand/robot_description.
    #    Delayed so Gazebo and the UR3 are already up.
    #    Active in ALL modes (gamepad won't publish poses so hand stays at spawn).
    spawn_virtual_hand = TimerAction(
        period=8.0,
        actions=[
            Node(
                condition=IfCondition(use_virtual_op),
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


    # RViz + App nodes: after home command sent
    # Use TimerAction after gripper to give home command time

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

    # Shared params for robot_controller
    _robot_ctrl_base = {
        'use_sim_time': True,
        'use_ur_driver': True,
        'use_gazebo_sim': True,
        'map_offset_x': 0.30,
        'map_scale_y': 0.9,
        'map_scale_z': 0.10,
        'map_offset_z': 0.45,
        'use_depth_camera': False,
        'camera_placement': 'front_facing',
    }
    # Virtual operator / gamepad: all 3 axes come from one synthetic hand.
    # use_gamepad=True keeps the historical torch_override=True default so
    # this path behaves the same after the use_gamepad parameter was added.
    _robot_ctrl_synthetic = {**_robot_ctrl_base, 'use_virtual_operator': True, 'use_gamepad': True}
    # Webcam: bimanual: right hand Y,Z + left hand X (depth).
    # use_gamepad omitted -> defaults False -> torch_override=False -> landmark_callback fires.
    _robot_ctrl_webcam = {**_robot_ctrl_base, 'use_virtual_operator': False}

    app_nodes = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_gripper,
            on_exit=[
                # Real webcam path (default)
                # Only launch camera if neither virtual operator nor gamepad is active
                Node(
                    condition=IfCondition(
                        PythonExpression(["'", use_virtual_op, "' == 'false' and '", use_gamepad, "' == 'false'"])
                    ),
                    package='vision_input',
                    executable='camera_node',
                    name='camera_node',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Virtual operator path
                # Launched in its own terminal so pynput can capture
                # keyboard + mouse without competing with the launch console.
                ExecuteProcess(
                    condition=IfCondition(use_virtual_op),
                    cmd=[
                        'bash', '-c',
                        'source ~/ros2_ws/install/setup.bash && '
                        'ros2 run vision_input virtual_operator '
                        '--ros-args -p use_sim_time:=true'
                    ],
                    output='screen',
                ),
                # Gamepad operator path 
                Node(
                    condition=IfCondition(use_gamepad),
                    package='joy',
                    executable='joy_node',
                    name='joy_node',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                Node(
                    condition=IfCondition(use_gamepad),
                    package='vision_input',
                    executable='gamepad_operator',
                    name='gamepad_operator',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Hand mirror (webcam only)
                # Fuses left + right webcam hands into a single 3D
                # world pose for the virtual LinkerHand L25 in Gazebo.
                Node(
                    condition=IfCondition(
                        PythonExpression(["'", use_virtual_op, "' == 'false' and '", use_gamepad, "' == 'false'"])
                    ),
                    package='vision_input',
                    executable='hand_mirror',
                    name='hand_mirror',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'map_offset_x': 0.30,
                        'map_scale_y': 0.9,
                        'map_scale_z': 0.10,
                        'map_offset_z': 0.45,
                        'camera_placement': 'front_facing',
                    }],
                ),
                # Robot controller 
                # Webcam mode: bimanual (left hand = depth, right = Y,Z)
                Node(
                    condition=IfCondition(
                        PythonExpression(["'", use_virtual_op, "' == 'false' and '", use_gamepad, "' == 'false'"])
                    ),
                    package='vision_input',
                    executable='robot_controller',
                    name='robot_controller',
                    output='screen',
                    parameters=[_robot_ctrl_webcam],
                ),
                # Virtual operator / gamepad: all 3 axes from one hand
                Node(
                    condition=IfCondition(
                        PythonExpression(["'", use_virtual_op, "' == 'true' or '", use_gamepad, "' == 'true'"])
                    ),
                    package='vision_input',
                    executable='robot_controller',
                    name='robot_controller',
                    output='screen',
                    parameters=[_robot_ctrl_synthetic],
                ),
                Node(
                    package='vision_input',
                    executable='tobii_node',
                    name='tobii_node',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'simulation_mode': True,
                        'fatigue_ramp_minutes': fatigue_ramp,
                    }],
                ),
                Node(
                    package='vision_input',
                    executable='fatigue_monitor',
                    name='fatigue_monitor',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
                # Fatigue override (numpad 1-4 to force levels)
                # Runs in its own terminal so pynput's global keyboard
                # hook captures keys regardless of window focus.
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
        fatigue_ramp_arg,
        virtual_op_arg,
        gamepad_arg,

        OpaqueFunction(function=setup_gazebo_model_path),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', '/tmp/intention_gazebo_models'),
        # IFRA LinkAttacher plugin lives in ros2_linkattacher's install lib dir
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',
            '/home/ranidu/ros2_ws/install/ros2_linkattacher/lib'
            + ':' + os.path.join(pkg_gazebo, '..', '..', 'lib')),

        gazebo,
        robot_state_publisher,
        camera_tf,
        spawn_robot,

        TimerAction(period=7.0, actions=[spawn_jsb]),
        chain_jtc_after_jsb,
        chain_gripper_after_jtc,

        send_home,
        rviz,
        app_nodes,

        # Virtual hand (only active when use_virtual_operator:=true)
        virtual_hand_rsp,
        spawn_virtual_hand,
    ])

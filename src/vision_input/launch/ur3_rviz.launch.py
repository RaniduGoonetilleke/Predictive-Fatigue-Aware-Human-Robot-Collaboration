import os
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    
    # Robot Description

    description_package = "vision_input"
    
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            get_package_share_directory("vision_input"), 
            "urdf", "ur3_with_gripper.urdf.xacro" 
        ]),
        " ", "name:=", "ur",
        " ", "ur_type:=", "ur3",
    ])
    robot_description = {"robot_description": robot_description_content}
    
    pkg_share = get_package_share_directory('vision_input')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'view_robot.rviz')
    
    # Nodes
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=['-d', rviz_config_file],
    )
    
  
    # CORRECTED CAMERA TRANSFORM
    # You are IN FRONT of the robot (+X direction)
    # Camera is to your side, looking at you and workspace
    
    camera_x = 0.6    # In front of robot (positive X) - near where you stand
    camera_y = 0.5    # To your left (robot's left = positive Y)
    camera_z = 0.4    # Above the table surface
    
    # Camera faces toward workspace (between you and robot)
    # Yaw of ~-120° (about -2.1 radians) points camera toward origin/workspace
    camera_yaw = -2.1   # Facing back toward robot and workspace
    camera_pitch = 0.2  # Slight downward tilt to see hands
    camera_roll = 0.0
    
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_tf",
        arguments=[
            str(camera_x), str(camera_y), str(camera_z),
            str(camera_yaw), str(camera_pitch), str(camera_roll),
            "base_link", "camera_link"
        ]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        rviz,
        camera_tf,
    ])
#!/usr/bin/env python3
"""
AR Pointing Demo Launch File
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node


def generate_launch_description():
    
    # PATHS
    pkg_share = get_package_share_directory('vision_input')
    rviz_config = os.path.join(pkg_share, 'rviz', 'ar_pointing.rviz')
    
    # ROBOT DESCRIPTION
    ur_description_pkg = get_package_share_directory('ur_description')
    
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([ur_description_pkg, 'urdf', 'ur.urdf.xacro']),
        ' ', 'name:=ur',
        ' ', 'ur_type:=ur3',
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # NODES
    
    # Robot State Publisher (publishes URDF transforms)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # AR Pointing Interface (camera + overlay)
    ar_pointing_interface = Node(
        package='vision_input',
        executable='ar_pointing_interface',
        name='ar_pointing_interface',
        output='screen'
    )
    
    # AR Robot Controller (moves robot based on AR targets)
    ar_robot_controller = Node(
        package='vision_input',
        executable='ar_robot_controller',
        name='ar_robot_controller',
        output='screen'
    )
    
    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher,
        ar_robot_controller,
        ar_pointing_interface,
        rviz,
    ])
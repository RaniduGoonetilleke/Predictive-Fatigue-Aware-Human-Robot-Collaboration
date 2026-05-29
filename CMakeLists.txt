cmake_minimum_required(VERSION 3.8)
project(intention_gazebo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Custom Gazebo model plugin: drives the virtual hand's joint positions
# from /virtual_hand/joint_states using Model::SetJointPositions.
# Avoids the global-physics-disable behavior of the standard
# gazebo_ros_joint_pose_trajectory plugin.
add_library(virtual_hand_joint_driver SHARED
  src/virtual_hand_joint_driver.cpp
)
target_include_directories(virtual_hand_joint_driver PUBLIC
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(virtual_hand_joint_driver
  ${GAZEBO_LIBRARIES}
)
ament_target_dependencies(virtual_hand_joint_driver
  rclcpp
  sensor_msgs
  geometry_msgs
)

install(TARGETS virtual_hand_joint_driver
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch
  urdf
  config
  worlds
  meshes
  DESTINATION share/${PROJECT_NAME}
)

# Add our install lib dir to GAZEBO_PLUGIN_PATH so Gazebo can find the
# plugin .so when launched after sourcing install/setup.bash.
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gazebo_plugin_path.sh.in"
)

ament_package()
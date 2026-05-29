// virtual_hand_joint_driver.cpp
//
// Gazebo Classic model plugin that drives BOTH the finger joints and the
// base world pose of the virtual_hand model from ROS2 topics.
//
// Why this exists: ROS2 gazebo_ros_pkgs has no clean way to set joint
// positions on a multi-joint visual model without fighting physics or
// freezing the global world. And /gazebo/set_entity_state (provided by
// libgazebo_ros_state.so) is a world plugin, not a system plugin, so
// loading it via -s prints "incorrect plugin type" and never publishes
// the service. Owning both fingers + pose here in one plugin sidesteps
// both problems and guarantees the hand cannot drift from reaction
// torques because we re-apply the world pose every Gazebo update tick.
//
// Subscribes to:
//   /virtual_hand/joint_states  sensor_msgs/JointState  (finger angles)
//   /virtual_hand/base_pose     geometry_msgs/Pose      (world pose)

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Joint.hh>
#include <ignition/math/Pose3.hh>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>

namespace intention_gazebo
{

class VirtualHandJointDriver : public gazebo::ModelPlugin
{
public:
  VirtualHandJointDriver() = default;

  ~VirtualHandJointDriver() override
  {
    update_conn_.reset();
    if (executor_) {
      executor_->cancel();
    }
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
  }

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;

    std::string joint_topic = "/virtual_hand/joint_states";
    if (sdf->HasElement("joint_topic")) {
      joint_topic = sdf->Get<std::string>("joint_topic");
    } else if (sdf->HasElement("topic")) {
      joint_topic = sdf->Get<std::string>("topic");
    }

    std::string pose_topic = "/virtual_hand/base_pose";
    if (sdf->HasElement("pose_topic")) {
      pose_topic = sdf->Get<std::string>("pose_topic");
    }

    // Initial pose = the model's spawn pose. Held until the first
    // /virtual_hand/base_pose message arrives.
    target_pose_ = model_->WorldPose();

    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    ros_node_ = std::make_shared<rclcpp::Node>(
      "virtual_hand_joint_driver_" + model->GetName());

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    js_sub_ = ros_node_->create_subscription<sensor_msgs::msg::JointState>(
      joint_topic, qos,
      std::bind(&VirtualHandJointDriver::OnJointState, this,
                std::placeholders::_1));
    pose_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Pose>(
      pose_topic, qos,
      std::bind(&VirtualHandJointDriver::OnPose, this,
                std::placeholders::_1));

    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(ros_node_);
    spin_thread_ = std::thread([this]() { executor_->spin(); });

    // Re-apply target world pose every physics tick. This is what kills
    // the slow rotation: any reaction torque the joint snap might apply
    // to the base is overwritten before it can integrate.
    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&VirtualHandJointDriver::OnWorldUpdate, this));

    gzmsg << "[VirtualHandJointDriver] attached to model '"
          << model->GetName() << "'\n"
          << "  joint topic: " << joint_topic << "\n"
          << "  pose  topic: " << pose_topic << "\n";
  }

private:
  void OnJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (!model_) {
      return;
    }

    std::map<std::string, double> positions;
    const size_t n = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
      auto joint = model_->GetJoint(msg->name[i]);
      if (joint) {
        positions[joint->GetScopedName()] = msg->position[i];
      }
    }

    if (positions.empty()) {
      return;
    }

    // Cache latest joint positions so OnWorldUpdate can re-apply them
    // every physics tick — prevents ODE solver from perturbing the
    // tiny-inertia finger links between ROS messages.
    {
      std::lock_guard<std::mutex> lock(joint_mutex_);
      target_joints_ = positions;
      joints_received_ = true;
    }

    model_->SetJointPositions(positions);
  }

  void OnPose(const geometry_msgs::msg::Pose::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    target_pose_ = ignition::math::Pose3d(
      msg->position.x, msg->position.y, msg->position.z,
      msg->orientation.w, msg->orientation.x,
      msg->orientation.y, msg->orientation.z);
  }

  void OnWorldUpdate()
  {
    if (!model_) {
      return;
    }

    // Re-apply base world pose every tick (prevents rotation drift)
    ignition::math::Pose3d pose;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      pose = target_pose_;
    }
    model_->SetWorldPose(pose);

    // Re-apply finger joint positions every tick (prevents finger shaking
    // caused by ODE solver perturbing tiny-inertia links between messages)
    {
      std::lock_guard<std::mutex> lock(joint_mutex_);
      if (joints_received_) {
        model_->SetJointPositions(target_joints_);
      }
    }
  }

  gazebo::physics::ModelPtr model_;
  gazebo::event::ConnectionPtr update_conn_;

  std::mutex pose_mutex_;
  ignition::math::Pose3d target_pose_;

  std::mutex joint_mutex_;
  std::map<std::string, double> target_joints_;
  bool joints_received_ = false;

  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spin_thread_;
};

GZ_REGISTER_MODEL_PLUGIN(VirtualHandJointDriver)

}  // namespace intention_gazebo

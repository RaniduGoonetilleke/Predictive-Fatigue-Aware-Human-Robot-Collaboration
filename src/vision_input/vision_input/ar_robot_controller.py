#!/usr/bin/env python3
"""
AR Robot Controller

Controls the UR3 robot based on AR target points.
Publishes workspace markers including corner numbers.

Author: Ranidu P. Goonetilleke
Date: February 2025
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import math


class ARRobotController(Node):
    
    def __init__(self):
        super().__init__('ar_robot_controller')
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("   AR ROBOT CONTROLLER")
        self.get_logger().info("=" * 50)
        
        # SUBSCRIBERS 
        self.target_sub = self.create_subscription(
            PointStamped,
            '/ar_target_point',
            self.target_callback,
            10
        )
        
        # PUBLISHERS
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/workspace_markers', 10)
        
        # ROBOT CONFIGURATION
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.current_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        self.target_joints = self.current_joints.copy()
        
        # WORKSPACE LIMITS
        self.MAX_REACH = 0.45
        self.MIN_REACH = 0.12
        self.MIN_Z = 0.02
        self.MAX_Z = 0.40
        
        # MOTION PARAMETERS
        self.smoothing = 0.08
        
        # UR3 LINK LENGTHS 
        self.L1 = 0.244
        self.L2 = 0.213
        
        # STATE
        self.last_target = None
        
        # TIMERS
        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.viz_timer = self.create_timer(0.2, self.publish_workspace_viz)
        
        self.get_logger().info("")
        self.get_logger().info("✅ Robot Controller Ready!")
        self.get_logger().info("")
    
    def target_callback(self, msg: PointStamped):
        """Receive target from AR interface."""
        
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        x, y, z, was_clamped = self.clamp_to_workspace(x, y, z)
        
        self.last_target = (x, y, z)
        
        success = self.solve_ik(x, y, z)
        
        if success:
            status = "✓" if not was_clamped else "⚠ (clamped)"
            self.get_logger().info(
                f"Target: ({x:+.3f}, {y:+.3f}, {z:+.3f}) {status}"
            )
    
    def clamp_to_workspace(self, x, y, z):
        """Ensure target is within robot workspace."""
        
        was_clamped = False
        
        if z < self.MIN_Z:
            z = self.MIN_Z
            was_clamped = True
        elif z > self.MAX_Z:
            z = self.MAX_Z
            was_clamped = True
        
        horizontal_dist = math.sqrt(x**2 + y**2)
        total_dist = math.sqrt(x**2 + y**2 + z**2)
        
        if total_dist > self.MAX_REACH:
            scale = self.MAX_REACH / total_dist
            x *= scale
            y *= scale
            z = max(self.MIN_Z, z * scale)
            was_clamped = True
        
        if horizontal_dist < self.MIN_REACH:
            if horizontal_dist > 0.01:
                scale = self.MIN_REACH / horizontal_dist
                x *= scale
                y *= scale
            else:
                x = self.MIN_REACH
                y = 0.0
            was_clamped = True
        
        return x, y, z, was_clamped
    
    def solve_ik(self, x, y, z):
        """Solve inverse kinematics."""
        
        try:
            q1 = math.atan2(y, x)
            
            r = math.sqrt(x**2 + y**2)
            d = math.sqrt(r**2 + z**2)
            d = min(d, self.L1 + self.L2 - 0.01)
            d = max(d, abs(self.L1 - self.L2) + 0.01)
            
            cos_q3 = (self.L1**2 + self.L2**2 - d**2) / (2 * self.L1 * self.L2)
            cos_q3 = max(-1.0, min(1.0, cos_q3))
            q3 = math.pi - math.acos(cos_q3)
            
            cos_beta = (self.L1**2 + d**2 - self.L2**2) / (2 * self.L1 * d)
            cos_beta = max(-1.0, min(1.0, cos_beta))
            beta = math.acos(cos_beta)
            
            alpha = math.atan2(z, r)
            q2 = -math.pi/2 + alpha + beta
            
            q4 = -q2 - q3 - 0.5
            q5 = -math.pi / 2
            q6 = 0.0
            
            self.target_joints = [q1, q2, q3, q4, q5, q6]
            
            return True
            
        except (ValueError, ZeroDivisionError) as e:
            self.get_logger().warn(f"IK failed: {e}")
            return False
    
    def control_loop(self):
        """Smooth motion control loop."""
        
        for i in range(len(self.current_joints)):
            diff = self.target_joints[i] - self.current_joints[i]
            self.current_joints[i] += self.smoothing * diff
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_joints
        
        self.joint_pub.publish(msg)
    
    def publish_workspace_viz(self):
        """Publish workspace visualization (reach bubble only)."""
        
        now = self.get_clock().now().to_msg()
        
        # Reach bubble
        bubble = Marker()
        bubble.header.frame_id = "base_link"
        bubble.header.stamp = now
        bubble.ns = "workspace"
        bubble.id = 0
        bubble.type = Marker.SPHERE
        bubble.action = Marker.ADD
        
        bubble.pose.position.z = 0.0
        bubble.pose.orientation.w = 1.0
        
        bubble.scale.x = self.MAX_REACH * 2
        bubble.scale.y = self.MAX_REACH * 2
        bubble.scale.z = self.MAX_REACH * 2
        
        bubble.color.r = 0.0
        bubble.color.g = 0.5
        bubble.color.b = 1.0
        bubble.color.a = 0.08
        
        bubble.lifetime.sec = 1
        self.marker_pub.publish(bubble)
        
        # Current target marker (green sphere)
        if self.last_target is not None:
            target = Marker()
            target.header.frame_id = "base_link"
            target.header.stamp = now
            target.ns = "target"
            target.id = 1
            target.type = Marker.SPHERE
            target.action = Marker.ADD
            
            target.pose.position.x = self.last_target[0]
            target.pose.position.y = self.last_target[1]
            target.pose.position.z = self.last_target[2]
            target.pose.orientation.w = 1.0
            
            target.scale.x = 0.03
            target.scale.y = 0.03
            target.scale.z = 0.03
            
            target.color.r = 0.0
            target.color.g = 1.0
            target.color.b = 0.0
            target.color.a = 1.0
            
            target.lifetime.sec = 1
            self.marker_pub.publish(target)


def main(args=None):
    rclpy.init(args=args)
    node = ARRobotController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
gamepad_sandbox.py — Pure Teleop Sandbox
Stripped-down controller: gamepad sticks -> IK -> robot arm.
No fatigue, no fetch, no geofence, no objects, no dwell timer.

Usage:
  ros2 launch intention_gazebo gazebo_sim.launch.py use_gamepad:=true
  # Then in a separate terminal (after killing robot_controller):
  ros2 run vision_input gamepad_sandbox
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math


class GamepadSandbox(Node):
    def __init__(self):
        super().__init__('gamepad_sandbox')

        # UR3 DH parameters
        self.d1 = 0.1519
        self.d4 = 0.11235
        self.L1 = 0.24365
        self.L2 = 0.21325

        # Workspace limits
        self.MIN_Z = 0.10
        self.MAX_Z = 0.52
        self.MAX_REACH = 0.50
        self.MIN_REACH = 0.18

        # Robot position (start at home-ish)
        self.robot_x = 0.30
        self.robot_y = 0.00
        self.robot_z = 0.40

        # Stick tuning
        self.stick_speed = 0.25   # metres per second at full deflection
        self.deadzone = 0.12

        # Smoothing
        self.smooth_x = self.robot_x
        self.smooth_y = self.robot_y
        self.smooth_z = self.robot_z
        self.alpha = 0.15  # EMA smoothing (0=frozen, 1=instant)

        # Joint state
        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.target_pos = [0.324, -1.240, 0.793, -1.135, -1.595, 0.217]
        self.joint_state_received = False
        self.deadman_held = False

        # Publishers
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Subscribers
        self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.js_cb, 10)

        # Control loop at 20Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            "SANDBOX READY — Hold LB + sticks to move. "
            "LX=left/right, RY=fwd/back, LT=up, RT=down")

    def js_cb(self, msg: JointState):
        if not self.joint_state_received:
            joint_dict = dict(zip(msg.name, msg.position))
            for i, name in enumerate(self.ur_joint_names):
                if name in joint_dict:
                    self.target_pos[i] = joint_dict[name]
            self.joint_state_received = True
            self.get_logger().info("Joint states synced!")

    def joy_cb(self, msg: Joy):
        axes = msg.axes
        buttons = msg.buttons
        if len(axes) < 6 or len(buttons) < 5:
            return

        self.deadman_held = bool(buttons[4])  # LB

        if not self.deadman_held:
            return

        dt = 0.05
        speed = self.stick_speed * dt

        # Left stick X -> Y (left/right)
        lx = axes[0]
        if abs(lx) > self.deadzone:
            self.robot_y += -lx * speed

        # Right stick Y -> X (forward/back)
        ry = axes[4]
        if abs(ry) > self.deadzone:
            self.robot_x += ry * speed

        # LT (axes[2]) -> up, RT (axes[5]) -> down
        lt = (1.0 - axes[2]) / 2.0
        rt = (1.0 - axes[5]) / 2.0
        if lt > 0.05:
            self.robot_z += lt * speed
        if rt > 0.05:
            self.robot_z -= rt * speed

        # Clamp to workspace: Z first, then limit horizontal to what's left
        self.robot_z = max(self.MIN_Z, min(self.MAX_Z, self.robot_z))
        z_local = self.robot_z - self.d1
        h_max_sq = (self.L1 + self.L2 - 0.01)**2 - z_local**2
        h_max = math.sqrt(max(0.01, h_max_sq))
        h = math.sqrt(self.robot_x**2 + self.robot_y**2)
        if h > h_max:
            scale = h_max / h
            self.robot_x *= scale
            self.robot_y *= scale
        if h < self.MIN_REACH and h > 0.001:
            scale = self.MIN_REACH / h
            self.robot_x *= scale
            self.robot_y *= scale

    def control_loop(self):
        if not self.joint_state_received or not self.deadman_held:
            return

        # EMA smooth toward target
        self.smooth_x += self.alpha * (self.robot_x - self.smooth_x)
        self.smooth_y += self.alpha * (self.robot_y - self.smooth_y)
        self.smooth_z += self.alpha * (self.robot_z - self.smooth_z)

        # IK
        self.solve_ik(self.smooth_x, self.smooth_y, self.smooth_z)

        # Send trajectory
        traj = JointTrajectory()
        traj.joint_names = self.ur_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in self.target_pos]
        pt.time_from_start = Duration(sec=0, nanosec=50_000_000)
        traj.points = [pt]
        self.traj_pub.publish(traj)

    def solve_ik(self, x, y, z):
        """Analytical UR3 IK: same math as robot_controller."""
        d1, d4, L1, L2 = self.d1, self.d4, self.L1, self.L2

        # Joint 1
        rho_sq = x*x + y*y
        min_rho = d4 + 0.005
        if rho_sq < min_rho * min_rho:
            theta = math.atan2(y, x) if rho_sq > 1e-6 else 0.0
            x = min_rho * math.cos(theta)
            y = min_rho * math.sin(theta)
            rho_sq = min_rho * min_rho

        phi = math.atan2(y, x)
        psi = math.atan2(d4, math.sqrt(rho_sq - d4*d4))
        self.target_pos[0] = phi - psi

        # Joints 2, 3
        r = math.sqrt(rho_sq - d4*d4)
        z_local = z - d1
        d = math.sqrt(r*r + z_local*z_local)

        max_d = L1 + L2 - 0.005
        if d > max_d:
            # Keep Z fixed: only shrink horizontal reach
            r_max_sq = max_d**2 - z_local**2
            r = math.sqrt(max(0.01, r_max_sq))
            d = max_d
        d = max(0.05, d)

        cos_elbow = (d*d - L1*L1 - L2*L2) / (2 * L1 * L2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        q3 = math.acos(cos_elbow)
        target_angle = math.atan2(z_local, r)
        inner_offset = math.atan2(L2*math.sin(q3), L1 + L2*math.cos(q3))
        q2 = -(target_angle + inner_offset)

        self.target_pos[1] = q2
        self.target_pos[2] = q3
        self.target_pos[3] = -math.pi/2 - q2 - q3  # gripper down
        self.target_pos[4] = -math.pi/2
        self.target_pos[5] = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GamepadSandbox()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

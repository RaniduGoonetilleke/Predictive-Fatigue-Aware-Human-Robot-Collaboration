#!/usr/bin/env python3
"""
gamepad_operator_node.py — Logitech F310 Gamepad Operator
Human Intention Prediction System

Industrial-grade gamepad input for UR3 teleoperation with:
  - Deadman's switch (LB) — robot only moves while held
  - Fatigue-adaptive dampening — stick sensitivity scales with operator fatigue
  - Same topic format as camera_node / virtual_operator — drop-in compatible

Topics published (exact camera_node format):
  /vision_input/hand_landmarks          Float32MultiArray  63 floats
  /vision_input/right_hand_landmarks    Float32MultiArray  63 floats
  /vision_input/left_hand_landmarks     Float32MultiArray  63 zeros
  /vision_input/gesture_text            String
  /vision_input/hand_velocity           Float32MultiArray  7 floats

Topics subscribed:
  /joy                                  sensor_msgs/Joy    (from joy_node)
  /fatigue/score                        Float32            (from fatigue_monitor)
  /fatigue/level                        String             (from fatigue_monitor)

Logitech F310 (X mode) mapping:
  Left stick X      -> Robot Y (left/right)
  Right stick Y     -> Robot X (forward/back)
  LT (axis 2)       -> Robot Z up
  RT (axis 5)       -> Robot Z down
  LB (button 4)     -> Deadman's switch, MUST hold to move
  RB (button 5)     -> POINT gesture
  A  (button 0)     -> OKAY gesture (confirm fetch / release)
  B  (button 1)     -> STOP (E-stop)
  X  (button 2)     -> GRASP gesture
  Y  (button 3)     -> NEUTRAL gesture
  Back (button 6)   -> Toggle mode: TORCH ↔ FETCH
  Start (button 7)  -> Return to NEUTRAL
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Float32, Bool
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformBroadcaster
import math


# Fatigue dampening multipliers: how much to scale stick sensitivity
FATIGUE_DAMPENING = {
    "FRESH":    1.0,
    "MILD":     0.70,
    "MODERATE": 0.40,
    "SEVERE":   0.15,
}

# Deadzone: ignore tiny stick drift
STICK_DEADZONE = 0.12

# LinkerHand L25 joint names (matches virtual_hand.urdf.xacro)
L25_JOINT_NAMES = [
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip', 'index_dip',
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip', 'middle_dip',
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip', 'ring_dip',
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip', 'pinky_dip',
]

GESTURE_POSES = {
    "POINT": [
         0.00,  0.80,  1.20,  1.20,  0.80,
         0.00,  0.00,  0.00,  0.00,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.40,  1.10,  1.40,
    ],
    "STOP": [
         0.40,  0.20,  0.10,  0.10,  0.05,
         0.10,  0.00,  0.00,  0.00,
         0.00,  0.00,  0.00,  0.00,
        -0.05,  0.00,  0.00,  0.00,
        -0.10,  0.00,  0.00,  0.00,
    ],
    "OKAY": [
         0.00,  0.00,  0.00,  0.00,  0.00,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.40,  1.10,  1.40,
         0.00,  1.30,  1.00,  1.30,
    ],
    "GRASP": [
         0.10,  0.80,  1.20,  1.20,  0.80,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.50,  1.20,  1.50,
         0.00,  1.40,  1.10,  1.40,
         0.00,  1.30,  1.00,  1.30,
    ],
    "NEUTRAL": [
         0.00,  0.15,  0.10,  0.10,  0.05,
         0.00,  0.20,  0.15,  0.10,
         0.00,  0.20,  0.15,  0.10,
         0.00,  0.20,  0.15,  0.10,
         0.00,  0.25,  0.18,  0.12,
    ],
}

# Default hand orientation: palm-down, fingers toward robot (-X)
_DEFAULT_ROLL  = 0.0
_DEFAULT_PITCH = math.pi / 2.0
_DEFAULT_YAW   = math.pi


def _rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class GamepadOperatorNode(Node):
    def __init__(self):
        super().__init__('gamepad_operator')

        # State
        # Tip position in normalized camera space (0-1)
        self.tip_x = 0.5     # centre
        self.tip_y = 0.5     # centre
        self.depth = 0.5     # mid depth

        self.gesture = "NEUTRAL"
        self.prev_gesture = "NEUTRAL"
        self.deadman_held = False
        self.torch_held = False
        self.gripper_open = True   # toggle state for RB

        # Fatigue
        self.fatigue_score = 0.0
        self.fatigue_level = "FRESH"

        # Stick sensitivity (units per second at FRESH)
        self.stick_speed = 0.3

        # Previous state for velocity calculation
        self.prev_tip_x = 0.5
        self.prev_tip_y = 0.5
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        # Button edge detection (prevent repeated triggers)
        self._prev_buttons = [0] * 12

        # Publishers (exact camera_node format)
        self.landmark_pub = self.create_publisher(
            Float32MultiArray, '/vision_input/hand_landmarks', 10)
        self.right_lm_pub = self.create_publisher(
            Float32MultiArray, '/vision_input/right_hand_landmarks', 10)
        self.left_lm_pub = self.create_publisher(
            Float32MultiArray, '/vision_input/left_hand_landmarks', 10)
        self.gesture_pub = self.create_publisher(
            String, '/vision_input/gesture_text', 10)
        self.velocity_pub = self.create_publisher(
            Float32MultiArray, '/vision_input/hand_velocity', 10)
        self.torch_override_pub = self.create_publisher(
            Bool, '/vision_input/torch_override', 10)
        self.gripper_pub = self.create_publisher(
            Float32, '/vision_input/gripper_command', 10)

        # Virtual hand publishers
        self.hand_pose_pub = self.create_publisher(
            Pose, '/virtual_hand/base_pose', 10)
        self.hand_joint_pub = self.create_publisher(
            JointState, '/virtual_hand/joint_states', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Virtual hand anchor + orientation (palm-down, fingers toward robot)
        self._hand_anchor = (0.80, 0.00, 0.60)
        self._qx, self._qy, self._qz, self._qw = _rpy_to_quat(
            _DEFAULT_ROLL, _DEFAULT_PITCH, _DEFAULT_YAW)

        # Subscribers
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.create_subscription(
            Float32, '/fatigue/score', self.fatigue_score_cb, 10)
        self.create_subscription(
            String, '/fatigue/level', self.fatigue_level_cb, 10)

        # Publish timer (20Hz: matches virtual operator)
        self.create_timer(0.05, self.publish_tick)

        self.get_logger().info(
            "GAMEPAD OPERATOR ready — hold LB (deadman) to move robot")


    # Fatigue callbacks

    def fatigue_score_cb(self, msg: Float32):
        self.fatigue_score = msg.data

    def fatigue_level_cb(self, msg: String):
        if msg.data != self.fatigue_level:
            old = self.fatigue_level
            self.fatigue_level = msg.data
            dampening = FATIGUE_DAMPENING.get(self.fatigue_level, 1.0)
            self.get_logger().warn(
                f"FATIGUE: {old} -> {self.fatigue_level} "
                f"(stick dampening: {dampening:.0%})")
        self.fatigue_level = msg.data

    # Joy callback: read sticks and buttons

    def joy_callback(self, msg: Joy):
        axes = msg.axes
        buttons = msg.buttons

        if len(axes) < 6 or len(buttons) < 8:
            return

        # Deadman's switch (LB = button 4)
        self.deadman_held = bool(buttons[4])

        # Button edge detection (trigger on press, not hold)
        def just_pressed(idx):
            pressed = bool(buttons[idx]) and not bool(self._prev_buttons[idx])
            return pressed

        # Mode toggle: TORCH / FETCH (Back = button 6) 
        if just_pressed(6):
            self.torch_held = not self.torch_held
            mode = "TORCH" if self.torch_held else "FETCH"
            self.get_logger().info(f"MODE: {mode}")

        # Gesture buttons 
        if just_pressed(0):   # A -> OKAY
            self.gesture = "OKAY"
        elif just_pressed(2):  # X -> GRASP
            self.gesture = "GRASP"
        elif just_pressed(1):  # B -> STOP
            self.gesture = "STOP"
        elif just_pressed(3):  # Y -> NEUTRAL
            self.gesture = "NEUTRAL"
        elif just_pressed(5):  # RB -> POINT
            self.gesture = "POINT"

        # Analog sticks & triggers (only when deadman held)
        if self.deadman_held:
            dampening = FATIGUE_DAMPENING.get(self.fatigue_level, 1.0)
            dt = 0.05  # ~20Hz
            speed = self.stick_speed * dampening * dt

            # Left stick X -> Robot Y (push left = hand moves left)
            lx = axes[0]
            if abs(lx) > STICK_DEADZONE:
                self.tip_x += lx * speed

            # Right stick Y -> Robot X (pull back = increase X, toward operator)
            ry = axes[4]
            if abs(ry) > STICK_DEADZONE:
                self.depth += -ry * speed

            # LT (axes[2]) -> Z up, RT (axes[5]) -> Z down
            # Triggers: 1.0 (released) -> -1.0 (fully pressed)
            lt_val = (1.0 - axes[2]) / 2.0
            rt_val = (1.0 - axes[5]) / 2.0

            if lt_val > 0.05:
                self.tip_y -= lt_val * speed
            if rt_val > 0.05:
                self.tip_y += rt_val * speed

            # Wide clamp: let the hand reach across the full table
            self.tip_x = max(-0.5, min(1.5, self.tip_x))
            self.tip_y = max(-0.5, min(1.5, self.tip_y))
            self.depth = max(-0.5, min(1.5, self.depth))

            # Gesture stays NEUTRAL until operator explicitly presses a button

        # Save button state for edge detection
        self._prev_buttons = list(buttons)

    # Build 63-float landmark array (same as virtual_operator)

    def _build_landmark_array(self, tx, ty, d):
        """
        Build 63-float hand landmark array.
        data[24]=tx (index tip X), data[25]=ty (index tip Y), data[26]=d (depth)
        All other landmarks form a plausible POINT-gesture skeleton.
        """
        wx = tx - 0.05
        wy = min(1.0, ty + 0.16)
        lms = [
            (wx,         wy,        0.00),   # 0 WRIST
            (wx - 0.04,  wy - 0.02, 0.00),   # 1 THUMB_CMC
            (wx - 0.06,  wy - 0.06, 0.00),   # 2 THUMB_MCP
            (wx - 0.08,  wy - 0.09, 0.00),   # 3 THUMB_IP
            (wx - 0.09,  wy - 0.12, 0.00),   # 4 THUMB_TIP
            (wx + 0.02,  wy - 0.05, 0.00),   # 5 INDEX_MCP
            (wx + 0.02,  wy - 0.09, 0.00),   # 6 INDEX_PIP
            (wx + 0.01,  wy - 0.13, 0.00),   # 7 INDEX_DIP
            (tx,         ty,        d),       # 8 INDEX_TIP (KEY)
            (wx + 0.04,  wy - 0.05, 0.00),   # 9 MIDDLE_MCP
            (wx + 0.05,  wy - 0.07, 0.00),   # 10 MIDDLE_PIP
            (wx + 0.05,  wy - 0.06, 0.00),   # 11 MIDDLE_DIP
            (wx + 0.05,  wy - 0.05, 0.00),   # 12 MIDDLE_TIP
            (wx + 0.05,  wy - 0.05, 0.00),   # 13 RING_MCP
            (wx + 0.06,  wy - 0.06, 0.00),   # 14 RING_PIP
            (wx + 0.06,  wy - 0.05, 0.00),   # 15 RING_DIP
            (wx + 0.06,  wy - 0.04, 0.00),   # 16 RING_TIP
            (wx + 0.06,  wy - 0.04, 0.00),   # 17 PINKY_MCP
            (wx + 0.07,  wy - 0.04, 0.00),   # 18 PINKY_PIP
            (wx + 0.07,  wy - 0.03, 0.00),   # 19 PINKY_DIP
            (wx + 0.07,  wy - 0.02, 0.00),   # 20 PINKY_TIP
        ]
        flat = []
        for (x, y, z) in lms:
            flat.extend([x, y, z])
        return flat

    # Publish at 20Hz
   
    def publish_tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if now > self.prev_time else 0.05

        # Torch override (always publish, even without deadman)
        torch_msg = Bool()
        torch_msg.data = self.torch_held
        self.torch_override_pub.publish(torch_msg)

        # Gripper command (always publish) 
        grip_msg = Float32()
        grip_msg.data = 0.0 if self.gripper_open else 0.4
        self.gripper_pub.publish(grip_msg)

        # If deadman not held -> keep position, just reset gesture
        if not self.deadman_held:
            self.prev_time = now
            self.gesture = "NEUTRAL"
            return

        pub_gesture = self.gesture

        # Landmarks
        lm_data = self._build_landmark_array(
            self.tip_x, self.tip_y, self.depth)
        lm_msg = Float32MultiArray()
        lm_msg.data = lm_data
        self.landmark_pub.publish(lm_msg)
        self.right_lm_pub.publish(lm_msg)

        left_msg = Float32MultiArray()
        left_msg.data = [0.0] * 63
        self.left_lm_pub.publish(left_msg)

        # Gesture
        g_msg = String()
        g_msg.data = pub_gesture
        self.gesture_pub.publish(g_msg)

        # Velocity 
        vx = (self.tip_x - self.prev_tip_x) / dt if dt > 0 else 0.0
        vy = (self.tip_y - self.prev_tip_y) / dt if dt > 0 else 0.0
        speed = math.sqrt(vx * vx + vy * vy)
        pred_x = self.tip_x + vx * 0.2
        pred_y = self.tip_y + vy * 0.2

        v_msg = Float32MultiArray()
        v_msg.data = [
            float(vx), float(vy), 0.0,
            float(speed),
            float(pred_x), float(pred_y), float(self.depth),
        ]
        self.velocity_pub.publish(v_msg)

        # Virtual hand
        self._update_virtual_hand(self.tip_x, self.tip_y, self.depth,
                                  pub_gesture)

        self.prev_tip_x = self.tip_x
        self.prev_tip_y = self.tip_y
        self.prev_time = now

        # Gesture persists until another button is pressed.
        # robot_controller handles one-shot transitions internally.

        # Log dampening periodically
        if not hasattr(self, '_log_count'):
            self._log_count = 0
        self._log_count += 1
        if self._log_count % 100 == 0:  # every 5s
            dampening = FATIGUE_DAMPENING.get(self.fatigue_level, 1.0)
            self.get_logger().info(
                f"[GAMEPAD] pos=({self.tip_x:.2f},{self.tip_y:.2f},{self.depth:.2f}) "
                f"gesture={pub_gesture} deadman={'ON' if self.deadman_held else 'OFF'} "
                f"fatigue={self.fatigue_level} dampening={dampening:.0%}")



    # Virtual hand: translate (tip_x, tip_y, depth) to world pose
    
    def _update_virtual_hand(self, tx, ty, d, gesture):
        ax, ay, az = self._hand_anchor
        # Same mapping as virtual_operator: normalized (0-1) -> world offset
        px = ax + (d  - 0.5) * 0.80   # depth range ~0.80m
        py = ay - (tx - 0.5) * 1.00   # lateral range ~1.00m
        pz = az + (0.5 - ty) * 0.60   # vertical range ~0.60m

        # Publish Pose (Gazebo plugin consumes this
        pose = Pose()
        pose.position.x = px
        pose.position.y = py
        pose.position.z = pz
        pose.orientation.x = self._qx
        pose.orientation.y = self._qy
        pose.orientation.z = self._qz
        pose.orientation.w = self._qw
        self.hand_pose_pub.publish(pose)

        # Publish TF (RViz consumes this)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'virtual_hand_base_link'
        tf.transform.translation.x = px
        tf.transform.translation.y = py
        tf.transform.translation.z = pz
        tf.transform.rotation.x = self._qx
        tf.transform.rotation.y = self._qy
        tf.transform.rotation.z = self._qz
        tf.transform.rotation.w = self._qw
        self._tf_broadcaster.sendTransform(tf)

        # Publish finger joint states
        poses = GESTURE_POSES.get(gesture, GESTURE_POSES["NEUTRAL"])
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = L25_JOINT_NAMES
        js.position = [float(v) for v in poses]
        self.hand_joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = GamepadOperatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

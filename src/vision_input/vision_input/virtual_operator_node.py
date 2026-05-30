#!/usr/bin/env python3
"""
virtual_operator_node.py — Virtual Operator for Gazebo Simulation
Human Intention Prediction System

Drop-in replacement for camera_node.py during Gazebo simulation.
Publishes identical topics via mouse+keyboard or scripted sequences.
Includes physiologically-realistic hand tremor model and full CSV logging.

Topics published (exact camera_node format):
  /vision_input/hand_landmarks          Float32MultiArray  63 floats
  /vision_input/right_hand_landmarks    Float32MultiArray  63 floats
  /vision_input/left_hand_landmarks     Float32MultiArray  63 zeros
  /vision_input/gesture_text            String
  /vision_input/hand_velocity           Float32MultiArray  7 floats
  /vision_input/virtual_hand_markers    MarkerArray (RViz hand visualization)

Controls:
  Space+Mouse -> tip_x, tip_y (hold Space to clutch, release to free mouse)
  Space+Scroll -> depth (only while Space held)
  W/S         -> depth +/- (forward/back)
  A/D         -> left/right
  E/C         -> up/down
  Arrows      -> rotate yaw/pitch (5° per press)
  < / >       -> rotate roll (5° per press)
  LShift      -> STOP (E-STOP, instant)
  p/o/g/n     -> gesture (POINT/OKAY/GRASP/NEUTRAL)
  1/2/3       -> quick positions (screwdriver/bolt/cup)
  h           -> center/home (position only)
  r           -> reset all (position + orientation + tremor)
  F1-F8       -> scripted sequences
  [/]         -> decrease/increase tremor amplitude
  backslash   -> toggle auto-tremor (scales with fatigue)
  l           -> toggle CSV logging
  v           -> toggle verbose
  +/-         -> sensitivity
  q           -> quit
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Float32
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformBroadcaster
import math
import time
import csv
import os
import random
import threading
import sys

# Optional: pynput for mouse/keyboard capture 
try:
    from pynput import mouse as _pynput_mouse, keyboard as _pynput_keyboard
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False

# Gazebo positioning is now handled by publishing geometry_msgs/Pose on
# /virtual_hand/base_pose, which the custom virtual_hand_joint_driver
# plugin (intention_gazebo) consumes inside Gazebo. No gazebo_msgs needed.

# Tremor amplitude per fatigue level

TREMOR_BY_LEVEL = {
    "FRESH":    0.000,   # rock-solid — healthy resting hand has no visible tremor
    "MILD":     0.008,
    "MODERATE": 0.018,
    "SEVERE":   0.035,
}


# Scripted Sequences: (time_s, tip_x, tip_y, depth, gesture, description)

FETCH_SEQUENCE = [
    (0.0,  0.50, 0.50, 0.50, "NEUTRAL", "Hand at rest"),
    (1.0,  0.50, 0.50, 0.50, "POINT",   "Begin pointing"),
    (2.0,  0.35, 0.60, 0.50, "POINT",   "Point toward screwdriver"),
    (4.0,  0.35, 0.60, 0.50, "POINT",   "Hold for dwell detection"),
    (5.0,  0.35, 0.60, 0.50, "OKAY",    "Confirm selection"),
    (6.0,  0.50, 0.50, 0.50, "NEUTRAL", "Withdraw, robot fetches"),
    (14.0, 0.50, 0.50, 0.50, "NEUTRAL", "Wait for delivery"),
    (15.0, 0.50, 0.40, 0.50, "OKAY",    "Accept delivered tool"),
    (16.0, 0.50, 0.50, 0.50, "NEUTRAL", "Complete"),
]

INSPECTION_SWEEP = [
    (0.0,  0.30, 0.50, 0.50, "POINT",   "Start left"),
    (3.0,  0.70, 0.50, 0.50, "POINT",   "Sweep right"),
    (6.0,  0.70, 0.30, 0.50, "POINT",   "Sweep up-right"),
    (9.0,  0.30, 0.70, 0.50, "POINT",   "Sweep down-left"),
    (11.0, 0.50, 0.50, 0.50, "POINT",   "Return center"),
    (12.0, 0.50, 0.50, 0.50, "STOP",    "Stop"),
    (14.0, 0.50, 0.50, 0.50, "OKAY",    "Resume"),
    (15.0, 0.60, 0.40, 0.50, "POINT",   "Continue"),
    (17.0, 0.50, 0.50, 0.50, "NEUTRAL", "Done"),
]

# tobii_node auto-ramps fatigue over fatigue_ramp_minutes (default 10 min).
# This sequence is short (~28s) so auto-tremor provides the tremor input.
# For full fatigue validation, run after tobii has been running ~5-8 min,
# or set: ros2 param set /tobii_node sim_fatigue_override 0.7
FATIGUE_SEQUENCE = [
    (0.0,  0.50, 0.50, 0.50, "POINT",   "Normal FRESH tremor~0.002"),
    (3.0,  0.40, 0.45, 0.50, "POINT",   "Moving fatigue ramping"),
    (8.0,  0.60, 0.55, 0.50, "POINT",   "MILD: slower speed tremor~0.008"),
    (12.0, 0.35, 0.45, 0.50, "POINT",   "Fatigue increasing"),
    (16.0, 0.55, 0.40, 0.50, "POINT",   "MODERATE: confirm needed tremor~0.018"),
    (18.0, 0.55, 0.40, 0.50, "OKAY",    "Operator confirms"),
    (20.0, 0.40, 0.55, 0.50, "POINT",   "Cautious operation"),
    (24.0, 0.45, 0.50, 0.50, "POINT",   "SEVERE: lockout tremor~0.035"),
    (26.0, 0.45, 0.50, 0.50, "POINT",   "Robot refuses (lockout active)"),
    (28.0, 0.50, 0.50, 0.50, "NEUTRAL", "Complete"),
]

ESTOP_SEQUENCE = [
    (0.0,  0.50, 0.50, 0.50, "POINT",   "Moving"),
    (1.0,  0.35, 0.40, 0.50, "POINT",   "Moving to target"),
    (3.0,  0.35, 0.40, 0.50, "STOP",    "EMERGENCY STOP"),
    (5.0,  0.35, 0.40, 0.50, "OKAY",    "Unfreeze"),
    (6.0,  0.60, 0.60, 0.50, "POINT",   "Resume moving"),
    (8.0,  0.60, 0.60, 0.50, "STOP",    "Second STOP"),
    (10.0, 0.60, 0.60, 0.50, "OKAY",    "Unfreeze"),
    (11.0, 0.50, 0.50, 0.50, "NEUTRAL", "Done"),
]

AXIS_X_ONLY = [
    (0.0, 0.30, 0.50, 0.50, "POINT",   "Left  (robot Y+)"),
    (3.0, 0.70, 0.50, 0.50, "POINT",   "Right (robot Y-)"),
    (6.0, 0.30, 0.50, 0.50, "POINT",   "Left again"),
    (8.0, 0.50, 0.50, 0.50, "NEUTRAL", "Done"),
]

AXIS_Y_ONLY = [
    (0.0, 0.50, 0.20, 0.50, "POINT",   "High (robot Z low)"),
    (3.0, 0.50, 0.80, 0.50, "POINT",   "Low  (robot Z high)"),
    (6.0, 0.50, 0.20, 0.50, "POINT",   "High again"),
    (8.0, 0.50, 0.50, 0.50, "NEUTRAL", "Done"),
]

AXIS_DEPTH_ONLY = [
    (0.0, 0.50, 0.50, 0.30, "POINT",   "Depth close"),
    (3.0, 0.50, 0.50, 0.70, "POINT",   "Depth far"),
    (6.0, 0.50, 0.50, 0.30, "POINT",   "Close again"),
    (8.0, 0.50, 0.50, 0.50, "NEUTRAL", "Done"),
]


def _offset_seq(seq, dt):
    """Shift all times in a sequence by dt seconds."""
    return [(t + dt, x, y, d, g, desc) for (t, x, y, d, g, desc) in seq]


def _build_full_demo():
    """Concatenate all four main sequences with 3-second pauses."""
    pause = 3.0
    seqs = [FETCH_SEQUENCE, INSPECTION_SWEEP, FATIGUE_SEQUENCE, ESTOP_SEQUENCE]
    result = []
    offset = 0.0
    for seq in seqs:
        result.extend(_offset_seq(seq, offset))
        offset += seq[-1][0] + pause
    return result


FULL_DEMO = _build_full_demo()



# LinkerHand L25: 21-joint gesture poses (radians)
# Joint order matches linkerhand_l25_right.urdf

L25_JOINT_NAMES = [
    # Thumb (5)
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
    # Index (4)
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip', 'index_dip',
    # Middle (4)
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip', 'middle_dip',
    # Ring (4)
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip', 'ring_dip',
    # Pinky (4)
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip', 'pinky_dip',
]

GESTURE_POSES = {
    # Only index finger extended; thumb curled across palm, others fully curled
    "POINT": [
        # thumb_cmc_roll, yaw, pitch, mcp, ip : thumb tucked across palm
         0.00,  0.80,  1.20,  1.20,  0.80,
        # index: roll=0, pitch=0 (straight), pip=0, dip=0
         0.00,  0.00,  0.00,  0.00,
        # middle: fully curled
         0.00,  1.50,  1.20,  1.50,
        # ring: fully curled
         0.00,  1.50,  1.20,  1.50,
        # pinky: fully curled
         0.00,  1.40,  1.10,  1.40,
    ],
    # Palm flat, all fingers extended and spread open
    "STOP": [
        # thumb: extended out to side
         0.40,  0.20,  0.10,  0.10,  0.05,
        # index: straight
         0.10,  0.00,  0.00,  0.00,
        # middle: straight
         0.00,  0.00,  0.00,  0.00,
        # ring: straight
        -0.05,  0.00,  0.00,  0.00,
        # pinky: straight
        -0.10,  0.00,  0.00,  0.00,
    ],
    # Thumbs-up: thumb extended, all other fingers curled into a fist
    "OKAY": [
        # thumb fully extended (all joints at zero = straight along metacarpal axis)
         0.00,  0.00,  0.00,  0.00,  0.00,
        # index: fully curled
         0.00,  1.50,  1.20,  1.50,
        # middle: fully curled
         0.00,  1.50,  1.20,  1.50,
        # ring: fully curled
         0.00,  1.40,  1.10,  1.40,
        # pinky: fully curled
         0.00,  1.30,  1.00,  1.30,
    ],
    # Full power grasp: all fingers tightly closed
    "GRASP": [
        # thumb: across palm
         0.10,  0.80,  1.20,  1.20,  0.80,
        # index: fully curled
         0.00,  1.50,  1.20,  1.50,
        # middle: fully curled
         0.00,  1.50,  1.20,  1.50,
        # ring: fully curled
         0.00,  1.40,  1.10,  1.40,
        # pinky: fully curled
         0.00,  1.30,  1.00,  1.30,
    ],
    # Relaxed open hand: slight natural curl
    "NEUTRAL": [
        # thumb: relaxed
         0.00,  0.15,  0.10,  0.10,  0.05,
        # index: slight curl
         0.00,  0.20,  0.15,  0.10,
        # middle
         0.00,  0.20,  0.15,  0.10,
        # ring
         0.00,  0.20,  0.15,  0.10,
        # pinky
         0.00,  0.25,  0.18,  0.12,
    ],
}



# Main Node

class VirtualOperatorNode(Node):

    def __init__(self):
        super().__init__('virtual_operator')

        # Publishers (exact camera_node topic names/types)
        self.landmark_pub  = self.create_publisher(
            Float32MultiArray, '/vision_input/hand_landmarks', 10)
        self.right_lm_pub  = self.create_publisher(
            Float32MultiArray, '/vision_input/right_hand_landmarks', 10)
        self.left_lm_pub   = self.create_publisher(
            Float32MultiArray, '/vision_input/left_hand_landmarks', 10)
        self.gesture_pub   = self.create_publisher(
            String, '/vision_input/gesture_text', 10)
        self.velocity_pub  = self.create_publisher(
            Float32MultiArray, '/vision_input/hand_velocity', 10)
        self.hand_joint_pub = self.create_publisher(
            JointState, '/virtual_hand/joint_states', 10)
        # Base world pose for the virtual hand. Consumed by:
        #   - the custom Gazebo plugin (virtual_hand_joint_driver) which
        #     calls Model::SetWorldPose every tick → moves Gazebo hand
        #   - the dynamic TF broadcast below → moves RViz hand
        self.hand_pose_pub = self.create_publisher(
            Pose, '/virtual_hand/base_pose', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.create_subscription(JointState, '/joint_states',
                                 self._js_cb, 10)
        self.create_subscription(Float32, '/fatigue/score',
                                 self._fatigue_score_cb, 10)
        self.create_subscription(String, '/fatigue/level',
                                 self._fatigue_level_cb, 10)
        # Gamepad handoff: pause keyboard publishing when LB is held
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self._gamepad_active = False
        # Cache gamepad position so we can sync on resume
        self.create_subscription(Float32MultiArray,
                                 '/vision_input/hand_landmarks',
                                 self._gamepad_lm_cb, 10)
        self._gamepad_last_pos = None  # (tip_x, tip_y, depth)

        # Thread safety
        self._lock = threading.Lock()

        # Position state
        self.tip_x = 0.50    # 0-1, normalized mouse X
        self.tip_y = 0.50    # 0-1, normalized mouse Y
        self.depth = 0.50    # 0-1, scroll wheel
        self.gesture = "NEUTRAL"
        self.sensitivity = 1.0
        self.axis_lock = None   # None | 'x' | 'y' | 'depth'

        # Clutch mode (hold Space to move hand)
        self._clutch_active = False
        self._clutch_origin_x = 0  # screen px when Space was pressed
        self._clutch_origin_y = 0
        self._clutch_tip_x0 = 0.5  # tip values when Space was pressed
        self._clutch_tip_y0 = 0.5
        self._last_mouse_x = 0    # last known screen px (for clutch origin)
        self._last_mouse_y = 0

        # Tremor model
        self.tremor_amp    = 0.0
        self.tremor_mode   = "off"   # "off" | "manual" | "auto"
        self.tremor_freq_x = 6.0    # Hz — fatigue-band
        self.tremor_freq_y = 6.5    # slightly different → no circular patterns
        self.tremor_phase_x = random.uniform(0, 2 * math.pi)
        self.tremor_phase_y = random.uniform(0, 2 * math.pi)
        self.drift_x = 0.0
        self.drift_y = 0.0

        # Script state
        self.script_active     = False
        self.script_seq        = []
        self.script_name       = ""
        self.script_start_time = 0.0

        # Fatigue (from subscribers)
        self.fatigue_score = 0.0
        self.fatigue_level = "FRESH"

        # Robot joint state (for logging)
        self.joint_pos   = [0.0] * 6
        self.js_received = False

        # Velocity computation
        self.prev_tip_x = 0.50
        self.prev_tip_y = 0.50
        self.prev_time  = time.time()
        self.vel_x      = 0.0
        self.vel_y      = 0.0

        # Logging
        self.logging_active    = False
        self._log_file         = None
        self._log_writer       = None
        self._log_start_time   = None

        # Misc
        self.verbose = False
        self._display_initialized = False

        # Screen size for normalizing mouse coords
        self.screen_w, self.screen_h = self._get_screen_size()

        # Virtual hand pose state
        # The hand floats around a workstation anchor point. tip_x/tip_y/depth
        # map to (y, z, x) offsets in the world frame. The same Pose drives
        # both the Gazebo plugin (SetWorldPose) and the RViz TF broadcast.
        self._hand_anchor = (0.80, 0.00, 0.60)   # x, y, z in world

        # Hand orientation as roll/pitch/yaw (radians).
        # Initial "Safe Boot" pose: palm facing down, fingers
        # forward toward the robot, like a hand resting on a mouse.
        # Adjustable at runtime via arrow keys + Page Up/Down.
        self._HAND_INITIAL_RPY = (0.0, math.pi / 2.0, math.pi)
        self._hand_roll  = self._HAND_INITIAL_RPY[0]
        self._hand_pitch = self._HAND_INITIAL_RPY[1]
        self._hand_yaw   = self._HAND_INITIAL_RPY[2]
        self._ROT_STEP   = math.radians(5.0)   # 5° per keypress
        self._TRANS_STEP = 0.02                # WASD translation step (normalised)

        # Data directory
        os.makedirs(os.path.expanduser('~/ros2_ws/data'), exist_ok=True)

        # Timers
        self.create_timer(1.0 / 30.0, self._main_loop)   # 30 Hz publish
        self.create_timer(0.10,        self._update_display)  # 10 Hz display

        # Input listeners
        self._start_input()

        self.get_logger().info("=" * 55)
        self.get_logger().info("  VIRTUAL OPERATOR NODE")
        self.get_logger().info(f"  pynput: {'OK' if PYNPUT_OK else 'MISSING (pip install pynput)'}")
        self.get_logger().info(f"  Screen: {self.screen_w}x{self.screen_h}")
        self.get_logger().info("  Press 'p' to start pointing. 'q' to quit.")
        self.get_logger().info("=" * 55)

    # Screen detection
    def _get_screen_size(self):
        try:
            import subprocess
            out = subprocess.check_output(
                ['xdpyinfo'], stderr=subprocess.DEVNULL).decode()
            for line in out.split('\n'):
                if 'dimensions:' in line:
                    dims = line.strip().split()[1]   # "1920x1080"
                    w, h = dims.split('x')
                    return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080

    # Subscribers
    def _js_cb(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        for i, n in enumerate(names):
            if n in d:
                self.joint_pos[i] = d[n]
        self.js_received = True

    def _fatigue_score_cb(self, msg: Float32):
        self.fatigue_score = msg.data

    def _fatigue_level_cb(self, msg: String):
        new = msg.data
        self.fatigue_level = new
        if self.tremor_mode == "auto":
            self.tremor_amp = TREMOR_BY_LEVEL.get(new, 0.002)

    def _joy_cb(self, msg: Joy):
        """Detect gamepad deadman switch (LB = button 4).
        When held, keyboard yields control to gamepad."""
        if len(msg.buttons) > 4:
            self._gamepad_active = bool(msg.buttons[4])

    def _gamepad_lm_cb(self, msg: Float32MultiArray):
        """Cache gamepad landmark position while gamepad is active."""
        if self._gamepad_active and len(msg.data) >= 27:
            self._gamepad_last_pos = (msg.data[24], msg.data[25], msg.data[26])

    # Input setup
    def _start_input(self):
        if not PYNPUT_OK:
            self.get_logger().warn(
                "pynput not available — keyboard/mouse control disabled. "
                "Install with: pip install pynput")
            return

        self._mouse_listener = _pynput_mouse.Listener(
            on_move=self._on_mouse_move,
            on_scroll=self._on_scroll)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self._kb_listener = _pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    # Mouse callbacks
    def _on_mouse_move(self, x, y):
        with self._lock:
            if self.script_active:
                return
            # Store raw screen position for clutch origin capture
            self._last_mouse_x = x
            self._last_mouse_y = y
            if not self._clutch_active:
                return  # mouse free for RViz when clutch not held
            # Delta from clutch origin -> normalized movement
            dx_px = x - self._clutch_origin_x
            dy_px = y - self._clutch_origin_y
            sens = self.sensitivity
            nx = self._clutch_tip_x0 + (dx_px / self.screen_w) * sens
            ny = self._clutch_tip_y0 + (dy_px / self.screen_h) * sens
            # No clamping: ghost hand can move freely anywhere
            if self.axis_lock == 'x':
                self.tip_x = nx
            elif self.axis_lock == 'y':
                self.tip_y = ny
            elif self.axis_lock is None:
                self.tip_x = nx
                self.tip_y = ny

    def _on_scroll(self, x, y, dx, dy):
        with self._lock:
            if self.script_active:
                return
            if not self._clutch_active:
                return  # scroll free for RViz zoom when clutch not held
            if self.axis_lock not in ('x', 'y'):
                self.depth = self.depth - dy * 0.05

    # Keyboard callback
    def _on_key_press(self, key):
        # Clutch: hold Space to control hand, release to free mouse
        if key == _pynput_keyboard.Key.space:
            with self._lock:
                if not self._clutch_active:
                    self._clutch_active = True
                    self._clutch_origin_x = getattr(self, '_last_mouse_x', 0)
                    self._clutch_origin_y = getattr(self, '_last_mouse_y', 0)
                    self._clutch_tip_x0 = self.tip_x
                    self._clutch_tip_y0 = self.tip_y
            return

        # LShift = STOP gesture (E-STOP). Single tap, instantaneous.
        # Replaces 's' so WASD can be used for translation.
        if key in (_pynput_keyboard.Key.shift, _pynput_keyboard.Key.shift_l):
            with self._lock:
                self.gesture = "STOP"
                self.script_active = False
            return

        # Extract character
        c = None
        try:
            if hasattr(key, 'char') and key.char:
                c = key.char.lower()
        except Exception:
            pass

        with self._lock:
            self._handle_char_key(c)

        # Special keys (function keys, arrows) — outside char block
        try:
            name = key.name if hasattr(key, 'name') else None
        except Exception:
            name = None

        if name:
            with self._lock:
                self._handle_special_key(name)

    def _on_key_release(self, key):
        if key == _pynput_keyboard.Key.space:
            with self._lock:
                self._clutch_active = False

    def _handle_char_key(self, c):
        if c is None:
            return
        # WASD/QE translation (game-style)
        # W/S = depth (forward/back into robot space)
        # A/D = horizontal (left/right)
        # Q/E = vertical (down/up: note tip_y is screen-Y inverted)
        step = self._TRANS_STEP
        if c == 'w':
            self.depth -= step   # forward (toward robot)
            return
        elif c == 's':
            self.depth += step   # backward (away from robot)
            return
        elif c == 'a':
            self.tip_x += step   # left
            return
        elif c == 'd':
            self.tip_x -= step   # right
            return
        elif c == 'e':
            self.tip_y -= step   # up
            return
        elif c == 'c':
            self.tip_y += step   # down
            return

        # Gestures (note: STOP moved to LShift, 's' is now backward)
        if c == 'p':
            self.gesture = "POINT"
            self.script_active = False
        elif c == 'o':
            self.gesture = "OKAY"
            self.script_active = False
        elif c == 'g':
            self.gesture = "GRASP"
            self.script_active = False
        elif c == 'n':
            self.gesture = "NEUTRAL"
            self.script_active = False
        # 1/2/3 reserved for fatigue_override (numpad)
        # Numpad 5/6 are WoZ triggers handled by fatigue_override_node,
        # which runs a global pynput hook and works in every launch.
        elif c == '0':
            self.gesture = "RESET_WORLD"
            self.script_active = False
        elif c == 'h':
            self.tip_x, self.tip_y, self.depth = 0.50, 0.50, 0.50
            self.gesture = "NEUTRAL"
        # Tremor
        elif c == '[':
            self.tremor_amp = max(0.0, self.tremor_amp - 0.005)
            self.tremor_mode = "manual"
        elif c == ']':
            self.tremor_amp = min(0.05, self.tremor_amp + 0.005)
            self.tremor_mode = "manual"
        elif c == '\\':
            if self.tremor_mode == "auto":
                self.tremor_mode = "off"
                self.tremor_amp = 0.0
            else:
                self.tremor_mode = "auto"
                self.tremor_amp = TREMOR_BY_LEVEL.get(self.fatigue_level, 0.002)
        # Utility
        elif c == 'r':
            self.tip_x, self.tip_y, self.depth = 0.50, 0.50, 0.50
            self.gesture = "NEUTRAL"
            self.tremor_amp = 0.0
            self.tremor_mode = "off"
            self.drift_x = 0.0
            self.drift_y = 0.0
            self.script_active = False
            # Reset hand orientation to Safe Boot pose
            self._hand_roll, self._hand_pitch, self._hand_yaw = self._HAND_INITIAL_RPY
        elif c == 'l':
            if self.logging_active:
                self._stop_logging()
            else:
                self._start_logging()
        elif c == 'v':
            self.verbose = not self.verbose
        # Roll rotation: < (comma) / > (period)
        elif c == ',':
            self._hand_roll -= self._ROT_STEP   # roll left
        elif c == '.':
            self._hand_roll += self._ROT_STEP   # roll right
        elif c in ('+', '='):
            self.sensitivity = min(3.0, self.sensitivity + 0.1)
        elif c == '-':
            self.sensitivity = max(0.2, self.sensitivity - 0.1)
        elif c == 'q':
            self._cleanup()
            rclpy.shutdown()

    def _handle_special_key(self, name):
        # Function keys → scripts
        script_map = {
            'f1': (FETCH_SEQUENCE,    "FETCH",      False),
            'f2': (INSPECTION_SWEEP,  "INSPECTION", False),
            'f3': (FATIGUE_SEQUENCE,  "FATIGUE",    True),   # auto-tremor ON
            'f4': (ESTOP_SEQUENCE,    "ESTOP",      False),
            'f5': (FULL_DEMO,         "FULL_DEMO",  False),
            'f6': (AXIS_X_ONLY,       "AXIS_X",     False),
            'f7': (AXIS_Y_ONLY,       "AXIS_Y",     False),
            'f8': (AXIS_DEPTH_ONLY,   "AXIS_DEPTH", False),
        }
        if name in script_map:
            seq, sname, auto_tremor = script_map[name]
            self._start_script(seq, sname)
            if auto_tremor:
                self.tremor_mode = "auto"
                self.tremor_amp = TREMOR_BY_LEVEL.get(self.fatigue_level, 0.002)
            return

        # Arrow keys: hand rotation (5° per press)
        # Left/Right = yaw, Up/Down = pitch, < / > = roll
        rs = self._ROT_STEP
        if name == 'left':
            self._hand_yaw -= rs
        elif name == 'right':
            self._hand_yaw += rs
        elif name == 'up':
            self._hand_pitch -= rs
        elif name == 'down':
            self._hand_pitch += rs
        elif name == 'esc':
            self.axis_lock = None

    # Script playback
    def _start_script(self, seq, name):
        self.script_seq        = seq
        self.script_name       = name
        self.script_start_time = time.time()
        self.script_active     = True
        self.get_logger().info(f"Script started: {name} ({seq[-1][0]:.0f}s)")

    def _advance_script(self):
        """Interpolate between waypoints. Call inside _main_loop (under lock)."""
        if not self.script_active or not self.script_seq:
            return
        elapsed = time.time() - self.script_start_time
        seq = self.script_seq

        if elapsed >= seq[-1][0]:
            # Sequence complete: snap to final waypoint
            _, tx, ty, d, g, _ = seq[-1]
            self.tip_x, self.tip_y, self.depth, self.gesture = tx, ty, d, g
            self.script_active = False
            self.get_logger().info(f"Script '{self.script_name}' complete.")
            return

        # Find surrounding waypoints
        wp_from = seq[0]
        wp_to   = seq[-1]
        for i in range(len(seq) - 1):
            if seq[i][0] <= elapsed < seq[i + 1][0]:
                wp_from = seq[i]
                wp_to   = seq[i + 1]
                break

        seg_dur = wp_to[0] - wp_from[0]
        if seg_dur > 0.001:
            t = (elapsed - wp_from[0]) / seg_dur
            t = max(0.0, min(1.0, t))
            t_s = 3 * t * t - 2 * t * t * t   # smoothstep
        else:
            t_s = 1.0

        self.tip_x   = wp_from[1] + (wp_to[1] - wp_from[1]) * t_s
        self.tip_y   = wp_from[2] + (wp_to[2] - wp_from[2]) * t_s
        self.depth   = wp_from[3] + (wp_to[3] - wp_from[3]) * t_s
        self.gesture = wp_from[4]   # use from-waypoint gesture

    # Tremor model
    def _apply_tremor(self, tx, ty):
        """Apply fatigue-realistic sinusoidal tremor + random walk drift."""
        if self.tremor_amp < 0.0001:
            return tx, ty
        now = time.time()
        # Primary sinusoidal component (6 Hz fatigue band)
        tx_noise = self.tremor_amp * math.sin(
            2 * math.pi * self.tremor_freq_x * now + self.tremor_phase_x)
        ty_noise = self.tremor_amp * math.sin(
            2 * math.pi * self.tremor_freq_y * now + self.tremor_phase_y)
        # Random walk drift
        drift_rate = 0.0003 + self.tremor_amp * 0.08
        max_drift  = 0.03
        self.drift_x += random.gauss(0, drift_rate)
        self.drift_y += random.gauss(0, drift_rate)
        self.drift_x = max(-max_drift, min(max_drift, self.drift_x))
        self.drift_y = max(-max_drift, min(max_drift, self.drift_y))
        # Combined
        nx = tx + tx_noise + self.drift_x
        ny = ty + ty_noise + self.drift_y
        return nx, ny

    # Landmark array builder
    def _build_landmark_array(self, tx, ty, d):
        """
        Build 63-float hand landmark array.
        data[24]=tx (index tip X), data[25]=ty (index tip Y), data[26]=d (depth)
        All other landmarks form a plausible POINT-gesture skeleton.
        """
        wx = tx - 0.05
        wy = min(1.0, ty + 0.16)
        # 21 landmarks (x, y, z) — MediaPipe normalized space
        lms = [
            (wx,         wy,        0.00),   # 0 WRIST
            (wx - 0.04,  wy - 0.02, 0.00),   # 1 THUMB_CMC
            (wx - 0.06,  wy - 0.06, 0.00),   # 2 THUMB_MCP
            (wx - 0.08,  wy - 0.09, 0.00),   # 3 THUMB_IP
            (wx - 0.09,  wy - 0.12, 0.00),   # 4 THUMB_TIP
            (wx + 0.02,  wy - 0.05, 0.00),   # 5 INDEX_MCP
            (wx + 0.02,  wy - 0.09, 0.00),   # 6 INDEX_PIP
            (wx + 0.01,  wy - 0.13, 0.00),   # 7 INDEX_DIP
            (tx,         ty,        d),       # 8 INDEX_TIP ← KEY: data[24,25,26]
            (wx + 0.04,  wy - 0.05, 0.00),   # 9 MIDDLE_MCP
            (wx + 0.05,  wy - 0.07, 0.00),   # 10 MIDDLE_PIP
            (wx + 0.05,  wy - 0.06, 0.00),   # 11 MIDDLE_DIP
            (wx + 0.05,  wy - 0.05, 0.00),   # 12 MIDDLE_TIP
            (wx + 0.05,  wy - 0.05, 0.00),   # 13 RING_MCP
            (wx + 0.06,  wy - 0.06, 0.00),   # 14 RING_PIP
            (wx + 0.06,  wy - 0.05, 0.00),   # 15 RING_DIP
            (wx + 0.06,  wy - 0.04, 0.00),   # 16 RING_TIP
            (wx + 0.06,  wy - 0.04, 0.00),   # 17 PINKY_MCP
            (wx + 0.07,  wy - 0.05, 0.00),   # 18 PINKY_PIP
            (wx + 0.07,  wy - 0.04, 0.00),   # 19 PINKY_DIP
            (wx + 0.07,  wy - 0.03, 0.00),   # 20 PINKY_TIP
        ]
        flat = []
        for (x, y, z) in lms:
            flat.extend([float(max(0.0, min(1.0, x))),
                         float(max(0.0, min(1.0, y))),
                         float(z)])
        return flat

    # 30 Hz main loop
    def _main_loop(self):
        # Yield to gamepad when deadman switch (LB) is held
        if self._gamepad_active:
            self._was_gamepad = True
            return

        # Coming back from gamepad -> reset gesture so it doesn't
        # jump back to whatever was active before LB was pressed
        if getattr(self, '_was_gamepad', False):
            self._was_gamepad = False
            with self._lock:
                self.gesture = "NEUTRAL"
                # Sync position to where the gamepad left the hand
                if self._gamepad_last_pos is not None:
                    self.tip_x, self.tip_y, self.depth = self._gamepad_last_pos
                    self._gamepad_last_pos = None

        now = time.time()
        with self._lock:
            if self.script_active:
                self._advance_script()

            base_tx  = self.tip_x
            base_ty  = self.tip_y
            pub_d    = self.depth
            pub_g    = self.gesture

        # Apply tremor (outside lock — only modifies drift_x/y but that's ok)
        pub_tx, pub_ty = self._apply_tremor(base_tx, base_ty)

        # Velocity
        dt = now - self.prev_time
        if dt > 0.001:
            self.vel_x = (pub_tx - self.prev_tip_x) / dt
            self.vel_y = (pub_ty - self.prev_tip_y) / dt
        speed = math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        pred_x = pub_tx + self.vel_x * 0.5
        pred_y = pub_ty + self.vel_y * 0.5
        self.prev_tip_x = pub_tx
        self.prev_tip_y = pub_ty
        self.prev_time  = now

        # Publish landmarks
        lm_data = self._build_landmark_array(pub_tx, pub_ty, pub_d)
        lm_msg = Float32MultiArray()
        lm_msg.data = lm_data
        self.landmark_pub.publish(lm_msg)
        self.right_lm_pub.publish(lm_msg)
        # Left hand = zeros (virtual operator is right-hand only)
        left_msg = Float32MultiArray()
        left_msg.data = [0.0] * 63
        self.left_lm_pub.publish(left_msg)

        # Publish gesture
        g_msg = String()
        g_msg.data = pub_g
        self.gesture_pub.publish(g_msg)

        # Publish velocity (7 floats — exact camera_node format)
        v_msg = Float32MultiArray()
        v_msg.data = [
            float(self.vel_x), float(self.vel_y), 0.0,
            float(speed),
            float(pred_x), float(pred_y), float(pub_d),
        ]
        self.velocity_pub.publish(v_msg)

        # Virtual hand URDF joint states
        self._publish_hand_joint_states(pub_g)

        # Gazebo hand position
        self._update_gazebo_hand(pub_tx, pub_ty, pub_d)

        # Logging
        if self.logging_active:
            self._log_row(pub_tx, pub_ty, pub_d, pub_g, speed)

        if self.verbose:
            self.get_logger().info(
                f"tx={pub_tx:.3f} ty={pub_ty:.3f} d={pub_d:.3f} "
                f"g={pub_g} spd={speed:.3f} trem={self.tremor_amp:.4f}")

    # Virtual hand URDF joint states
    def _publish_hand_joint_states(self, gesture: str):
        """Publish JointState so robot_state_publisher animates the L25 mesh
        in RViz. Gazebo's hand visual remains in NEUTRAL — finger gestures
        are an RViz-only visualization in this design."""
        poses = GESTURE_POSES.get(gesture, GESTURE_POSES["NEUTRAL"])
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = L25_JOINT_NAMES
        js.position = [float(v) for v in poses]
        self.hand_joint_pub.publish(js)

    # Virtual hand world pose
    # Publishes the hand's desired world pose. Two consumers:
    #   1. Custom Gazebo plugin (virtual_hand_joint_driver) -> SetWorldPose
    #      every physics tick. This both moves the hand AND prevents the
    #      slow rotation that joint reaction torques would otherwise cause.
    #   2. TransformBroadcaster below -> world->virtual_hand_base_link, so
    #      the RViz RobotModel display follows the same pose.
    @staticmethod
    def _rpy_to_quat(roll: float, pitch: float, yaw: float):
        """Standard RPY → quaternion conversion (ZYX intrinsic)."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def _update_gazebo_hand(self, tx, ty, d):
        ax, ay, az = self._hand_anchor
        # No limits: ghost hand can move freely anywhere in workspace
        px = ax + (d  - 0.5) * 0.80   # ~0.80m depth range
        py = ay - (tx - 0.5) * 1.00   # ~1.00m lateral range
        pz = az + (0.5 - ty) * 0.60   # ~0.60m vertical range

        # Orientation from current roll/pitch/yaw (operator-adjustable
        # at runtime via arrow keys + < / >). Default Safe Boot pose
        # is palm-down, fingers forward toward robot.
        qx, qy, qz, qw = self._rpy_to_quat(
            self._hand_roll, self._hand_pitch, self._hand_yaw)

        pose = Pose()
        pose.position.x = px
        pose.position.y = py
        pose.position.z = pz
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.hand_pose_pub.publish(pose)

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id  = 'virtual_hand_base_link'
        tf.transform.translation.x = px
        tf.transform.translation.y = py
        tf.transform.translation.z = pz
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)

    # CSV Logging
    def _start_logging(self):
        from datetime import datetime
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f'~/ros2_ws/data/virtual_operator_{ts}.csv')
        self._log_file   = open(path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'timestamp_s', 'tip_x', 'tip_y', 'depth', 'gesture',
            'hand_vx', 'hand_vy', 'speed',
            'tremor_amplitude', 'tremor_mode',
            'robot_j1', 'robot_j2', 'robot_j3', 'robot_j4', 'robot_j5', 'robot_j6',
            'fatigue_level', 'fatigue_score',
            'sequence_name',
        ])
        self._log_start_time = time.time()
        self.logging_active  = True
        self.get_logger().info(f"Logging → {path}")

    def _stop_logging(self):
        self.logging_active = False
        if self._log_file:
            self._log_file.close()
            self._log_file   = None
            self._log_writer = None
        self.get_logger().info("Logging stopped.")

    def _log_row(self, tx, ty, d, gesture, speed):
        if not self._log_writer:
            return
        with self._lock:
            seq_name = self.script_name if self.script_active else ""
        self._log_writer.writerow([
            round(time.time() - self._log_start_time, 4),
            round(tx, 4), round(ty, 4), round(d, 4), gesture,
            round(self.vel_x, 4), round(self.vel_y, 4), round(speed, 4),
            round(self.tremor_amp, 4), self.tremor_mode,
            *[round(j, 4) for j in self.joint_pos],
            self.fatigue_level, round(self.fatigue_score, 4),
            seq_name,
        ])

    # Terminal display
    def _update_display(self):
        with self._lock:
            tx      = self.tip_x
            ty      = self.tip_y
            d       = self.depth
            gesture = self.gesture
            tamp    = self.tremor_amp
            tmode   = self.tremor_mode
            fl      = self.fatigue_level
            fs      = self.fatigue_score
            sname   = self.script_name if self.script_active else "—"
            log_s   = "ON " if self.logging_active else "OFF"
            ax_s    = f"LOCK({self.axis_lock})" if self.axis_lock else "Free    "
            clutch  = self._clutch_active
            vx      = self.vel_x
            vy      = self.vel_y

        speed_s = math.sqrt(vx ** 2 + vy ** 2)

        lines = [
            "=======================================================",
            "         VIRTUAL OPERATOR — INTENTION SYSTEM         ",
            "=======================================================",
            f"  Gesture: {gesture:<10}  Pos: {tx:.3f},{ty:.3f},{d:.3f}         ",
            f"  Velocity: {speed_s:.3f} u/s  Axis: {ax_s:<10} {'[HAND]' if clutch else '[FREE]'} ",
            f"  Tremor:  {tmode:<6} amp={tamp:.4f}   Fatigue: {fl:<8}   ",
            f"  FatigueScore: {fs:.3f}  Script: {sname:<18}  ",
            f"  Logging: {log_s}                                       ",
            "=======================================================",
            "  [P]oint [LShift]STOP [O]kay [G]rasp [N]eutral      ",
            "  [WASD] Move XY  [E/C] Up/Down  [R]eset             ",
            "  [Arrows] Yaw/Pitch  [< / >] Roll                    ",
            "  [1]Red [2]Blue [3]Green [H]ome [0]Reset [Q]uit     ",
            "  [F1-F8] Scripts  [L]og  [V]erbose  [/] Tremor      ",
            "  [SPACE] Hold=move hand  Release=free mouse          ",
            "========================================================",
        ]
        n = len(lines)
        if not self._display_initialized:
            self._display_initialized = True
            sys.stdout.write('\n' * n)
        sys.stdout.write(f'\033[{n}A')
        for line in lines:
            sys.stdout.write(f'\r{line}\033[K\n')
        sys.stdout.flush()

    # Cleanup
    def _cleanup(self):
        if self.logging_active:
            self._stop_logging()



def main(args=None):
    rclpy.init(args=args)
    node = VirtualOperatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

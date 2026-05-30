#!/usr/bin/env python3
"""
fatigue_override_node.py — Manual Fatigue Level Override
Human Intention Prediction System

Listens for numpad keys via pynput and publishes to /fatigue/override.
fatigue_monitor honours this override when present, otherwise falls
back to its automatic PERCLOS/jerk-based scoring.

Key bindings:
  Numpad 1 -> FRESH      (force)
  Numpad 2 -> MILD       (force)
  Numpad 3 -> MODERATE   (force)
  Numpad 4 -> SEVERE     (force)
  Numpad 5 -> SOFT E-STOP (Feature 3 WoZ, 3 s latch)
  Numpad 6 -> FORCE SACCADE (Feature 4 WoZ, 2 s hold)
  Numpad 0 -> clear override (resume automatic scoring)
  Top-row 1-6 -> same (laptop fallback)
  Top-row 0 -> RESET WORLD (respawn all fetch cubes)

Published topics:
  /fatigue/override           std_msgs/String
    "FRESH" | "MILD" | "MODERATE" | "SEVERE" | ""   (empty = no override)
  /soft_estop_manual          std_msgs/Bool  (one-shot True on Numpad 5)
  /gaze/force_saccade         std_msgs/Bool  (one-shot True on Numpad 6)
  /vision_input/gesture_text  std_msgs/String
    "RESET_WORLD" when top-row 0 is pressed

Uses pynput global keyboard hook, so keys are captured regardless of
which window has focus. Safe to run alongside camera_node, virtual
operator, or gamepad operator.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

try:
    from pynput import keyboard
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False


KEY_TO_LEVEL = {
    keyboard.KeyCode.from_vk(65457): "FRESH",    # numpad 1
    keyboard.KeyCode.from_vk(65458): "MILD",     # numpad 2
    keyboard.KeyCode.from_vk(65459): "MODERATE", # numpad 3
    keyboard.KeyCode.from_vk(65460): "SEVERE",   # numpad 4
    keyboard.KeyCode.from_vk(65456): "",         # numpad 0 -> clear
} if PYNPUT_OK else {}

# WoZ triggers for Features 3 & 4. Numpad 5/6 are free after 1-4 are
# claimed by fatigue override. We match BOTH NumLock states because
# pynput emits a different keysym when NumLock is OFF (e.g. KP_Begin
# for Numpad 5 on most Linux keyboards):
#   NumLock ON : KP_5=65461, KP_6=65462
#   NumLock OFF: KP_Begin=65437 (->5), KP_Right=65432 (->6)
SOFT_ESTOP_VKS = {65461, 65437}
FORCE_SACCADE_VKS = {65462, 65432}

# Fallback: top-row number keys (for laptops without numpad)
# Note: top-row '0' is reserved for world reset (publishes RESET_WORLD
# to /vision_input/gesture_text). Numpad 0 remains the fatigue clear.
TOPROW_TO_LEVEL = {
    '1': "FRESH",
    '2': "MILD",
    '3': "MODERATE",
    '4': "SEVERE",
}


class FatigueOverrideNode(Node):
    def __init__(self):
        super().__init__('fatigue_override')

        self.override_pub = self.create_publisher(
            String, '/fatigue/override', 10)

        # Top-row '0' -> publish RESET_WORLD on the gesture topic so
        # robot_controller teleports all cubes back to their starting
        # positions. Works in every scenario because this node is
        # always launched (see gazebo_sim.launch.py).
        self.gesture_pub = self.create_publisher(
            String, '/vision_input/gesture_text', 10)

        # WoZ triggers: one-shot Bool(True) publishes. tobii_node handles
        # the latch/hold semantics, so this node stays stateless.
        self.soft_estop_pub = self.create_publisher(
            Bool, '/soft_estop_manual', 10)
        self.force_saccade_pub = self.create_publisher(
            Bool, '/gaze/force_saccade', 10)

        self.current_override = ""   # empty = no override

        # Republish at 2 Hz so fatigue_monitor always knows current state
        self.create_timer(0.5, self._republish)

        if not PYNPUT_OK:
            self.get_logger().error(
                "pynput not installed — override node is a no-op. "
                "Install with: pip install pynput")
            return

        # Start global keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_key)
        self.listener.daemon = True
        self.listener.start()

        self.get_logger().info("=" * 55)
        self.get_logger().info("  FATIGUE OVERRIDE NODE")
        self.get_logger().info("  Numpad 1 → FRESH")
        self.get_logger().info("  Numpad 2 → MILD")
        self.get_logger().info("  Numpad 3 → MODERATE")
        self.get_logger().info("  Numpad 4 → SEVERE")
        self.get_logger().info("  Numpad 0 → clear (auto)")
        self.get_logger().info("  Numpad 5 → SOFT E-STOP (3 s latch)")
        self.get_logger().info("  Numpad 6 → FORCE SACCADE (2 s hold)")
        self.get_logger().info("  (top-row 1-6 also work)")
        self.get_logger().info("  Top-row 0 → RESET WORLD (respawn cubes)")
        self.get_logger().info("=" * 55)

    def _set_override(self, level: str):
        if level == self.current_override:
            return
        self.current_override = level
        msg = String()
        msg.data = level
        self.override_pub.publish(msg)
        if level == "":
            self.get_logger().warn("FATIGUE OVERRIDE CLEARED — automatic scoring resumed")
        else:
            self.get_logger().warn(f"FATIGUE OVERRIDE → {level}")

    def _trigger_soft_estop(self):
        self.soft_estop_pub.publish(Bool(data=True))
        self.get_logger().warn("🛑 SOFT E-STOP triggered (3 s latch)")

    def _trigger_force_saccade(self):
        self.force_saccade_pub.publish(Bool(data=True))
        self.get_logger().warn("👁  FORCE SACCADE triggered (2 s hold)")

    def _on_key(self, key):
        # Numpad keycodes (fatigue levels)
        if key in KEY_TO_LEVEL:
            self._set_override(KEY_TO_LEVEL[key])
            return

        # WoZ triggers: match by raw vk code so either NumLock state works.
        vk = getattr(key, 'vk', None)
        if vk in SOFT_ESTOP_VKS:
            self._trigger_soft_estop()
            return
        if vk in FORCE_SACCADE_VKS:
            self._trigger_force_saccade()
            return

        # Top-row number keys (fallback for laptops)
        try:
            ch = key.char
        except AttributeError:
            return
        if ch in TOPROW_TO_LEVEL:
            self._set_override(TOPROW_TO_LEVEL[ch])
        elif ch == '5':
            self._trigger_soft_estop()
        elif ch == '6':
            self._trigger_force_saccade()
        elif ch == '0':
            # World reset: publish RESET_WORLD gesture. Robot controller listens for this and teleports all cubes back to their starting positions. Works in every scenario because this node is always launched (see gazebo_sim.launch.py and real_ur3.launch.py).
            msg = String()
            msg.data = "RESET_WORLD"
            self.gesture_pub.publish(msg)
            self.get_logger().warn("WORLD RESET → respawning cubes")

    def _republish(self):
        msg = String()
        msg.data = self.current_override
        self.override_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FatigueOverrideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

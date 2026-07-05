#!/usr/bin/env python3
"""
fatigue_monitor.py — Composite Fatigue Scoring Engine
Human Intention Prediction System

Composite score (from thesis):
  F = 0.30 × blink_rate_norm
    + 0.25 × blink_duration_norm
    + 0.30 × PERCLOS
    + 0.15 × hand_jerk_norm

Fatigue levels:
  FRESH     0.00 — 0.30   Full speed, full autonomy
  MILD      0.30 — 0.60   Reduced speed
  MODERATE  0.60 — 0.80   Requires OKAY confirmation for actions
  SEVERE    0.80 — 1.00   Safety lockout, no movement commands accepted

Subscribed Topics:
  /eye_tracking/raw            Float32MultiArray from tobii_node
  /vision_input/hand_velocity  Float32MultiArray from camera_node

Published Topics:
  /fatigue/score       std_msgs/Float32         Composite score 0.0-1.0
  /fatigue/level       std_msgs/String          FRESH / MILD / MODERATE / SEVERE
  /fatigue/components  Float32MultiArray        [blink_norm, dur_norm, perclos,
                                                  jerk_norm, composite, raw_blink_rate,
                                                  raw_blink_dur, raw_perclos, raw_jerk]

Parameters:
  w_blink_rate      float  (default: 0.30)
  w_blink_duration  float  (default: 0.25)
  w_perclos         float  (default: 0.30)
  w_hand_jerk       float  (default: 0.15)
  smoothing_alpha   float  (default: 0.3)    EMA smoothing on composite score
  publish_rate      float  (default: 5.0)    Hz
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray
import time
import math


class FatigueMonitor(Node):
    def __init__(self):
        super().__init__('fatigue_monitor')

        
        # Parameters: weights from thesis
        
        self.declare_parameter('w_blink_rate', 0.30)
        self.declare_parameter('w_blink_duration', 0.25)
        self.declare_parameter('w_perclos', 0.30)
        self.declare_parameter('w_hand_jerk', 0.15)
        self.declare_parameter('smoothing_alpha', 0.3)
        self.declare_parameter('publish_rate', 5.0)

        # Normalisation thresholds (from fatigue literature)
        # Blink rate: normalised between 12/min (rested) and 28/min (fatigued)
        self.declare_parameter('blink_rate_baseline', 12.0)
        self.declare_parameter('blink_rate_max', 28.0)
        # Blink duration: normalised between 0.12s (rested) and 0.38s (fatigued)
        self.declare_parameter('blink_dur_baseline', 0.12)
        self.declare_parameter('blink_dur_max', 0.38)
        # PERCLOS: already 0-1; deployed threshold 0.08 (more sensitive than
        # the 0.15 drowsiness criterion of Dinges 1998)
        self.declare_parameter('perclos_threshold', 0.08)
        # Hand jerk: normalised from velocity data
        self.declare_parameter('jerk_baseline', 0.3)
        self.declare_parameter('jerk_max', 3.0)

        publish_rate = self.get_parameter('publish_rate').value

        
        # Subscribers
        
        self.eye_sub = self.create_subscription(
            Float32MultiArray, '/eye_tracking/raw', self.eye_callback, 10)
        self.velocity_sub = self.create_subscription(
            Float32MultiArray, '/vision_input/hand_velocity', self.velocity_callback, 10)
        # Manual override from fatigue_override_node (numpad keys)
        self.override_sub = self.create_subscription(
            String, '/fatigue/override', self.override_callback, 10)

        
        # Publishers
        
        self.score_pub = self.create_publisher(Float32, '/fatigue/score', 10)
        self.level_pub = self.create_publisher(String, '/fatigue/level', 10)
        self.components_pub = self.create_publisher(
            Float32MultiArray, '/fatigue/components', 10)

        
        # State
        
        self.raw_blink_rate = 0.0
        self.raw_blink_duration = 0.15
        self.raw_perclos = 0.0
        self.raw_pupil_l = 4.0
        self.raw_pupil_r = 4.0
        self.raw_eye_openness = 1.0

        self.raw_hand_jerk = 0.0
        self.prev_hand_speed = 0.0
        self.prev_velocity_time = time.time()
        self.hand_jerk_history = []

        self.composite_score = 0.0     # smoothed
        self.fatigue_level = "FRESH"
        self.prev_level = "FRESH"

        # Manual override state (empty string = no override, fall back to auto)
        self.manual_override = ""
        # Nominal scores for each override level (middle of each band)
        self._override_scores = {
            "FRESH":    0.15,
            "MILD":     0.45,
            "MODERATE": 0.70,
            "SEVERE":   0.90,
        }

        self.eye_data_received = False
        self.velocity_data_received = False

        
        # Timers
        
        self.create_timer(1.0 / publish_rate, self.compute_and_publish)
        self.create_timer(3.0, self.log_state)

        
        # Startup
        
        self.get_logger().info("=" * 55)
        self.get_logger().info("  FATIGUE MONITOR")
        self.get_logger().info(f"  Weights: blink_rate={self.get_parameter('w_blink_rate').value}, "
                               f"blink_dur={self.get_parameter('w_blink_duration').value}, "
                               f"perclos={self.get_parameter('w_perclos').value}, "
                               f"jerk={self.get_parameter('w_hand_jerk').value}")
        self.get_logger().info(f"  Levels: FRESH<0.3, MILD<0.6, MODERATE<0.8, SEVERE>=0.8")
        self.get_logger().info("=" * 55)

    
    # Eye tracking data callback
    
    def eye_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 8:
            return
        self.raw_blink_rate = msg.data[0]
        self.raw_blink_duration = msg.data[1]
        self.raw_perclos = msg.data[2]
        self.raw_pupil_l = msg.data[3]
        self.raw_pupil_r = msg.data[4]
        # gaze_x = msg.data[5] : used in gaze fusion, not fatigue
        # gaze_y = msg.data[6]
        self.raw_eye_openness = msg.data[7]
        self.eye_data_received = True

    
    # Hand velocity callback: compute jerk
    
    def velocity_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            return
        speed = msg.data[3]   # current hand speed from camera_node
        now = time.time()
        dt = now - self.prev_velocity_time

        if dt > 0.001:
            # Jerk = rate of change of speed (simplified, true jerk is d³x/dt³)
            acceleration = abs(speed - self.prev_hand_speed) / dt
            self.hand_jerk_history.append(acceleration)
            # Rolling window: last 30 samples
            self.hand_jerk_history = self.hand_jerk_history[-30:]
            self.raw_hand_jerk = sum(self.hand_jerk_history) / len(self.hand_jerk_history)

        self.prev_hand_speed = speed
        self.prev_velocity_time = now
        self.velocity_data_received = True

    
    # Manual override callback (numpad keys)
    
    def override_callback(self, msg: String):
        new_override = msg.data.strip().upper() if msg.data else ""
        if new_override == self.manual_override:
            return
        if new_override and new_override not in self._override_scores:
            self.get_logger().warn(f"Ignoring invalid fatigue override: {new_override}")
            return
        old = self.manual_override or "AUTO"
        self.manual_override = new_override
        new_label = new_override or "AUTO"
        self.get_logger().warn(f"FATIGUE OVERRIDE: {old} → {new_label}")

    
    # Normalisation helpers
    
    def normalise(self, value, baseline, maximum):
        """Normalise value to 0-1 range given baseline (normal) and maximum (fatigued)"""
        if maximum <= baseline:
            return 0.0
        norm = (value - baseline) / (maximum - baseline)
        return max(0.0, min(1.0, norm))

    
    # Core computation
    
    def compute_and_publish(self):
        # Get parameters (live-tunable)
        w_br = self.get_parameter('w_blink_rate').value
        w_bd = self.get_parameter('w_blink_duration').value
        w_pc = self.get_parameter('w_perclos').value
        w_hj = self.get_parameter('w_hand_jerk').value
        alpha = self.get_parameter('smoothing_alpha').value

        br_base = self.get_parameter('blink_rate_baseline').value
        br_max = self.get_parameter('blink_rate_max').value
        bd_base = self.get_parameter('blink_dur_baseline').value
        bd_max = self.get_parameter('blink_dur_max').value
        pc_thresh = self.get_parameter('perclos_threshold').value
        jk_base = self.get_parameter('jerk_baseline').value
        jk_max = self.get_parameter('jerk_max').value

        # Normalise each component to 0-1 
        blink_rate_norm = self.normalise(self.raw_blink_rate, br_base, br_max)
        blink_dur_norm = self.normalise(self.raw_blink_duration, bd_base, bd_max)

        # PERCLOS: already 0-1 but scale relative to threshold
        # Above threshold = increasingly fatigued
        perclos_norm = min(1.0, self.raw_perclos / max(0.01, pc_thresh * 2.0))

        jerk_norm = self.normalise(self.raw_hand_jerk, jk_base, jk_max)

        # -Weighted composite
        raw_score = (w_br * blink_rate_norm +
                     w_bd * blink_dur_norm +
                     w_pc * perclos_norm +
                     w_hj * jerk_norm)
        raw_score = max(0.0, min(1.0, raw_score))

        # Exponential moving average for stability
        self.composite_score += alpha * (raw_score - self.composite_score)
        self.composite_score = max(0.0, min(1.0, self.composite_score))

        # Manual override takes precedence
        if self.manual_override:
            self.fatigue_level = self.manual_override
            self.composite_score = self._override_scores[self.manual_override]
        else:
            # Automatic level determination from computed score
            if self.composite_score < 0.30:
                self.fatigue_level = "FRESH"
            elif self.composite_score < 0.60:
                self.fatigue_level = "MILD"
            elif self.composite_score < 0.80:
                self.fatigue_level = "MODERATE"
            else:
                self.fatigue_level = "SEVERE"

        # Log level transitions
        if self.fatigue_level != self.prev_level:
            self.get_logger().warn(
                f"FATIGUE LEVEL CHANGE: {self.prev_level} → {self.fatigue_level} "
                f"(score={self.composite_score:.3f})")
            self.prev_level = self.fatigue_level

        # Publish score
        score_msg = Float32()
        score_msg.data = self.composite_score
        self.score_pub.publish(score_msg)

        # Publish level
        level_msg = String()
        level_msg.data = self.fatigue_level
        self.level_pub.publish(level_msg)

        # Publish components (for metrics panel + thesis data) 
        comp_msg = Float32MultiArray()
        comp_msg.data = [
            float(blink_rate_norm),         # [0] normalised blink rate
            float(blink_dur_norm),          # [1] normalised blink duration
            float(perclos_norm),            # [2] normalised PERCLOS
            float(jerk_norm),               # [3] normalised hand jerk
            float(self.composite_score),    # [4] composite score
            float(self.raw_blink_rate),     # [5] raw blink rate (blinks/min)
            float(self.raw_blink_duration), # [6] raw avg blink duration (s)
            float(self.raw_perclos),        # [7] raw PERCLOS
            float(self.raw_hand_jerk),      # [8] raw hand jerk
        ]
        self.components_pub.publish(comp_msg)


    # Periodic logging
  
    def log_state(self):
        eye_status = "YES" if self.eye_data_received else "NO"
        vel_status = "YES" if self.velocity_data_received else "NO"
        self.get_logger().info(
            f"FATIGUE: {self.fatigue_level} ({self.composite_score:.3f}) | "
            f"eye_data={eye_status} vel_data={vel_status} | "
            f"BR={self.raw_blink_rate:.0f}/min "
            f"BD={self.raw_blink_duration:.3f}s "
            f"PC={self.raw_perclos:.3f} "
            f"JK={self.raw_hand_jerk:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = FatigueMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

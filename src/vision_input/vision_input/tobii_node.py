#!/usr/bin/env python3
"""
tobii_node.py — Eye Tracking Interface
Human Intention Prediction System

Modes:
  SIMULATION: Realistic synthetic eye data with controllable fatigue ramp
  LIVE: Tobii Pro Glasses 3 via WiFi (g3pylib)

Topics Published:
  /eye_tracking/raw          Float32MultiArray  [blink_rate, avg_blink_dur, perclos,
                                                  pupil_l, pupil_r, gaze_x, gaze_y,
                                                  eye_openness]
  /eye_tracking/gaze_point   PointStamped       Gaze mapped to workspace coordinates
  /eye_tracking/status       String             Connection and fatigue info
  /soft_estop                Bool               Feature 3: TEPR-triggered soft e-stop latch
  /gaze/committed            Bool               Feature 4: fixation ≥400ms reached
  /gaze/speed_dps            Float32            Feature 4: instantaneous gaze angular speed
  /gaze/is_fixation          Bool               Feature 4: classifier state (vel < 100 dps)

Topics Subscribed:
  /soft_estop_manual         Bool               Wizard-of-Oz trigger (e.g. Numpad 5)

Parameters:
  simulation_mode        bool   (default: True)
  fatigue_ramp_minutes   float  (default: 10.0)  Time from fresh to severe
  sim_fatigue_override   float  (default: -1.0)  Manual override (-1 = auto ramp)
  publish_rate           float  (default: 10.0)  Hz
  tobii_ip               string (default: '192.168.75.51')
  tepr_csv_path          string (default: ~/ros2_ws/src/report/data/tepr_log.csv)
  gaze_csv_path          string (default: ~/ros2_ws/src/report/data/gaze_log.csv)

Live demo control:
  ros2 param set /tobii_node sim_fatigue_override 0.0   -> Fresh
  ros2 param set /tobii_node sim_fatigue_override 0.5   -> Mild/Moderate
  ros2 param set /tobii_node sim_fatigue_override 0.85  -> Severe
  ros2 param set /tobii_node sim_fatigue_override -1.0  -> Resume auto ramp
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
from geometry_msgs.msg import PointStamped
import time
import math
import random
import csv
import os
import threading
import queue
from collections import deque
from pathlib import Path

# Optional: live Tobii streaming via g3pylib.
try:
    import asyncio
    from g3pylib import connect_to_glasses
    HAS_G3PYLIB = True
except Exception:
    HAS_G3PYLIB = False


# TEPR parameters (Feature 3)
BASELINE_WIN_S = 3.0
SPIKE_FRAC = 0.15
ESTOP_COOLDOWN_S = 3.0

# Gaze commitment parameters (Feature 4)
SACCADE_VEL_DPS = 100.0
FIXATION_WIN_DEG = 2.0
FIXATION_DWELL_S = 0.4
FORCE_SACCADE_HOLD_S = 2.0  # WoZ Numpad 6 pin duration

# Simulation-only FOV for converting 2D normalized gaze → unit 3D direction.
# Live mode uses Tobii gaze3d directly (FOV-independent).
SIM_FOV_H_DEG = 95.0
SIM_FOV_V_DEG = 63.0


class TobiiNode(Node):
    def __init__(self):
        super().__init__('tobii_node')


        # Parameters

        self.declare_parameter('simulation_mode', True)
        self.declare_parameter('fatigue_ramp_minutes', 10.0)
        self.declare_parameter('sim_fatigue_override', -1.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('tobii_ip', '192.168.75.51')
        self.declare_parameter(
            'tepr_csv_path',
            str(Path.home() / 'ros2_ws' / 'src' / 'report' / 'data' / 'tepr_log.csv'))
        self.declare_parameter(
            'gaze_csv_path',
            str(Path.home() / 'ros2_ws' / 'src' / 'report' / 'data' / 'gaze_log.csv'))

        self.simulation_mode = self.get_parameter('simulation_mode').value
        publish_rate = self.get_parameter('publish_rate').value

        # Publishers
        self.raw_pub = self.create_publisher(
            Float32MultiArray, '/eye_tracking/raw', 10)
        self.gaze_pub = self.create_publisher(
            PointStamped, '/eye_tracking/gaze_point', 10)
        self.status_pub = self.create_publisher(
            String, '/eye_tracking/status', 10)

        # Feature 3
        self.soft_estop_pub = self.create_publisher(Bool, '/soft_estop', 10)
        # Feature 4
        self.gaze_committed_pub = self.create_publisher(Bool, '/gaze/committed', 10)
        self.gaze_speed_pub = self.create_publisher(Float32, '/gaze/speed_dps', 10)
        self.is_fixation_pub = self.create_publisher(Bool, '/gaze/is_fixation', 10)

        # Subscribers (Wizard-of-Oz fallback for soft e-stop)
        self.create_subscription(
            Bool, '/soft_estop_manual', self._manual_estop_cb, 10)
        # WoZ force-saccade: Numpad 6 pretends the operator glanced away
        # for FORCE_SACCADE_HOLD_S, pinning /gaze/committed False.
        self.create_subscription(
            Bool, '/gaze/force_saccade', self._force_saccade_cb, 10)
        self._force_saccade_until = 0.0  # monotonic time.time()

        # Simulation state
        self.start_time = time.time()

        # Blink tracking
        self.blink_history = []          # timestamps of blink starts
        self.blink_durations = []        # durations of recent blinks
        self.sim_blink_active = False
        self.sim_blink_end_time = 0.0
        self.sim_next_blink_time = time.time() + random.uniform(2.0, 5.0)

        # PERCLOS rolling window
        self.perclos_window = []         # (timestamp, is_closed) tuples
        self.PERCLOS_WINDOW_SEC = 60.0   # 1-minute rolling window

        # Gaze simulation
        self.sim_gaze_x = 0.5
        self.sim_gaze_y = 0.5
        self.sim_gaze_target_x = 0.5
        self.sim_gaze_target_y = 0.5
        self.sim_saccade_timer = time.time() + random.uniform(0.5, 2.0)

        # Pupil
        self.sim_pupil_l = 4.0          # mm baseline
        self.sim_pupil_r = 4.0

        # Feature 3: TEPR state
        self._pupil_buffer = deque()     # (t, pupil_mm) pairs, trimmed to BASELINE_WIN_S
        self._estop_latched = False
        self._estop_release_at = 0.0

        self._tepr_csv_path = Path(self.get_parameter('tepr_csv_path').value)
        self._tepr_csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_tepr_file = not self._tepr_csv_path.exists()
        self._tepr_csv = open(self._tepr_csv_path, 'a', newline='')
        self._tepr_writer = csv.writer(self._tepr_csv)
        if new_tepr_file:
            self._tepr_writer.writerow(['t_sec', 'pupil_mm', 'baseline_mm', 'trigger'])
            self._tepr_csv.flush()
        self._tepr_t0 = time.time()

        # Feature 4: Gaze commitment state

        self._prev_gaze_dir = None       # last 3D unit vector
        self._prev_gaze_t = None
        self._fixation_start_t = None
        self._fixation_anchor = None     # gaze_dir at start of current candidate
        self._gaze_committed = False
        self._last_is_fixation = False

        # Live Tobii (g3pylib) plumbing

        # Samples pushed from background asyncio thread → drained in update_live.
        self._live_queue = queue.Queue(maxsize=200)
        self._live_stop = threading.Event()
        self._live_thread = None
        self._live_warned = False
        self._live_last_rx_time = 0.0

        if not self.simulation_mode:
            self._start_live_thread()

        # Timers
        self.create_timer(1.0 / publish_rate, self.update)
        self.create_timer(1.0, self.publish_status)
        self.create_timer(5.0, self.log_sim_state)
        # Soft e-stop auto-release + publish heartbeat at 20 Hz so robot_controller
        # sees a fresh signal even if pupil samples stop arriving.
        self.create_timer(0.05, self._soft_estop_heartbeat)

        # Startup log
        self.get_logger().info("=" * 55)
        self.get_logger().info("  TOBII EYE TRACKING NODE")
        mode = "SIMULATION" if self.simulation_mode else "LIVE"
        self.get_logger().info(f"  Mode: {mode}")
        if self.simulation_mode:
            ramp = self.get_parameter('fatigue_ramp_minutes').value
            self.get_logger().info(f"  Fatigue ramp: {ramp:.1f} minutes")
            self.get_logger().info(f"  Override: ros2 param set /tobii_node sim_fatigue_override <0-1>")
        else:
            self.get_logger().info(f"  Tobii IP: {self.get_parameter('tobii_ip').value}")
            self.get_logger().info(f"  g3pylib available: {HAS_G3PYLIB}")
        self.get_logger().info(f"  TEPR log:  {self._tepr_csv_path}")
        self.get_logger().info("=" * 55)

    # Fatigue level (drives simulation realism)
    def get_sim_fatigue_level(self):
        """Returns 0.0 (fresh) to 1.0 (severe) for simulation"""
        override = self.get_parameter('sim_fatigue_override').value
        if override >= 0.0:
            return max(0.0, min(1.0, override))

        elapsed_min = (time.time() - self.start_time) / 60.0
        ramp_min = self.get_parameter('fatigue_ramp_minutes').value
        if ramp_min <= 0:
            return 0.0
        return min(1.0, elapsed_min / ramp_min)

    # Main update
    def update(self):
        if self.simulation_mode:
            self.update_simulation()
        else:
            self.update_live()

    # Simulation engine

    def update_simulation(self):
        now = time.time()
        fatigue = self.get_sim_fatigue_level()

        # BLINK GENERATION
        # Fresh: ~15 blinks/min (one every ~4s)
        # Fatigued: ~25-30 blinks/min (one every ~2s) with longer durations
        base_interval = 4.0 - (fatigue * 2.5)
        base_interval = max(1.2, base_interval)

        if not self.sim_blink_active and now >= self.sim_next_blink_time:
            self.sim_blink_active = True

            # Blink duration: fresh ~150ms, fatigued ~300-400ms
            base_dur = 0.15 + fatigue * 0.35
            duration = base_dur + random.gauss(0, 0.04)
            duration = max(0.08, min(0.8, duration))

            # Microsleep at high fatigue (>0.5)
            if fatigue > 0.5 and random.random() < (fatigue - 0.5) * 0.4:
                duration = random.uniform(0.5, 2.0)
                self.get_logger().warn(
                    f"SIM: Microsleep event ({duration:.2f}s) "
                    f"[fatigue={fatigue:.2f}]")

            self.sim_blink_end_time = now + duration
            self.blink_history.append(now)
            self.blink_durations.append(duration)

            # Keep only last 60s of history
            self.blink_history = [t for t in self.blink_history if t > now - 60.0]
            self.blink_durations = self.blink_durations[-30:]

        if self.sim_blink_active and now >= self.sim_blink_end_time:
            self.sim_blink_active = False
            jitter = random.gauss(0, base_interval * 0.3)
            self.sim_next_blink_time = now + max(0.5, base_interval + jitter)

        # EYE OPENNESS
        if self.sim_blink_active:
            eye_openness = 0.0
        elif fatigue > 0.3:
            # Droopy eyelids: partial closure
            droop = (fatigue - 0.3) * 0.7     # 0 to 0.49
            noise = random.gauss(0, 0.03)
            eye_openness = max(0.2, min(1.0, 1.0 - droop + noise))
        else:
            eye_openness = 1.0 + random.gauss(0, 0.01)
            eye_openness = min(1.0, eye_openness)

        # PERCLOS
        is_closed = eye_openness < 0.3
        self.perclos_window.append((now, is_closed))
        cutoff = now - self.PERCLOS_WINDOW_SEC
        self.perclos_window = [(t, c) for t, c in self.perclos_window if t > cutoff]

        if len(self.perclos_window) > 10:
            perclos = sum(1 for _, c in self.perclos_window if c) / len(self.perclos_window)
        else:
            perclos = 0.0

        # BLINK METRICS
        blink_rate = float(len(self.blink_history))   # per minute

        recent_dur = self.blink_durations[-10:] if self.blink_durations else [0.15]
        avg_blink_duration = sum(recent_dur) / len(recent_dur)

        # PUPIL DIAMETER
        # Fatigue -> slight dilation, more variability
        base_pupil = 4.0 + fatigue * 0.8
        self.sim_pupil_l += 0.3 * (base_pupil + random.gauss(0, 0.15) - self.sim_pupil_l)
        self.sim_pupil_r += 0.3 * (base_pupil + random.gauss(0, 0.15) - self.sim_pupil_r)
        self.sim_pupil_l = max(2.0, min(8.0, self.sim_pupil_l))
        self.sim_pupil_r = max(2.0, min(8.0, self.sim_pupil_r))

        # GAZE POINT
        # Fresh: purposeful saccades, stable fixations
        # Fatigued: slower saccades, more drift, less precision
        if now >= self.sim_saccade_timer:
            # New saccade target
            saccade_range = 0.3 + fatigue * 0.2
            self.sim_gaze_target_x = 0.5 + random.uniform(-saccade_range, saccade_range)
            self.sim_gaze_target_y = 0.5 + random.uniform(-saccade_range, saccade_range)
            self.sim_gaze_target_x = max(0.05, min(0.95, self.sim_gaze_target_x))
            self.sim_gaze_target_y = max(0.05, min(0.95, self.sim_gaze_target_y))

            # Fresh: frequent purposeful saccades. Fatigued: slower, less frequent
            interval = random.uniform(0.3, 1.5) + fatigue * 1.0
            self.sim_saccade_timer = now + interval

        # Move gaze toward target with speed dependent on fatigue
        saccade_speed = 0.15 - fatigue * 0.08   # slower when tired
        saccade_speed = max(0.03, saccade_speed)

        self.sim_gaze_x += saccade_speed * (self.sim_gaze_target_x - self.sim_gaze_x)
        self.sim_gaze_y += saccade_speed * (self.sim_gaze_target_y - self.sim_gaze_y)

        # Drift noise (increases with fatigue)
        drift = 0.002 + fatigue * 0.01
        self.sim_gaze_x += random.gauss(0, drift)
        self.sim_gaze_y += random.gauss(0, drift)
        self.sim_gaze_x = max(0.0, min(1.0, self.sim_gaze_x))
        self.sim_gaze_y = max(0.0, min(1.0, self.sim_gaze_y))

        # PUBLISH RAW
        raw_msg = Float32MultiArray()
        raw_msg.data = [
            float(blink_rate),            # [0] blinks per minute
            float(avg_blink_duration),    # [1] seconds
            float(perclos),               # [2] 0.0 — 1.0
            float(self.sim_pupil_l),      # [3] mm
            float(self.sim_pupil_r),      # [4] mm
            float(self.sim_gaze_x),       # [5] normalised 0-1
            float(self.sim_gaze_y),       # [6] normalised 0-1
            float(eye_openness),          # [7] 0=closed, 1=open
        ]
        self.raw_pub.publish(raw_msg)

        # PUBLISH GAZE POINT (mapped to workspace)
        gaze_msg = PointStamped()
        gaze_msg.header.stamp = self.get_clock().now().to_msg()
        gaze_msg.header.frame_id = "base_link"
        gaze_msg.point.x = 0.3
        gaze_msg.point.y = -(self.sim_gaze_x - 0.5) * 0.7
        gaze_msg.point.z = -(self.sim_gaze_y - 0.5) * 0.55 + 0.30
        self.gaze_pub.publish(gaze_msg)

        # Feature 3 & 4: feed pupil + gaze into detectors
        # Pupil input is zeroed while the eye is fully closed, real blinks
        # produce spurious pupil readings too, so we skip during blinks.
        if not self.sim_blink_active:
            self._process_pupil_sample(now, self.sim_pupil_l, self.sim_pupil_r)
        sim_dir = self._sim_gaze_to_unit_vector(self.sim_gaze_x, self.sim_gaze_y)
        self._process_gaze_sample(now, sim_dir)

    # Feature 3: TEPR Soft E-Stop detector

    def _process_pupil_sample(self, now, pupil_l, pupil_r):
        """Feed one (left, right) pupil sample into the rolling baseline and
        TEPR trigger logic. Skips invalid samples; auto-releases the latch
        after ESTOP_COOLDOWN_S. Appends one row per sample to tepr_log.csv.
        """
        # Reject invalid Tobii values (blink or dropped stream)
        if pupil_l <= 0.0 or pupil_r <= 0.0:
            return

        pupil_mm = (pupil_l + pupil_r) / 2.0

        # Maintain 3-second rolling buffer
        self._pupil_buffer.append((now, pupil_mm))
        cutoff = now - BASELINE_WIN_S
        while self._pupil_buffer and self._pupil_buffer[0][0] < cutoff:
            self._pupil_buffer.popleft()

        # Baseline needs a warm-up: before then, use the current sample
        # so baseline==pupil and no spurious trigger fires.
        if len(self._pupil_buffer) < 5:
            baseline_mm = pupil_mm
        else:
            baseline_mm = sum(p for _, p in self._pupil_buffer) / len(self._pupil_buffer)

        # Trigger on edge: rising above +15% for the first time
        trigger_flag = 0
        if pupil_mm > baseline_mm * (1.0 + SPIKE_FRAC) and not self._estop_latched:
            self._estop_latched = True
            self._estop_release_at = now + ESTOP_COOLDOWN_S
            trigger_flag = 1
            self.get_logger().warn(
                f"TEPR SOFT E-STOP: pupil={pupil_mm:.2f}mm "
                f"> baseline×1.15={baseline_mm * 1.15:.2f}mm "
                f"(cooldown {ESTOP_COOLDOWN_S:.1f}s)")

        # CSV row: schema: t_sec,pupil_mm,baseline_mm,trigger
        self._tepr_writer.writerow([
            f"{now - self._tepr_t0:.4f}",
            f"{pupil_mm:.4f}",
            f"{baseline_mm:.4f}",
            trigger_flag,
        ])
        self._tepr_csv.flush()

    def _manual_estop_cb(self, msg: Bool):
        """Wizard-of-Oz path: any True on /soft_estop_manual latches the
        same soft e-stop as the TEPR detector would."""
        if msg.data and not self._estop_latched:
            now = time.time()
            self._estop_latched = True
            self._estop_release_at = now + ESTOP_COOLDOWN_S
            self.get_logger().warn(
                f"MANUAL SOFT E-STOP: /soft_estop_manual triggered "
                f"(cooldown {ESTOP_COOLDOWN_S:.1f}s)")

    def _force_saccade_cb(self, msg: Bool):
        """WoZ path: any True on /gaze/force_saccade pins /gaze/committed
        to False for FORCE_SACCADE_HOLD_S, regardless of what the
        classifier says. Auto-releases at the end of the hold."""
        if msg.data:
            self._force_saccade_until = time.time() + FORCE_SACCADE_HOLD_S
            self.get_logger().warn(
                f"FORCE SACCADE: /gaze/committed pinned False "
                f"for {FORCE_SACCADE_HOLD_S:.1f}s")

    def _soft_estop_heartbeat(self):
        """20 Hz publisher for /soft_estop so the robot_controller sees a
        continuous signal; also enforces the auto-release cooldown even if
        pupil samples have stopped arriving (Tobii drop fail-safe)."""
        now = time.time()
        if self._estop_latched and now >= self._estop_release_at:
            self._estop_latched = False
        out = Bool()
        out.data = bool(self._estop_latched)
        self.soft_estop_pub.publish(out)


    # Feature 4: Saccade-gated prediction commitment

    def _sim_gaze_to_unit_vector(self, gaze_x, gaze_y):
        """Simulation helper: convert normalised 2D gaze (0-1) into a 3D
        unit vector using a fixed FOV. Live mode uses Tobii gaze3d directly
        and should NOT call this, it's sim-only so synthesized saccades
        look like ~500 dps bursts, matching the plotting script's fixture.
        """
        ax = math.radians((gaze_x - 0.5) * SIM_FOV_H_DEG)
        ay = math.radians((gaze_y - 0.5) * SIM_FOV_V_DEG)
        # Forward-axis = +z; x = horizontal, y = vertical
        x = math.sin(ax) * math.cos(ay)
        y = math.sin(ay)
        z = math.cos(ax) * math.cos(ay)
        mag = math.sqrt(x * x + y * y + z * z)
        if mag < 1e-9:
            return (0.0, 0.0, 1.0)
        return (x / mag, y / mag, z / mag)

    def _process_gaze_sample(self, now, gaze_dir):
        """Classify the latest gaze sample as saccade/fixation, update the
        400ms commitment latch, and publish the trio of gaze topics.

        gaze_dir: 3D unit vector (tuple or list of 3 floats).
        """
        if gaze_dir is None:
            return

        if self._prev_gaze_dir is None or self._prev_gaze_t is None:
            self._prev_gaze_dir = gaze_dir
            self._prev_gaze_t = now
            return

        dt = now - self._prev_gaze_t
        if dt < 1e-4:
            return

        # Angular velocity via dot product of unit vectors (FOV-independent).
        dot_prev = (gaze_dir[0] * self._prev_gaze_dir[0] +
                    gaze_dir[1] * self._prev_gaze_dir[1] +
                    gaze_dir[2] * self._prev_gaze_dir[2])
        dot_prev = max(-1.0, min(1.0, dot_prev))
        angle_step_deg = math.degrees(math.acos(dot_prev))
        speed_dps = angle_step_deg / dt

        is_fixation_now = speed_dps < SACCADE_VEL_DPS

        if not is_fixation_now:
            # Saccade: drop commitment and reset candidate
            self._fixation_start_t = None
            self._fixation_anchor = None
            self._gaze_committed = False
        else:
            # Fixation candidate: track dwell + angular spread from anchor
            if self._fixation_start_t is None:
                self._fixation_start_t = now
                self._fixation_anchor = gaze_dir
            else:
                dot_anchor = (gaze_dir[0] * self._fixation_anchor[0] +
                              gaze_dir[1] * self._fixation_anchor[1] +
                              gaze_dir[2] * self._fixation_anchor[2])
                dot_anchor = max(-1.0, min(1.0, dot_anchor))
                spread_deg = math.degrees(math.acos(dot_anchor))
                if spread_deg > FIXATION_WIN_DEG:
                    # Wandered out of the 2° window: restart candidate here
                    self._fixation_start_t = now
                    self._fixation_anchor = gaze_dir
                    self._gaze_committed = False
                elif (now - self._fixation_start_t) >= FIXATION_DWELL_S:
                    self._gaze_committed = True

        self._last_is_fixation = is_fixation_now

        # WoZ override: pin committed False while the force-saccade timer
        # is active. Does NOT touch _gaze_committed itself, so the
        # classifier resumes cleanly once the hold expires.
        if now < self._force_saccade_until:
            committed_out = False
        else:
            committed_out = bool(self._gaze_committed)

        # Publish
        self.gaze_speed_pub.publish(Float32(data=float(speed_dps)))
        self.is_fixation_pub.publish(Bool(data=bool(is_fixation_now)))
        self.gaze_committed_pub.publish(Bool(data=committed_out))

        self._prev_gaze_dir = gaze_dir
        self._prev_gaze_t = now

    # Live Tobii connection: g3pylib in a background asyncio thread
    def _start_live_thread(self):
        """Launch the asyncio event loop in a daemon thread. Samples are
        pushed to self._live_queue; update_live() drains them on the ROS
        timer tick so we stay thread-safe with rclpy callbacks."""
        if not HAS_G3PYLIB:
            self.get_logger().warn(
                "LIVE MODE: g3pylib is not installed. Install with "
                "`pip install g3pylib` — falling back to empty data.")
            return
        self._live_thread = threading.Thread(
            target=self._live_worker, daemon=True, name='tobii_live')
        self._live_thread.start()

    def _live_worker(self):
        """Background thread: run the g3pylib asyncio loop forever."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._live_async_loop())
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.get_logger().error(
                f"Live Tobii thread crashed: {type(e).__name__}: {e}\n{tb}"
                "Soft e-stop fail-safe will auto-release; gaze gate will default open.")

    async def _live_async_loop(self):
        """Connect to the glasses and stream gaze/pupil into the queue."""
        tobii_ip = self.get_parameter('tobii_ip').value
        self.get_logger().info(f"Connecting to Tobii at {tobii_ip} (via with_hostname) ...")
        async with connect_to_glasses.with_hostname(tobii_ip) as g3:
            async with g3.stream_rtsp(scene_camera=False, gaze=True) as streams:
                self.get_logger().info("Tobii gaze stream active.")
                # DataStream.decode() is an async context manager that yields
                # a queue of (json_dict, timestamp) tuples. The JSON dict is
                # the Tobii G3 gaze record: keys like 'gaze3d', 'eyeleft',
                # 'eyeright'. _extract_live_sample handles dict access.
                async with streams.gaze.decode() as gaze_queue:
                    while not self._live_stop.is_set():
                        gaze, _ts = await gaze_queue.get()
                        t_now = time.time()
                        self._live_last_rx_time = t_now
                        try:
                            self._live_queue.put_nowait(
                                self._extract_live_sample(t_now, gaze))
                        except queue.Full:
                            try:
                                self._live_queue.get_nowait()
                                self._live_queue.put_nowait(
                                    self._extract_live_sample(t_now, gaze))
                            except queue.Empty:
                                pass

    def _extract_live_sample(self, t_now, gaze):
        """Pull pupil + gaze direction out of whatever g3pylib hands us.
        Field names across g3pylib versions vary; we try a few. Returns
        (t, pupil_l, pupil_r, gaze_dir_tuple_or_None).
        """
        # Pupil diameter per eye
        pupil_l = 0.0
        pupil_r = 0.0
        try:
            pupil_l = float(gaze.eyeleft.pupildiameter)
        except Exception:
            try:
                pupil_l = float(gaze['eyeleft']['pupildiameter'])
            except Exception:
                pass
        try:
            pupil_r = float(gaze.eyeright.pupildiameter)
        except Exception:
            try:
                pupil_r = float(gaze['eyeright']['pupildiameter'])
            except Exception:
                pass

        # 3D gaze direction (unit vector in scene-camera frame)
        gaze_dir = None
        raw = None
        for attr in ('gaze3d', 'gazeDirection', 'gaze_direction'):
            try:
                raw = getattr(gaze, attr)
                break
            except Exception:
                try:
                    raw = gaze[attr]
                    break
                except Exception:
                    continue
        if raw is not None:
            try:
                x = float(raw[0]); y = float(raw[1]); z = float(raw[2])
                mag = math.sqrt(x * x + y * y + z * z)
                if mag > 1e-6:
                    gaze_dir = (x / mag, y / mag, z / mag)
            except Exception:
                gaze_dir = None

        return (t_now, pupil_l, pupil_r, gaze_dir)

    def update_live(self):
        """Drain any queued live samples into the pupil + gaze pipelines."""
        drained_any = False
        while True:
            try:
                t_now, pupil_l, pupil_r, gaze_dir = self._live_queue.get_nowait()
            except queue.Empty:
                break
            drained_any = True
            self._process_pupil_sample(t_now, pupil_l, pupil_r)
            self._process_gaze_sample(t_now, gaze_dir)

        # Mirror the "raw" topic so downstream consumers (fatigue_monitor) keep
        # receiving data in LIVE mode: blink/PERCLOS pipeline not yet wired
        # to the live stream, so those fields stay zero and must not be used
        # by downstream logic in LIVE mode (documented in the header).
        if not drained_any and not self._live_warned:
            self.get_logger().warn(
                "LIVE MODE: no samples received from Tobii yet — check the "
                "glasses connection. Soft e-stop will auto-release; gaze "
                "commitment will default open.")
            self._live_warned = True

        raw_msg = Float32MultiArray()
        raw_msg.data = [0.0] * 8
        self.raw_pub.publish(raw_msg)

    # Status and logging
    def publish_status(self):
        msg = String()
        if self.simulation_mode:
            fatigue = self.get_sim_fatigue_level()
            override = self.get_parameter('sim_fatigue_override').value
            mode = "OVERRIDE" if override >= 0 else "AUTO_RAMP"
            msg.data = f"SIMULATED|{mode}|fatigue={fatigue:.3f}"
        else:
            age = time.time() - self._live_last_rx_time if self._live_last_rx_time else -1.0
            if age < 0:
                msg.data = "LIVE|NOT_CONNECTED"
            elif age > 1.0:
                msg.data = f"LIVE|STALE|last_rx={age:.1f}s"
            else:
                msg.data = f"LIVE|OK|rx_age={age:.2f}s"
        self.status_pub.publish(msg)

    def log_sim_state(self):
        if not self.simulation_mode:
            return
        fatigue = self.get_sim_fatigue_level()
        elapsed = (time.time() - self.start_time) / 60.0
        blink_rate = len(self.blink_history)
        recent_dur = self.blink_durations[-10:] if self.blink_durations else [0.0]
        avg_dur = sum(recent_dur) / len(recent_dur)

        perclos = 0.0
        if len(self.perclos_window) > 10:
            perclos = sum(1 for _, c in self.perclos_window if c) / len(self.perclos_window)

        self.get_logger().info(
            f"SIM [{elapsed:.1f}min] fatigue={fatigue:.2f} | "
            f"blinks={blink_rate}/min | dur={avg_dur:.3f}s | "
            f"PERCLOS={perclos:.3f} | "
            f"pupil=({self.sim_pupil_l:.1f},{self.sim_pupil_r:.1f})mm | "
            f"estop={self._estop_latched} | committed={self._gaze_committed}")

    def destroy_node(self):
        # Clean up the live thread and close CSV before teardown.
        self._live_stop.set()
        try:
            self._tepr_csv.flush()
            self._tepr_csv.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TobiiNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import csv
import mediapipe as mp
import joblib
import os
import math
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as RosPoint
import time 


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('camera_id', 0)

        # 1. Load the AI Brain
        home = os.path.expanduser('~')
        model_path = os.path.join(home, 'ros2_ws', 'data', 'gesture_brain.joblib')
        
        self.get_logger().info(f"Loading AI Brain from: {model_path}")
        try:
            self.brain = joblib.load(model_path)
            self.brain_ready = True
            self.get_logger().info("BRAIN LOADED SUCCESSFULLY!")
        except Exception as e:
            self.get_logger().error(f"FAILED TO LOAD BRAIN: {e}")
            self.brain_ready = False

        self.gesture_names = {0: 'GRASP', 1: 'POINT', 2: 'STOP', 3: 'OKAY', 4: 'NEUTRAL'}

        # 2. Publishers
        self.image_pub = self.create_publisher(Image, '/vision_input/hand_stream', 10)
        # Backward-compat: same as right_hand_landmarks
        self.landmark_pub = self.create_publisher(Float32MultiArray, '/vision_input/hand_landmarks', 10)
        self.right_landmark_pub = self.create_publisher(Float32MultiArray, '/vision_input/right_hand_landmarks', 10)
        self.left_landmark_pub = self.create_publisher(Float32MultiArray, '/vision_input/left_hand_landmarks', 10)
        self.gesture_pub = self.create_publisher(String, '/vision_input/gesture_text', 10)
        self.marker_pub = self.create_publisher(Marker, '/vision_input/intention_marker', 10)
        self.velocity_pub = self.create_publisher(Float32MultiArray, '/vision_input/hand_velocity', 10)

        # Subscribe to robot mode for camera-feed HUD overlay
        self.current_robot_mode = "FETCH"
        self.mode_sub = self.create_subscription(
            String, '/vision_input/robot_mode', self._robot_mode_cb, 10)

        # 3. MediaPipe: 2 hands for bimanual mode
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        cam_id = self.get_parameter('camera_id').value
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.033, self.timer_callback)

        # 4. Hand velocity tracking (ONE history, ONE format)
        self.hand_history = []

        # Gesture smoothing buffer
        self.gesture_buffer = []
        self.buffer_size = 5  # Must see same gesture 5 times in a row to confirm it

        # Velocity Gating
        # Freezes fingertip position & gesture output when fingers
        # are in transit between gestures (the "transitional frame"
        # problem). Uses fingertip velocity, not wrist velocity,
        # because the fingertip moves far more during gesture changes.
        self._vg_prev_tips = None       # (index_tip, middle_tip) from prev frame
        self._vg_prev_time = None
        self._vg_frozen = False         # True while velocity gate is active
        self._vg_threshold = 1.8        # normalised units/s — tuned to finger morphing speed
        self._vg_release_threshold = 0.6  # must drop below this to unfreeze
        self._vg_frozen_landmarks = None  # last good landmark msg to re-publish while frozen
        self._vg_consecutive = 0         # consecutive stable-gesture frame count
        self._vg_required_frames = 10    # ~333ms at 30fps before accepting new gesture

        _latency_path = os.path.join(os.path.expanduser('~'),
            'ros2_ws', 'data', 'latency_log.csv')
        _latency_need_header = not os.path.exists(_latency_path) \
            or os.path.getsize(_latency_path) == 0
        self.latency_log = open(_latency_path, 'a', newline='')
        self.latency_writer = csv.writer(self.latency_log)
        if _latency_need_header:
            self.latency_writer.writerow([
                'event', 'gesture', 'timestamp'
            ])
            self.latency_log.flush()


    # VELOCITY GATE: filters transitional frames

    def _update_velocity_gate(self, hand_landmarks):
        """
        Compute fingertip velocity. If fingers are moving fast
        (gesture transition in progress), activate the gate to
        freeze landmark output and ignore classifier noise.
        Returns True if the gate is OPEN (safe to use data),
        False if FROZEN (transitional, use last good data).
        """
        now = time.time()
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]

        if self._vg_prev_tips is None or self._vg_prev_time is None:
            self._vg_prev_tips = (index_tip.x, index_tip.y, middle_tip.x, middle_tip.y)
            self._vg_prev_time = now
            return True  # first frame, allow through

        dt = now - self._vg_prev_time
        if dt < 0.001:
            return not self._vg_frozen

        # Fingertip velocity: index + middle combined
        prev_ix, prev_iy, prev_mx, prev_my = self._vg_prev_tips
        v_index = math.sqrt((index_tip.x - prev_ix)**2 + (index_tip.y - prev_iy)**2) / dt
        v_middle = math.sqrt((middle_tip.x - prev_mx)**2 + (middle_tip.y - prev_my)**2) / dt
        finger_speed = max(v_index, v_middle)

        self._vg_prev_tips = (index_tip.x, index_tip.y, middle_tip.x, middle_tip.y)
        self._vg_prev_time = now

        if self._vg_frozen:
            # Release only when fingers have clearly settled
            if finger_speed < self._vg_release_threshold:
                self._vg_frozen = False
                self._vg_consecutive = 0  # reset debounce counter on unfreeze
        else:
            # Freeze if fingers are moving fast (gesture transition)
            if finger_speed > self._vg_threshold:
                self._vg_frozen = True

        return not self._vg_frozen

    def _robot_mode_cb(self, msg: String):
        self.current_robot_mode = msg.data

    def get_stable_gesture(self, raw_gesture):
        """
        Only change gesture if we see the SAME one
        multiple frames in a row. Prevents flickering.
        Uses two layers:
          1. Velocity gate (blocks during transitions)
          2. Consecutive-frame debounce (10 frames = ~333ms)
        """

        # STOP: 2-frame debounce (~66 ms at 30 fps), filters single-frame
        # RF noise that was locking the E-Stop on startup. Still well within
        # ISO 15066 response-time margins for collaborative robots.
        if raw_gesture == "STOP":
            self._stop_candidate_count = getattr(self, '_stop_candidate_count', 0) + 1
            if self._stop_candidate_count >= 2:
                self.gesture_buffer.clear()
                self._vg_consecutive = 0
                self._vg_frozen = False
                self.last_stable_gesture = "STOP"
                return "STOP"
            # One-frame STOP: fall through so it doesn't latch the E-Stop.
            return getattr(self, 'last_stable_gesture', 'NEUTRAL')
        else:
            self._stop_candidate_count = 0

        # If velocity gate is frozen, ignore classifier entirely
        if self._vg_frozen:
            if hasattr(self, 'last_stable_gesture'):
                return self.last_stable_gesture
            return raw_gesture

        self.gesture_buffer.append(raw_gesture)

        # Keep buffer at fixed size
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        # Consecutive-frame debounce: must see the SAME gesture
        # for N frames in a row before switching
        current = getattr(self, 'last_stable_gesture', None)
        if raw_gesture == current:
            # Same as current: no switch needed, reset counter
            self._vg_consecutive = 0
            return current
        else:
           # Different from current: count consecutive appearances
            self._vg_consecutive += 1
            if self._vg_consecutive >= self._vg_required_frames:
                # Seen it enough times: accept the switch
                self._vg_consecutive = 0
                return raw_gesture
            else:
                # Not enough evidence yet: hold current gesture
                if current is not None:
                    return current
                return raw_gesture

    # VELOCITY PREDICTION (One clean function)

    def compute_and_publish_velocity(self, hand_landmarks):
        """
        Track where the hand WAS and predict where it WILL BE.
        Publishes velocity data for the ghost robot.
        """
        wrist = hand_landmarks.landmark[0]
        now = time.time()
        
        # Remember where the hand is
        self.hand_history.append((wrist.x, wrist.y, wrist.z, now))
        
        # Only keep last 10 positions
        if len(self.hand_history) > 10:
            self.hand_history.pop(0)
        
        # Need at least 3 positions to calculate velocity
        if len(self.hand_history) < 3:
            return
        
        # Where was the hand 3 frames ago?
        old_x, old_y, old_z, old_time = self.hand_history[-3]
        # Where is it now?
        new_x, new_y, new_z, new_time = self.hand_history[-1]
        
        # How much time passed?
        dt = new_time - old_time
        if dt < 0.001:
            return
        
        # VELOCITY = how fast and in what direction
        vel_x = (new_x - old_x) / dt
        vel_y = (new_y - old_y) / dt
        vel_z = (new_z - old_z) / dt
        
        # SPEED = how fast overall
        speed = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        
        # Only predict if hand is moving fast enough: reduce so it at appears even with slow movements, if you remove it, it always appears
        if speed < 0.2:  
            return
        
        # PREDICT: Where will the hand be in 0.5 seconds?
        predict_time = 0.5
        pred_x = max(0.0, min(1.0, new_x + vel_x * predict_time))
        pred_y = max(0.0, min(1.0, new_y + vel_y * predict_time))
        pred_z = max(-1.0, min(1.0, new_z + vel_z * predict_time))
        
        # Publish velocity + prediction for robot_controller ghost
        msg = Float32MultiArray()
        msg.data = [
            float(vel_x), float(vel_y), float(vel_z),
            float(speed),
            float(pred_x), float(pred_y), float(pred_z)
        ]
        self.velocity_pub.publish(msg)
        
        # Publish ghost arrow in RViz
        self.publish_prediction_marker(hand_landmarks, pred_x, pred_y)
        
        self.get_logger().info(
            f"Ghost: speed={speed:.2f}, predicting ({pred_x:.2f}, {pred_y:.2f})"
        )

    # GHOST ARROW MARKER

    def publish_prediction_marker(self, hand_landmarks, pred_x, pred_y):
        """
        Publish where we THINK the user will point next.
        Semi-transparent blue arrow showing predicted direction.
        """
        ghost = Marker()
        ghost.header.frame_id = "camera_link"
        ghost.header.stamp = self.get_clock().now().to_msg()
        ghost.ns = "prediction"
        ghost.id = 100
        ghost.type = Marker.ARROW
        ghost.action = Marker.ADD
        
        tip = hand_landmarks.landmark[8]
        base = hand_landmarks.landmark[5]
        
        start = RosPoint(
            x=0.5,
            y=-(pred_x - 0.5),
            z=-(pred_y - 0.5)
        )
        
        dir_y = -(tip.x - base.x)
        dir_z = -(tip.y - base.y)
        
        end = RosPoint(
            x=float(start.x + 0.5),
            y=float(start.y + dir_y * 2),
            z=float(start.z + dir_z * 2)
        )
        
        ghost.points = [start, end]
        ghost.scale.x = 0.015
        ghost.scale.y = 0.03
        ghost.scale.z = 0.03
        
        ghost.color.r = 0.5
        ghost.color.g = 0.0
        ghost.color.b = 0.8
        ghost.color.a = 0.7
        
        ghost.lifetime.nanosec = 500000000
        
        self.marker_pub.publish(ghost)


    # LANDMARK PROCESSING

    def process_landmarks_relative(self, landmarks):
        """Convert for the BRAIN (Relative)"""
        flat_list = []
        for lm in landmarks.landmark:
            flat_list.extend([lm.x, lm.y, lm.z])
        data_array = np.array(flat_list).reshape(-1, 3)
        wrist = data_array[0, :]
        relative_data = data_array - wrist
        return relative_data.flatten().tolist()

    def get_raw_landmarks(self, landmarks):
        """Extract Raw Data for the RECORDER"""
        flat_list = []
        for lm in landmarks.landmark:
            flat_list.extend([lm.x, lm.y, lm.z])
        return flat_list


    # PRESENTATION OVERLAY
    
    def draw_presentation_overlay(self, frame, gesture_text, color, hand_landmarks, h, w):
        """Professional overlay for supervisor demo."""

        # TOP BAR
        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(frame, f"GESTURE: {gesture_text}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        icons = {
            'POINT': 'Pointing >>', 'GRASP': 'Grasping [ ]',
            'STOP': 'STOP [X]', 'OKAY': 'Confirm [OK]', 'NEUTRAL': 'Idle ...',
        }
        icon_text = icons.get(gesture_text, '')
        cv2.putText(frame, icon_text, (w - 250, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # ── MODE HUD banner: top-center, screen-space 2D overlay ──
        mode_colors = {
            'TORCH':    (0, 165, 255),   # orange (BGR)
            'FETCH':    (255, 200, 50),  # cyan-blue
            'FETCHING': (0, 255, 0),     # green
        }
        mode = self.current_robot_mode
        mc = mode_colors.get(mode, (255, 255, 255))
        banner = f"MODE: {mode}"
        (tw, th), _ = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        bx = (w - tw) // 2
        by = 110
        cv2.rectangle(frame, (bx - 15, by - th - 12),
                      (bx + tw + 15, by + 12), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx - 15, by - th - 12),
                      (bx + tw + 15, by + 12), mc, 2)
        cv2.putText(frame, banner, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, mc, 3, cv2.LINE_AA)
        
        # BOTTOM BAR
        state_color = {
            'POINT': (255, 100, 0), 'STOP': (0, 0, 255),
            'GRASP': (0, 165, 255), 'OKAY': (0, 255, 0),
            'NEUTRAL': (150, 150, 150),
        }
        bar_color = state_color.get(gesture_text, (150, 150, 150))
        cv2.rectangle(frame, (0, h - 50), (w, h), bar_color, -1)
        
        robot_action = {
            'POINT': 'ROBOT: Moving to target', 'STOP': 'ROBOT: FROZEN',
            'GRASP': 'ROBOT: Closing gripper', 'OKAY': 'ROBOT: Action confirmed',
            'NEUTRAL': 'ROBOT: Standing by',
        }
        action_text = robot_action.get(gesture_text, 'ROBOT: Waiting...')
        cv2.putText(frame, action_text, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FINGER TIP HIGHLIGHT
        if gesture_text == "POINT" and hand_landmarks:
            tip = hand_landmarks.landmark[8]
            px, py = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (px, py), 15, (0, 0, 255), 3)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(frame, "TARGET", (px + 20, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    
    # MAIN LOOP
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_text = "WAITING..."
        color = (200, 200, 200)
        hand_landmarks = None  # Right hand landmarks (used for overlay + gesture)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks_item, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness):

                # MediaPipe label: 'Left'/'Right' from its own perspective.
                # Since we cv2.flip() the frame, labels are already corrected to match the
                # real operator's hand (Right = operator's right hand).
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'

                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks_item, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # Draw hand label near wrist
                wrist_lm = hand_landmarks_item.landmark[0]
                wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
                label_color = (0, 200, 0) if hand_label == 'Right' else (200, 100, 0)
                cv2.putText(frame, hand_label, (wx, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Publish landmarks to hand-specific topic
                raw_data = self.get_raw_landmarks(hand_landmarks_item)
                hand_msg = Float32MultiArray()
                hand_msg.data = raw_data

                if hand_label == 'Left':
                    self.left_landmark_pub.publish(hand_msg)
                    continue  # no gesture classification for left hand

                # ── RIGHT HAND from here ──
                hand_landmarks = hand_landmarks_item  # used by overlay

                # Quality Gate
                wrist = hand_landmarks_item.landmark[0]
                if 0.02 < wrist.x < 0.98 and 0.02 < wrist.y < 0.98:

                    # ── Velocity Gate: freeze landmarks during gesture transitions
                    gate_open = self._update_velocity_gate(hand_landmarks_item)
                    if gate_open:
                        # Fingers stable: publish live landmarks
                        self.landmark_pub.publish(hand_msg)
                        self.right_landmark_pub.publish(hand_msg)
                        self._vg_frozen_landmarks = hand_msg
                    elif self._vg_frozen_landmarks is not None:
                        # Fingers in transit: re-publish last good position
                        self.landmark_pub.publish(self._vg_frozen_landmarks)
                        self.right_landmark_pub.publish(self._vg_frozen_landmarks)

                    # Ask the brain
                    if self.brain_ready:
                        relative_data = self.process_landmarks_relative(hand_landmarks_item)
                        try:
                            proba = self.brain.predict_proba([relative_data])[0]
                            prediction_id = int(proba.argmax())
                            confidence = proba[prediction_id]
                            if confidence < 0.70:
                                gesture_text = getattr(self, '_last_confident_gesture', 'NEUTRAL')
                            else:
                                gesture_text = self.gesture_names.get(prediction_id, "UNKNOWN")
                                self._last_confident_gesture = gesture_text
                            

                            # GEOMETRIC OVERRIDES: landmark-based
                            # safety net for all 5 gestures
                            
                            thumb_tip = hand_landmarks.landmark[4]
                            index_tip = hand_landmarks.landmark[8]
                            index_mcp = hand_landmarks.landmark[5]
                            middle_tip = hand_landmarks.landmark[12]
                            middle_mcp = hand_landmarks.landmark[9]
                            ring_tip = hand_landmarks.landmark[16]
                            ring_mcp = hand_landmarks.landmark[13]
                            pinky_tip = hand_landmarks.landmark[20]
                            wrist = hand_landmarks.landmark[0]

                            def _dist(a, b):
                                return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

                            idx_len = _dist(index_tip, index_mcp)
                            mid_len = _dist(middle_tip, middle_mcp)
                            rng_len = _dist(ring_tip, ring_mcp)
                            avg_other_len = (mid_len + rng_len) / 2.0

                            all_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
                            avg_tip_x = sum(t.x for t in all_tips) / 4.0
                            avg_tip_y = sum(t.y for t in all_tips) / 4.0
                            tip_spread = sum(_dist(t, type('P', (), {'x': avg_tip_x, 'y': avg_tip_y})()) for t in all_tips) / 4.0

                            thumb_index_dist = _dist(thumb_tip, index_tip)
                            idx_from_others = _dist(index_tip, type('P', (), {
                                'x': (middle_tip.x + ring_tip.x + pinky_tip.x) / 3.0,
                                'y': (middle_tip.y + ring_tip.y + pinky_tip.y) / 3.0})())
                            all_from_wrist = sum(_dist(t, wrist) for t in all_tips) / 4.0

                            #  STOP: all fingers extended far from wrist 
                            if gesture_text == "STOP":
                                if _dist(index_tip, wrist) > (_dist(middle_tip, wrist) * 1.3):
                                    gesture_text = "POINT"

                            #  GRASP: index extended but RF said GRASP -> POINT 
                            if gesture_text == "GRASP":
                                if avg_other_len > 0.001 and idx_len > (avg_other_len * 1.2):
                                    gesture_text = "POINT"

                            #  POINT: all tips clustered -> actually GRASP 
                            if gesture_text == "POINT":
                                if idx_from_others < 0.06:
                                    gesture_text = "GRASP"

                            #  SCALE-INVARIANT FINGER EXTENSION 
                            #    A finger is extended iff the tip is further from
                            #    the wrist than the PIP joint. This removes all
                            #    camera-distance dependence (MediaPipe landmarks
                            #    are normalised to image size, so absolute wrist-
                            #    tip distances shrink as the hand moves away).
                            thumb_ip   = hand_landmarks.landmark[3]
                            index_pip  = hand_landmarks.landmark[6]
                            middle_pip = hand_landmarks.landmark[10]
                            ring_pip   = hand_landmarks.landmark[14]
                            pinky_pip  = hand_landmarks.landmark[18]

                            thumb_extended  = _dist(thumb_tip,  wrist) > _dist(thumb_ip,   wrist)
                            index_extended  = _dist(index_tip,  wrist) > _dist(index_pip,  wrist)
                            middle_extended = _dist(middle_tip, wrist) > _dist(middle_pip, wrist)
                            ring_extended   = _dist(ring_tip,   wrist) > _dist(ring_pip,   wrist)
                            pinky_extended  = _dist(pinky_tip,  wrist) > _dist(pinky_pip,  wrist)

                            # Order matters: more-specific shapes first.
                            # 1. TOGGLE (shaka): absolute final authority.
                            if (thumb_extended and pinky_extended
                                    and not index_extended
                                    and not middle_extended
                                    and not ring_extended):
                                gesture_text = "TOGGLE"

                            # 2. OKAY (thumbs-up): only thumb extended.
                            elif (thumb_extended
                                    and not index_extended
                                    and not middle_extended
                                    and not ring_extended
                                    and not pinky_extended):
                                gesture_text = "OKAY"
                            elif gesture_text == "OKAY" and (index_extended or pinky_extended):
                                gesture_text = "NEUTRAL"

                            # 3. GRASP (fist): every finger curled.
                            elif (not thumb_extended
                                    and not index_extended
                                    and not middle_extended
                                    and not ring_extended
                                    and not pinky_extended):
                                gesture_text = "GRASP"
                            elif gesture_text == "GRASP" and (
                                    index_extended or middle_extended
                                    or ring_extended or pinky_extended):
                                gesture_text = "NEUTRAL"

                            # 4. STOP guard: cancel STOP unless the hand really
                            #    is opening (index + middle extended at minimum).
                            elif gesture_text == "STOP":
                                if not index_extended and not middle_extended:
                                    gesture_text = "GRASP"

                            # SMOOTH the gesture (prevent flickering)
                            gesture_text = self.get_stable_gesture(gesture_text)
                            self.last_stable_gesture = gesture_text

                            # Set colors based on smoothed gesture
                            if gesture_text == "GRASP": color = (0, 0, 255)
                            elif gesture_text == "POINT": color = (255, 0, 0)
                            elif gesture_text == "STOP": color = (0, 255, 255)
                            elif gesture_text == "OKAY": color = (0, 255, 0)
                            elif gesture_text == "NEUTRAL": color = (100, 100, 100)

                            # Now publish the SMOOTHED gesture
                            text_msg = String()
                            text_msg.data = gesture_text
                            self.gesture_pub.publish(text_msg)     

                            # Log timestamp
                            self.latency_writer.writerow([
                                'gesture_detected', gesture_text, time.time()
                            ])
                            self.latency_log.flush()

                        except Exception as e:
                            self.get_logger().warn(f"Classification failed: {e}")
                else:
                    gesture_text = "OUT OF BOUNDS"

        # Draw overlay
        frame = self.draw_presentation_overlay(frame, gesture_text, color, hand_landmarks, h, w)

        # Publish camera feed
        ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(ros_image)

        # POINT GESTURE: Arrow + Ghost Prediction
        if gesture_text == "POINT" and hand_landmarks:
            # 1. Publish the pointing arrow (your existing code)
            tip = hand_landmarks.landmark[8]
            base = hand_landmarks.landmark[5]

            start_x = 0.5
            start_y = -(tip.x - 0.5)
            start_z = -(tip.y - 0.5)

            dir_y = -(tip.x - base.x)
            dir_z = -(tip.y - base.y)
            dir_x = 0.5

            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            start = RosPoint(x=float(start_x), y=float(start_y), z=float(start_z))
            end = RosPoint(
                x=float(start_x + dir_x),
                y=float(start_y + dir_y * 2),
                z=float(start_z + dir_z * 2)
            )

            marker.points = [start, end]
            marker.scale.x = 0.02
            marker.scale.y = 0.04
            marker.scale.z = 0.04
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.marker_pub.publish(marker)

            # 2. Compute velocity and publish ghost prediction
            self.compute_and_publish_velocity(hand_landmarks)

    def __del__(self):
        self.cap.release()

    def destroy_node(self):
        f = getattr(self, 'latency_log', None)
        if f is not None:
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
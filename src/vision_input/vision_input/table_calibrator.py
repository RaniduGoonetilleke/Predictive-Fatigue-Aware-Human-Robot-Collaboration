#!/usr/bin/env python3
"""
table_calibrator — 4-point homography calibration for overhead tripod camera

Run ONCE after positioning the tripod. Click 4 table corners in the live
webcam view; the script computes the image -> robot homography and saves it to
``~/ros2_ws/data/table_calibration.yaml``. On next launch, ``robot_controller``
and ``hand_mirror`` load the YAML and use ``cv2.perspectiveTransform`` to map
fingertip pixel coords directly into base_link X/Y, correcting the tripod's
perspective warp (non-orthogonal "security-camera" angle) that the linear
``map_scale_x/y`` can only approximate.

Usage:
    ros2 run vision_input table_calibrator \\
        --ros-args -p camera_id:=3

    # Override the 4 robot-side anchor points (metres, base_link frame).
    # Order: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT
    # (clockwise starting top-left as you face the camera view).
    ros2 run vision_input table_calibrator --ros-args \\
        -p corner_far_left:='[0.50, 0.25]' \\
        -p corner_far_right:='[0.50, -0.25]' \\
        -p corner_near_right:='[0.17, -0.25]' \\
        -p corner_near_left:='[0.17, 0.25]'

Keys:
    C — enter calibration mode, click 4 corners in order
    R — reload the saved YAML (verify load path)
    S — save current calibration to YAML
    Q — quit

RViz topic: /table_calibrator/corners (MarkerArray) — coloured spheres at
each anchor point so you can sanity-check the physical layout.
"""

import os
import math
import yaml

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


CALIB_PATH = os.path.join(
    os.path.expanduser('~'), 'ros2_ws', 'data', 'table_calibration.yaml')

# Click order, matching the 4 ROS params below.
# Uses the physical objects already in the scene (cubes + hazard zone) as
# anchors so the operator doesn't need tape marks. The Blue Cube must be
# temporarily moved to (0.17, 0.38) during calibration, otherwise all 3
# cubes are collinear at X=0.43 and cv2.findHomography returns a singular
# matrix. Put the Blue Cube back after saving.
CORNER_NAMES  = [
    'GREEN CUBE',
    'BLUE CUBE (moved)',
    'HAZARD ZONE',
    'RED CUBE',
    'ROBOT BASE',
]
CORNER_COLORS = [
    (0, 255,   0),   # BGR — green cube
    (255, 100, 0),   # blue cube (BGR blue-ish)
    (0, 0,   255),   # hazard zone (red)
    (0, 0,   200),   # red cube (darker red)
    (0, 255, 255),   # robot base (yellow)
]
RVIZ_COLORS   = [
    (0.0, 1.0, 0.0),   # green cube
    (0.0, 0.2, 1.0),   # blue cube
    (1.0, 0.0, 0.0),   # hazard zone
    (0.8, 0.0, 0.0),   # red cube
    (1.0, 1.0, 0.0),   # robot base
]
NUM_CORNERS = len(CORNER_NAMES)


class TableCalibrator(Node):

    def __init__(self):
        super().__init__('table_calibrator')

        # Camera
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)

        # 4 robot-space anchor points (metres, base_link frame).
        # Defaults use physical objects already in the scene so no tape
        # marks are needed. The Blue Cube must be temporarily moved to
        # (0.17, 0.38) before calibration: the three cubes are otherwise
        # collinear at X=0.43 which makes cv2.findHomography fail with a
        # singular matrix. Put the Blue Cube back after saving.
        self.declare_parameter('corner_green_cube',       [0.43,  0.38])
        self.declare_parameter('corner_blue_cube_moved',  [0.17,  0.38])
        self.declare_parameter('corner_hazard_zone',      [-0.288, 0.103])
        self.declare_parameter('corner_red_cube',         [0.43,  0.04])
        self.declare_parameter('corner_robot_base',       [0.0,   0.0])

        self.cam_id = self.get_parameter('camera_id').value
        self.W = self.get_parameter('frame_width').value
        self.H = self.get_parameter('frame_height').value

        self.robot_corners = np.array([
            self.get_parameter('corner_green_cube').value,
            self.get_parameter('corner_blue_cube_moved').value,
            self.get_parameter('corner_hazard_zone').value,
            self.get_parameter('corner_red_cube').value,
            self.get_parameter('corner_robot_base').value,
        ], dtype=np.float32)

        # Camera open
        self.get_logger().info(f"Opening camera_id={self.cam_id} @ {self.W}x{self.H}")
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Cannot open camera {self.cam_id}. "
                "Try a different camera_id param (e.g. 2, 3).")
            raise SystemExit(1)
        # MJPEG unlocks higher framerates on USB webcams.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Actual capture: {self.W}x{self.H}")

        # State
        # Seed with a sensible trapezoid so something is visible before
        # the first click-calibration.
        self.image_corners = np.array([
            [int(self.W * 0.30), int(self.H * 0.30)],   # green cube
            [int(self.W * 0.70), int(self.H * 0.30)],   # blue cube (moved)
            [int(self.W * 0.85), int(self.H * 0.80)],   # hazard zone
            [int(self.W * 0.15), int(self.H * 0.80)],   # red cube
            [int(self.W * 0.50), int(self.H * 0.50)],   # robot base
        ], dtype=np.float32)
        self.homography = None
        self.calibrating = False
        self.clicked = []
        self._compute_homography()
        self._try_load_yaml()  # pull in previous calibration if present

        # RViz corner markers (5 Hz)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/table_calibrator/corners', 10)
        self.create_timer(0.2, self._publish_corner_markers)

        # Main loop (30 Hz)
        self.create_timer(1.0 / 30.0, self._tick)

        # Mouse click handler
        cv2.namedWindow('Table Calibrator')
        cv2.setMouseCallback('Table Calibrator', self._on_mouse)

        self._print_help()

    # Homography

    def _compute_homography(self):
        try:
            H, _ = cv2.findHomography(
                self.image_corners.astype(np.float32),
                self.robot_corners.astype(np.float32))
            self.homography = H
        except Exception as e:
            self.get_logger().warn(f"Homography compute failed: {e}")
            self.homography = None

    def _image_to_robot(self, px, py):
        if self.homography is None:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    # YAML persistence

    def _save_yaml(self):
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        data = {
            'image_size': [int(self.W), int(self.H)],
            'image_corners':  self.image_corners.tolist(),
            'robot_corners':  self.robot_corners.tolist(),
            'homography':     self.homography.tolist() if self.homography is not None else None,
            'corner_order': CORNER_NAMES,
        }
        with open(CALIB_PATH, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        self.get_logger().info(f"Saved calibration → {CALIB_PATH}")

    def _try_load_yaml(self):
        if not os.path.exists(CALIB_PATH):
            self.get_logger().info(
                f"No existing calibration at {CALIB_PATH} — using defaults")
            return
        try:
            with open(CALIB_PATH, 'r') as f:
                data = yaml.safe_load(f)
            loaded_img = np.array(data['image_corners'], dtype=np.float32)
            loaded_rob = np.array(data['robot_corners'], dtype=np.float32)
            if loaded_img.shape[0] != NUM_CORNERS or loaded_rob.shape[0] != NUM_CORNERS:
                self.get_logger().warn(
                    f"Ignoring stale calibration ({loaded_img.shape[0]} corners, "
                    f"expected {NUM_CORNERS}) — delete {CALIB_PATH} and re-calibrate.")
                return
            self.image_corners = loaded_img
            self.robot_corners = loaded_rob
            self._compute_homography()
            self.get_logger().info(f"Loaded calibration from {CALIB_PATH}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load {CALIB_PATH}: {e}")

    # Calibration interaction

    def _on_mouse(self, event, x, y, flags, param):
        if not self.calibrating:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked) < NUM_CORNERS:
            self.clicked.append([x, y])
            name = CORNER_NAMES[len(self.clicked) - 1]
            self.get_logger().info(
                f"  Corner {len(self.clicked)}/{NUM_CORNERS} [{name}] @ ({x}, {y})")
            if len(self.clicked) == NUM_CORNERS:
                self.image_corners = np.array(self.clicked, dtype=np.float32)
                self._compute_homography()
                self._save_yaml()
                self.calibrating = False
                self.clicked = []
                self.get_logger().info("Calibration complete — saved.")

    def _start_calibration(self):
        self.calibrating = True
        self.clicked = []
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f" CALIBRATION — click these {NUM_CORNERS} points in order")
        self.get_logger().info(" ⚠  Make sure the BLUE CUBE is at (0.17, 0.38) first!")
        self.get_logger().info(" ⚠  (All 3 cubes in their default row are collinear →")
        self.get_logger().info("     homography would fail with a singular matrix.)")
        self.get_logger().info("")
        for i, (name, rxy) in enumerate(zip(CORNER_NAMES, self.robot_corners), 1):
            self.get_logger().info(
                f"   {i}. {name:<18s} → robot ({rxy[0]:+.2f}, {rxy[1]:+.2f}) m")
        self.get_logger().info("=" * 60)

    def _print_help(self):
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info(" TABLE CALIBRATOR — object-based homography calibration")
        self.get_logger().info("=" * 60)
        self.get_logger().info(" Before pressing C:")
        self.get_logger().info("   1. Move the BLUE CUBE from (0.43, 0.21) → (0.17, 0.38)")
        self.get_logger().info("      (otherwise all 3 cubes are collinear)")
        self.get_logger().info("   2. Check RViz — the 4 coloured spheres at /table_calibrator/corners")
        self.get_logger().info("      should match where these objects physically sit.")
        self.get_logger().info("")
        self.get_logger().info(" Anchor points (base_link metres):")
        for name, rxy in zip(CORNER_NAMES, self.robot_corners):
            self.get_logger().info(f"   • {name:<18s} → ({rxy[0]:+.2f}, {rxy[1]:+.2f})")
        self.get_logger().info("")
        self.get_logger().info(" Controls:  [C] calibrate   [S] save   [R] reload   [Q] quit")
        self.get_logger().info(" After saving, move the BLUE CUBE back to (0.43, 0.21).")
        self.get_logger().info("=" * 60)

    # Draw helpers

    def _draw_overlay(self, frame):
        # Table quadrilateral
        pts = self.image_corners.astype(np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (80, 180, 80))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 200, 120), 2)

        # Corner badges
        for i, (p, name, col) in enumerate(
                zip(pts, CORNER_NAMES, CORNER_COLORS)):
            cv2.circle(frame, tuple(p), 14, col, -1)
            cv2.circle(frame, tuple(p), 16, (255, 255, 255), 2)
            cv2.putText(frame, str(i + 1), (p[0] - 6, p[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            label_pos = (p[0] + 18, p[1] + 6)
            cv2.putText(frame, name, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # Info panel
        cv2.rectangle(frame, (10, 10), (360, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (360, 180), (255, 255, 255), 1)
        mode = "CALIBRATING" if self.calibrating else "LIVE PREVIEW"
        cv2.putText(frame, f"TABLE CALIBRATOR — {mode}", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        if self.calibrating:
            n = len(self.clicked)
            if n < NUM_CORNERS:
                nxt = CORNER_NAMES[n]
                col = CORNER_COLORS[n]
                cv2.putText(frame, f"Click {n + 1}/{NUM_CORNERS}:", (18, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, nxt, (110, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)
                rxy = self.robot_corners[n]
                cv2.putText(frame,
                            f"robot ({rxy[0]:+.2f}, {rxy[1]:+.2f}) m",
                            (18, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            for i, c in enumerate(self.clicked):
                cv2.circle(frame, tuple(c), 12, CORNER_COLORS[i], -1)
                cv2.putText(frame, str(i + 1), (c[0] - 5, c[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Move BLUE CUBE to (0.17, 0.38) first!",
                        (18, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
            cv2.putText(frame, "Then press [C] and click in order:",
                        (18, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            for i, name in enumerate(CORNER_NAMES):
                cv2.putText(frame, f"  {i + 1}. {name}", (18, 100 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            CORNER_COLORS[i], 1)
            hy = 100 + len(CORNER_NAMES) * 16 + 8
            if self.homography is not None:
                cv2.putText(frame, "Homography: OK", (18, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Homography: NOT SET", (18, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    def _draw_preview_crosshair(self, frame):
        # Mouse-less preview: show robot XY at the image centre so the
        # user can eyeball scale without a pointing gesture.
        cx, cy = self.W // 2, self.H // 2
        rxy = self._image_to_robot(cx, cy)
        cv2.drawMarker(frame, (cx, cy), (255, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        if rxy is not None:
            txt = f"centre → robot ({rxy[0]:+.2f}, {rxy[1]:+.2f}) m"
            cv2.putText(frame, txt, (cx - 170, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Main tick

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)  # mirror — matches camera_node convention
        self._draw_overlay(frame)

        cv2.imshow('Table Calibrator', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key in (ord('c'), ord('C')) and not self.calibrating:
            self._start_calibration()
        elif key in (ord('s'), ord('S')):
            self._save_yaml()
        elif key in (ord('r'), ord('R')):
            self._try_load_yaml()

    # RViz corner markers

    def _publish_corner_markers(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, (rxy, name, col) in enumerate(
                zip(self.robot_corners, CORNER_NAMES, RVIZ_COLORS)):
            sphere = Marker()
            sphere.header.frame_id = 'base_link'
            sphere.header.stamp = now
            sphere.ns = 'table_calib_spheres'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(rxy[0])
            sphere.pose.position.y = float(rxy[1])
            sphere.pose.position.z = 0.02
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05
            sphere.color.r, sphere.color.g, sphere.color.b = col
            sphere.color.a = 1.0
            sphere.lifetime.sec = 1
            ma.markers.append(sphere)

            label = Marker()
            label.header.frame_id = 'base_link'
            label.header.stamp = now
            label.ns = 'table_calib_labels'
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(rxy[0])
            label.pose.position.y = float(rxy[1])
            label.pose.position.z = 0.10
            label.pose.orientation.w = 1.0
            label.text = f"{i + 1} {name}"
            label.scale.z = 0.04
            label.color.r, label.color.g, label.color.b = col
            label.color.a = 1.0
            label.lifetime.sec = 1
            ma.markers.append(label)

        # Table outline (closed line strip)
        outline = Marker()
        outline.header.frame_id = 'base_link'
        outline.header.stamp = now
        outline.ns = 'table_calib_outline'
        outline.id = 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.008
        outline.color.r, outline.color.g, outline.color.b = 0.0, 0.8, 0.4
        outline.color.a = 1.0
        outline.lifetime.sec = 1
        for rxy in list(self.robot_corners) + [self.robot_corners[0]]:
            p = Point()
            p.x, p.y, p.z = float(rxy[0]), float(rxy[1]), 0.001
            outline.points.append(p)
        ma.markers.append(outline)

        self.marker_pub.publish(ma)

    # Cleanup

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TableCalibrator()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

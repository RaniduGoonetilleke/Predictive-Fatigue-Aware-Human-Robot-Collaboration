#!/usr/bin/env python3
"""
AR Pointing Interface - Simplified with Corner Markers

Shows a virtual table with numbered corners.
The same corners appear in RViz so you know the exact mapping.
Point at the table → Robot moves to that spot.

Author: Ranidu P. Goonetilleke
Date: February 2025
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
import mediapipe as mp


class ARPointingInterface(Node):
    
    def __init__(self):
        super().__init__('ar_pointing_interface')
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("   AR POINTING INTERFACE")
        self.get_logger().info("   With Matching Corner Markers")
        self.get_logger().info("=" * 50)
        
        # CAMERA SETUP 
        self.get_logger().info("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        
        # MEDIAPIPE SETUP
        self.get_logger().info("Setting up hand tracking...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # VIRTUAL TABLE SETUP
        self.setup_default_table()
        
        # Table corners in ROBOT coordinates (meters)
        # These MUST match the markers published to RViz!
        self.table_corners_robot = np.array([
            [0.15, 0.20, 0.0],    # Corner 1: B-R
            [0.15, -0.20, 0.0],   # Corner 2: B-L
            [0.40, -0.20, 0.0],   # Corner 3: F-L
            [0.40, 0.20, 0.0],    # Corner 4: F-R
        ], dtype=np.float32)
        
        # Corner labels
        self.corner_labels = ['1', '2', '3', '4']
        self.corner_names = ['B-R', 'B-L', 'F-L', 'F-R']
        
        # Compute homography
        self.compute_homography()
        
        # PUBLISHERS
        self.target_pub = self.create_publisher(
            PointStamped, 
            '/ar_target_point', 
            10
        )
        
        self.marker_pub = self.create_publisher(
            Marker,
            '/ar_visualization',
            10
        )
        
        self.corner_marker_pub = self.create_publisher(
            MarkerArray,
            '/ar_corner_markers',
            10
        )
        
        # STATE
        self.current_target_2d = None
        self.current_target_robot = None
        self.pointing_active = False
        
        # COLORS
        # Table colors
        self.COLOR_TABLE_FILL = (200, 255, 200)       # Light green fill
        self.COLOR_TABLE_ACTIVE = (150, 255, 150)     # Brighter when pointing
        self.COLOR_TABLE_BORDER = (0, 200, 100)       # Green border
        self.COLOR_GRID = (100, 180, 100)             # Grid lines
        
        # Corner colors (matching RViz)
        self.CORNER_COLORS = [
            (0, 0, 255),      # Corner 1: Red
            (0, 165, 255),    # Corner 2: Orange
            (0, 255, 255),    # Corner 3: Yellow
            (0, 255, 0),      # Corner 4: Green
        ]
        
        # Target and feedback
        self.COLOR_TARGET = (255, 0, 255)             # Magenta target
        self.COLOR_RAY = (255, 255, 0)                # Cyan ray
        self.COLOR_TEXT = (255, 255, 255)             # White text
        
        # MAIN LOOP
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 FPS
        self.corner_timer = self.create_timer(0.2, self.publish_corner_markers)  # 5 Hz
        
        self.get_logger().info("")
        self.get_logger().info("✅ AR Interface Ready!")
        self.get_logger().info("")
        self.get_logger().info("CORNER MAPPING:")
        for i, (label, name) in enumerate(zip(self.corner_labels, self.corner_names)):
            pos = self.table_corners_robot[i]
            self.get_logger().info(f"  Corner {label} ({name}): Robot ({pos[0]:.2f}, {pos[1]:.2f})")
        self.get_logger().info("")
        self.get_logger().info("CONTROLS:")
        self.get_logger().info("  • Point at the table with your INDEX FINGER")
        self.get_logger().info("  • Press 'C' to calibrate table corners")
        self.get_logger().info("  • Press 'R' to reset to defaults")
        self.get_logger().info("  • Press 'Q' to quit")
        self.get_logger().info("")
    
    def setup_default_table(self):
        """Set up default table corners based on camera resolution."""
        
        w = self.frame_width
        h = self.frame_height
        
        # Trapezoid shape representing perspective view of table
        # Bottom is closer (wider), top is farther (narrower)
        self.table_corners_2d = np.array([
            [int(w * 0.20), int(h * 0.80)],   # Corner 1: Bottom-left
            [int(w * 0.80), int(h * 0.80)],   # Corner 2: Bottom-right
            [int(w * 0.70), int(h * 0.40)],   # Corner 3: Top-right
            [int(w * 0.30), int(h * 0.40)],   # Corner 4: Top-left
        ], dtype=np.float32)
        
        self.get_logger().info("Default table corners set")
    
    def compute_homography(self):
        """Compute transformation from image to robot coordinates."""
        
        robot_xy = self.table_corners_robot[:, :2]
        
        self.homography, _ = cv2.findHomography(
            self.table_corners_2d, 
            robot_xy
        )
        
        self.get_logger().info("Homography matrix computed")
    
    def image_to_robot(self, image_point):
        """Convert 2D image point to 3D robot coordinates."""
        
        if self.homography is None:
            return None
        
        point = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
        point = np.array([point])
        
        transformed = cv2.perspectiveTransform(point, self.homography)
        
        robot_x = float(transformed[0][0][0])
        robot_y = float(transformed[0][0][1])
        robot_z = 0.0  # Table surface
        
        return np.array([robot_x, robot_y, robot_z])
    
    def point_in_table(self, point):
        """Check if a 2D point is inside the virtual table."""
        
        polygon = self.table_corners_2d.astype(np.int32)
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
        return result >= 0
    
    def is_pointing_gesture(self, hand_landmarks):
        """Detect pointing gesture (index finger extended)."""
        
        lm = hand_landmarks.landmark
        
        # Index finger extended
        index_extended = lm[8].y < lm[6].y
        
        # At least one other finger folded
        middle_folded = lm[12].y > lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y
        
        return index_extended and (middle_folded or ring_folded or pinky_folded)
    
    def get_pointing_intersection(self, hand_landmarks, frame_shape):
        """Calculate where pointing ray intersects the table."""
        
        h, w = frame_shape[:2]
        lm = hand_landmarks.landmark
        
        # Finger base and tip
        base = np.array([lm[5].x * w, lm[5].y * h])
        tip = np.array([lm[8].x * w, lm[8].y * h])
        
        # Direction
        direction = tip - base
        length = np.linalg.norm(direction)
        
        if length < 20:
            return None
        
        direction = direction / length
        
        # Extend ray to find table intersection
        for t in range(0, 800, 5):
            test_point = tip + direction * t
            
            if not (0 <= test_point[0] < w and 0 <= test_point[1] < h):
                break
            
            if self.point_in_table(test_point):
                return test_point.astype(int)
        
        return None
    
    def draw_virtual_table(self, frame, active=False):
        """Draw the virtual table with corner markers."""
        
        corners = self.table_corners_2d.astype(np.int32)
        
        # Fill color
        fill_color = self.COLOR_TABLE_ACTIVE if active else self.COLOR_TABLE_FILL
        
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [corners], fill_color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Border
        cv2.polylines(frame, [corners], True, self.COLOR_TABLE_BORDER, 3)
        
        # Grid lines
        self.draw_grid(frame, corners)
        
        # Draw corner markers with numbers
        self.draw_corner_markers(frame, corners)
        
        # Center label
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(frame, "ROBOT WORKSPACE", (center[0] - 90, center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)
        cv2.putText(frame, "(Point here to control robot)", (center[0] - 110, center[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_grid(self, frame, corners):
        """Draw perspective grid on the table."""
        
        num_lines = 4
        
        for i in range(1, num_lines):
            t = i / num_lines
            
            # Horizontal lines
            left = (corners[0] * (1 - t) + corners[3] * t).astype(int)
            right = (corners[1] * (1 - t) + corners[2] * t).astype(int)
            cv2.line(frame, tuple(left), tuple(right), self.COLOR_GRID, 1)
            
            # Vertical lines
            bottom = (corners[0] * (1 - t) + corners[1] * t).astype(int)
            top = (corners[3] * (1 - t) + corners[2] * t).astype(int)
            cv2.line(frame, tuple(bottom), tuple(top), self.COLOR_GRID, 1)
    
    def draw_corner_markers(self, frame, corners):
        """Draw numbered corner markers that match RViz."""
        
        for i, (corner, label, name) in enumerate(zip(corners, self.corner_labels, self.corner_names)):
            color = self.CORNER_COLORS[i]
            x, y = corner
            
            # Large filled circle
            cv2.circle(frame, (x, y), 20, color, -1)
            cv2.circle(frame, (x, y), 22, (255, 255, 255), 2)
            
            # Number label (centered in circle)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            cv2.putText(frame, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Corner name label (offset from circle)
            # Position label based on corner location
            if i == 0:  # Bottom-left
                label_pos = (x - 80, y + 35)
            elif i == 1:  # Bottom-right
                label_pos = (x + 25, y + 35)
            elif i == 2:  # Top-right
                label_pos = (x + 25, y - 10)
            else:  # Top-left
                label_pos = (x - 80, y - 10)
            
            cv2.putText(frame, name, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_pointing_feedback(self, frame, hand_landmarks, target_2d):
        """Draw pointing ray and target."""
        
        h, w = frame.shape[:2]
        lm = hand_landmarks.landmark
        
        # Fingertip position
        tip = (int(lm[8].x * w), int(lm[8].y * h))
        
        if target_2d is not None:
            # Ray from finger to target
            cv2.line(frame, tip, tuple(target_2d), self.COLOR_RAY, 2)
            
            tx, ty = target_2d
            
            # Target crosshair
            size = 15
            cv2.circle(frame, (tx, ty), size, self.COLOR_TARGET, 2)
            cv2.circle(frame, (tx, ty), 5, self.COLOR_TARGET, -1)
            
            # Crosshair lines
            cv2.line(frame, (tx - size - 8, ty), (tx - size + 5, ty), self.COLOR_TARGET, 2)
            cv2.line(frame, (tx + size - 5, ty), (tx + size + 8, ty), self.COLOR_TARGET, 2)
            cv2.line(frame, (tx, ty - size - 8), (tx, ty - size + 5), self.COLOR_TARGET, 2)
            cv2.line(frame, (tx, ty + size - 5), (tx, ty + size + 8), self.COLOR_TARGET, 2)
    
    def draw_info_panel(self, frame):
        """Draw information overlay."""
        
        # Background
        cv2.rectangle(frame, (10, 10), (280, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, 130), self.COLOR_TEXT, 2)
        
        # Title
        cv2.putText(frame, "AR POINTING INTERFACE", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        # Status
        if self.pointing_active:
            status = "POINTING"
            color = (0, 255, 0)
        else:
            status = "Point at table..."
            color = (150, 150, 150)
        
        cv2.putText(frame, f"Status: {status}", (20, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Coordinates
        if self.current_target_robot is not None:
            x, y, z = self.current_target_robot
            cv2.putText(frame, f"Robot X: {x:+.3f} m", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
            cv2.putText(frame, f"Robot Y: {y:+.3f} m", (20, 98),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
            cv2.putText(frame, f"Robot Z: {z:+.3f} m", (20, 116),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
        else:
            cv2.putText(frame, "Coordinates: ---", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    def draw_legend(self, frame):
        """Draw corner color legend."""
        
        h = frame.shape[0]
        start_y = h - 100
        
        # Background
        cv2.rectangle(frame, (10, start_y), (180, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, start_y), (180, h - 10), self.COLOR_TEXT, 1)
        
        cv2.putText(frame, "CORNER LEGEND:", (20, start_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for i, (label, name, color) in enumerate(zip(self.corner_labels, self.corner_names, self.CORNER_COLORS)):
            y = start_y + 35 + i * 15
            cv2.circle(frame, (25, y - 4), 6, color, -1)
            cv2.putText(frame, f"{label}: {name}", (38, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def draw_instructions(self, frame):
        """Draw controls at bottom."""
        
        h = frame.shape[0]
        cv2.putText(frame, "[C] Calibrate Table  [R] Reset  [Q] Quit", (200, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT, 1)
    
    def process_frame(self):
        """Main processing loop."""
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Reset state
        self.pointing_active = False
        self.current_target_2d = None
        self.current_target_robot = None
        
        # Process hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw hand
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
                # Check pointing
                if self.is_pointing_gesture(hand_landmarks):
                    target_2d = self.get_pointing_intersection(hand_landmarks, frame.shape)
                    
                    if target_2d is not None:
                        self.pointing_active = True
                        self.current_target_2d = target_2d
                        self.current_target_robot = self.image_to_robot(target_2d)
                        
                        self.draw_pointing_feedback(frame, hand_landmarks, target_2d)
                        self.publish_target()
        
        # Draw overlays
        self.draw_virtual_table(frame, active=self.pointing_active)
        self.draw_info_panel(frame)
        self.draw_legend(frame)
        self.draw_instructions(frame)
        
        # Show
        cv2.imshow('AR Pointing Interface', frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        self.handle_key(key)
    
    def publish_target(self):
        """Publish target to robot controller."""
        
        if self.current_target_robot is None:
            return
        
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x = float(self.current_target_robot[0])
        msg.point.y = float(self.current_target_robot[1])
        msg.point.z = float(self.current_target_robot[2])
        
        self.target_pub.publish(msg)
        
        # Target marker in RViz
        self.publish_target_marker()
    
    def publish_target_marker(self):
        """Publish target visualization for RViz."""
        
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ar_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.current_target_robot[0])
        marker.pose.position.y = float(self.current_target_robot[1])
        marker.pose.position.z = float(self.current_target_robot[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        
        # Magenta (matches camera)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 150000000
        
        self.marker_pub.publish(marker)
    
    def publish_corner_markers(self):
        """Publish corner markers to RViz that match the camera view."""
        
        marker_array = MarkerArray()
        
        # Corner colors in RViz (RGB normalized to 0-1)
        rviz_colors = [
            (1.0, 0.0, 0.0),    # Corner 1: Red
            (1.0, 0.5, 0.0),    # Corner 2: Orange
            (1.0, 1.0, 0.0),    # Corner 3: Yellow
            (0.0, 1.0, 0.0),    # Corner 4: Green
        ]
        
        now = self.get_clock().now().to_msg()
        
        for i, (corner, label, color) in enumerate(zip(self.table_corners_robot, self.corner_labels, rviz_colors)):
            
            # Sphere marker at corner
            sphere = Marker()
            sphere.header.frame_id = "base_link"
            sphere.header.stamp = now
            sphere.ns = "corner_spheres"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            
            sphere.pose.position.x = float(corner[0])
            sphere.pose.position.y = float(corner[1])
            sphere.pose.position.z = float(corner[2])
            sphere.pose.orientation.w = 1.0
            
            sphere.scale.x = 0.05
            sphere.scale.y = 0.05
            sphere.scale.z = 0.05
            
            sphere.color.r = color[0]
            sphere.color.g = color[1]
            sphere.color.b = color[2]
            sphere.color.a = 1.0
            
            sphere.lifetime.sec = 1
            
            marker_array.markers.append(sphere)
            
            # Text label above corner
            text = Marker()
            text.header.frame_id = "base_link"
            text.header.stamp = now
            text.ns = "corner_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            text.pose.position.x = float(corner[0])
            text.pose.position.y = float(corner[1])
            text.pose.position.z = 0.08  # Above the sphere
            text.pose.orientation.w = 1.0
            
            text.text = label
            text.scale.z = 0.06  # Text height
            
            text.color.r = color[0]
            text.color.g = color[1]
            text.color.b = color[2]
            text.color.a = 1.0
            
            text.lifetime.sec = 1
            
            marker_array.markers.append(text)
            
            # Cylinder pole under number (makes it easier to see)
            pole = Marker()
            pole.header.frame_id = "base_link"
            pole.header.stamp = now
            pole.ns = "corner_poles"
            pole.id = i
            pole.type = Marker.CYLINDER
            pole.action = Marker.ADD
            
            pole.pose.position.x = float(corner[0])
            pole.pose.position.y = float(corner[1])
            pole.pose.position.z = 0.03
            pole.pose.orientation.w = 1.0
            
            pole.scale.x = 0.01
            pole.scale.y = 0.01
            pole.scale.z = 0.06
            
            pole.color.r = color[0]
            pole.color.g = color[1]
            pole.color.b = color[2]
            pole.color.a = 0.8
            
            pole.lifetime.sec = 1
            
            marker_array.markers.append(pole)
        
        # Table surface outline
        table_outline = Marker()
        table_outline.header.frame_id = "base_link"
        table_outline.header.stamp = now
        table_outline.ns = "table_outline"
        table_outline.id = 0
        table_outline.type = Marker.LINE_STRIP
        table_outline.action = Marker.ADD
        
        # Add corners + first corner again to close the loop
        from geometry_msgs.msg import Point
        for corner in self.table_corners_robot:
            p = Point()
            p.x = float(corner[0])
            p.y = float(corner[1])
            p.z = 0.001
            table_outline.points.append(p)
        
        # Close the loop
        p = Point()
        p.x = float(self.table_corners_robot[0][0])
        p.y = float(self.table_corners_robot[0][1])
        p.z = 0.001
        table_outline.points.append(p)
        
        table_outline.scale.x = 0.01  # Line width
        
        table_outline.color.r = 0.0
        table_outline.color.g = 0.8
        table_outline.color.b = 0.4
        table_outline.color.a = 1.0
        
        table_outline.lifetime.sec = 1
        
        marker_array.markers.append(table_outline)
        
        # Table surface (filled)
        table_surface = Marker()
        table_surface.header.frame_id = "base_link"
        table_surface.header.stamp = now
        table_surface.ns = "table_surface"
        table_surface.id = 0
        table_surface.type = Marker.CUBE
        table_surface.action = Marker.ADD
        
        # Calculate center and size of table
        center_x = np.mean(self.table_corners_robot[:, 0])
        center_y = np.mean(self.table_corners_robot[:, 1])
        
        size_x = np.max(self.table_corners_robot[:, 0]) - np.min(self.table_corners_robot[:, 0])
        size_y = np.max(self.table_corners_robot[:, 1]) - np.min(self.table_corners_robot[:, 1])
        
        table_surface.pose.position.x = float(center_x)
        table_surface.pose.position.y = float(center_y)
        table_surface.pose.position.z = -0.005
        table_surface.pose.orientation.w = 1.0
        
        table_surface.scale.x = float(size_x)
        table_surface.scale.y = float(size_y)
        table_surface.scale.z = 0.01
        
        table_surface.color.r = 0.2
        table_surface.color.g = 0.6
        table_surface.color.b = 0.3
        table_surface.color.a = 0.5
        
        table_surface.lifetime.sec = 1
        
        marker_array.markers.append(table_surface)
        
        self.corner_marker_pub.publish(marker_array)
    
    def handle_key(self, key):
        """Handle keyboard input."""
        
        if key == ord('q') or key == ord('Q'):
            self.get_logger().info("Quitting...")
            self.cleanup()
            rclpy.shutdown()
        
        elif key == ord('c') or key == ord('C'):
            self.calibrate_table()
        
        elif key == ord('r') or key == ord('R'):
            self.get_logger().info("Resetting to defaults...")
            self.setup_default_table()
            self.compute_homography()
    
    def calibrate_table(self):
        """Interactive calibration - click 4 corners."""
        
        self.get_logger().info("")
        self.get_logger().info("=" * 40)
        self.get_logger().info("TABLE CALIBRATION")
        self.get_logger().info("=" * 40)
        self.get_logger().info("Click 4 corners in order:")
        self.get_logger().info("  1. Front-Left (RED)")
        self.get_logger().info("  2. Front-Right (ORANGE)")
        self.get_logger().info("  3. Back-Right (YELLOW)")
        self.get_logger().info("  4. Back-Left (GREEN)")
        self.get_logger().info("")
        
        corners = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                self.get_logger().info(f"Corner {len(corners)}: ({x}, {y})")
        
        cv2.setMouseCallback('AR Pointing Interface', mouse_callback)
        
        while len(corners) < 4:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Instructions
            cv2.rectangle(frame, (10, 10), (420, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"CALIBRATION: Click corner {len(corners)+1}/4", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if len(corners) < 4:
                color = self.CORNER_COLORS[len(corners)]
                name = self.corner_names[len(corners)]
                # Convert BGR to display
                cv2.putText(frame, f"Next: {name}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame, "Press ESC to cancel", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Draw clicked corners
            for i, corner in enumerate(corners):
                color = self.CORNER_COLORS[i]
                cv2.circle(frame, tuple(corner), 15, color, -1)
                cv2.circle(frame, tuple(corner), 17, (255, 255, 255), 2)
                cv2.putText(frame, str(i+1), (corner[0]-7, corner[1]+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Lines between corners
            if len(corners) >= 2:
                for i in range(len(corners) - 1):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[i+1]), (0, 255, 0), 2)
                if len(corners) == 4:
                    cv2.line(frame, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
            
            cv2.imshow('AR Pointing Interface', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.get_logger().info("Calibration cancelled")
                cv2.setMouseCallback('AR Pointing Interface', lambda *args: None)
                return
        
        # Apply new corners
        self.table_corners_2d = np.array(corners, dtype=np.float32)
        self.compute_homography()
        
        self.get_logger().info("✅ Table calibration complete!")
        cv2.setMouseCallback('AR Pointing Interface', lambda *args: None)
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def destroy_node(self):
        self.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = ARPointingInterface()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
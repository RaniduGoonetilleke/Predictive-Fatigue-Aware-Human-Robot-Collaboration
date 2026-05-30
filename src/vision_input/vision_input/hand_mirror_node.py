"""
hand_mirror_node  — Sensor-agnostic virtual hand mirror

Fuses left + right webcam hand data into a single 3D world pose and
publishes it so the LinkerHand L25 in Gazebo mirrors the system's
internal perception of the operator's hand.

Data flow:
  /vision_input/hand_landmarks       (right hand tip_x/tip_y)   ──┐
  /vision_input/left_hand_landmarks  (left hand tip_y -> depth) ──┼── fuse -> /virtual_hand/base_pose
  /vision_input/gesture_text         (classified gesture)       ──┘       -> /virtual_hand/joint_states

Uses the *exact same* map_scale / map_offset parameters as robot_controller
so the virtual hand sits at the 3D position the robot is responding to.

Option B: each axis holds its last known value when that hand drops out of
frame, so loss of one hand never freezes the whole virtual hand.
"""

import math
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster


# Joint names (must match linkerhand_l25_right.urdf)
L25_JOINT_NAMES = [
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip', 'index_dip',
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip', 'middle_dip',
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip', 'ring_dip',
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip', 'pinky_dip',
]

# Canned finger poses (copy from virtual_operator_node)
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

# Default orientation: palm-down, fingers pointing toward robot (+X).
# This matches the virtual_operator's Safe Boot pose.
_DEFAULT_ROLL  = math.pi       # flip palm down
_DEFAULT_PITCH = 0.0
_DEFAULT_YAW   = -math.pi / 2  # fingers toward +X


def _rpy_to_quat(roll, pitch, yaw):
    """Standard RPY → quaternion (ZYX intrinsic)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # qx
        cr * sp * cy + sr * cp * sy,  # qy
        cr * cp * sy - sr * sp * cy,  # qz
        cr * cp * cy + sr * sp * sy,  # qw
    )


class HandMirrorNode(Node):

    def __init__(self):
        super().__init__('hand_mirror')

        # Parameters (same defaults as robot_controller)
        self.declare_parameter('map_offset_x', 0.30)
        self.declare_parameter('map_scale_x', 1.2)   # overhead X-axis (16:9 aspect fix)
        self.declare_parameter('map_scale_y', 0.9)
        self.declare_parameter('map_scale_z', 0.10)
        self.declare_parameter('map_offset_z', 0.45)
        self.declare_parameter('camera_placement', 'front_facing')

        # Publishers
        self.pose_pub = self.create_publisher(
            Pose, '/virtual_hand/base_pose', 10)
        self.joint_pub = self.create_publisher(
            JointState, '/virtual_hand/joint_states', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.create_subscription(
            Float32MultiArray, '/vision_input/hand_landmarks',
            self._right_hand_cb, 10)
        self.create_subscription(
            Float32MultiArray, '/vision_input/left_hand_landmarks',
            self._left_hand_cb, 10)
        self.create_subscription(
            String, '/vision_input/gesture_text',
            self._gesture_cb, 10)

        # State (Option B: each axis holds last known value)
        map_offset_x = self.get_parameter('map_offset_x').value
        map_offset_z = self.get_parameter('map_offset_z').value

        self._right_tip_x = 0.5    # normalized 0-1
        self._right_tip_y = 0.5
        self._left_tip_y  = 0.5
        self._gesture = "NEUTRAL"

        # Smoothed world-space position (EMA)
        self._smooth_x = map_offset_x
        self._smooth_y = 0.0
        self._smooth_z = map_offset_z

        # Left-hand depth smoother (matches robot_controller alpha=0.08)
        self._smooth_left_x = map_offset_x

        # Precompute static orientation quaternion
        self._qx, self._qy, self._qz, self._qw = _rpy_to_quat(
            _DEFAULT_ROLL, _DEFAULT_PITCH, _DEFAULT_YAW)

        # Table-calibration homography (optional)
        self._homography = None
        self._calib_W = 1280.0
        self._calib_H = 720.0
        self._load_table_calibration()

        # 30 Hz publish timer
        self.create_timer(1.0 / 30.0, self._tick)

        self.get_logger().info(
            "HandMirror active — fusing webcam hands → virtual hand 3D pose")

    # Homography loader (matches robot_controller)

    def _load_table_calibration(self):
        path = os.path.join(
            os.path.expanduser('~'), 'ros2_ws', 'data', 'table_calibration.yaml')
        if not os.path.exists(path):
            self.get_logger().info(
                "HandMirror: no table_calibration.yaml — using linear mapping.")
            return
        try:
            import yaml
            import numpy as np
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            H = data.get('homography')
            if H is None:
                return
            self._homography = np.array(H, dtype=np.float64)
            img_size = data.get('image_size', [1280, 720])
            self._calib_W = float(img_size[0])
            self._calib_H = float(img_size[1])
            self.get_logger().info(
                f"HandMirror: loaded homography ({int(self._calib_W)}x{int(self._calib_H)})")
        except Exception as e:
            self.get_logger().warn(f"HandMirror: failed to load {path}: {e}")

    def _homography_xy(self, norm_x, norm_y):
        if self._homography is None:
            return None
        try:
            import numpy as np
            px = norm_x * self._calib_W
            py = norm_y * self._calib_H
            v = np.array([[px, py, 1.0]], dtype=np.float64).T
            out = self._homography @ v
            w = out[2, 0]
            if abs(w) < 1e-9:
                return None
            return float(out[0, 0] / w), float(out[1, 0] / w)
        except Exception:
            return None

    # Callbacks: just store latest data

    def _right_hand_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 27:
            return
        # landmark[8] = index finger tip -> data[24]=x, data[25]=y
        self._right_tip_x = data[24]
        self._right_tip_y = data[25]

    def _left_hand_cb(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 27:
            return
        # landmark[8] = index finger tip -> data[25]=y (vertical position)
        self._left_tip_y = data[25]

    def _gesture_cb(self, msg: String):
        g = msg.data.strip()
        if g in GESTURE_POSES:
            self._gesture = g

    # Main tick: fuse -> publish

    def _tick(self):
        map_offset_x = self.get_parameter('map_offset_x').value
        map_scale_x  = self.get_parameter('map_scale_x').value
        map_scale_y  = self.get_parameter('map_scale_y').value
        map_scale_z  = self.get_parameter('map_scale_z').value
        map_offset_z = self.get_parameter('map_offset_z').value
        placement    = self.get_parameter('camera_placement').value

        if placement == 'front_facing':
            # Same transform as robot_controller lines 1324-1328 
            # Right hand -> Y, Z
            raw_y = -(self._right_tip_x - 0.5) * map_scale_y
            raw_z = (self._right_tip_y - 0.5) * map_scale_z + map_offset_z
            # Left hand -> X (depth slider, same as robot_controller:1510-1514)
            x_min, x_max = 0.15, 0.55
            raw_x_from_left = x_min + (1.0 - self._left_tip_y) * (x_max - x_min)
            self._smooth_left_x += 0.08 * (raw_x_from_left - self._smooth_left_x)
            raw_x = self._smooth_left_x
        else:
            # Overhead: right hand -> X,Y; left hand -> Z 
            # Prefer homography from table_calibration.yaml if available;
            # fall back to linear map_scale_x/y math otherwise.
            rxy = self._homography_xy(self._right_tip_x, self._right_tip_y)
            if rxy is not None:
                raw_x, raw_y = rxy
            else:
                raw_x = -(self._right_tip_y - 0.5) * map_scale_x + map_offset_x
                raw_y = -(self._right_tip_x - 0.5) * map_scale_y
            # Left hand vertical → robot Z
            raw_z = -(self._left_tip_y - 0.5) * map_scale_z + map_offset_z

        # EMA smooth the world position (alpha=0.18 — responsive but not jittery)
        alpha = 0.18
        self._smooth_x += alpha * (raw_x - self._smooth_x)
        self._smooth_y += alpha * (raw_y - self._smooth_y)
        self._smooth_z += alpha * (raw_z - self._smooth_z)

        # Publish Pose (consumed by virtual_hand_joint_driver plugin)
        pose = Pose()
        pose.position.x = self._smooth_x
        pose.position.y = self._smooth_y
        pose.position.z = self._smooth_z
        pose.orientation.x = self._qx
        pose.orientation.y = self._qy
        pose.orientation.z = self._qz
        pose.orientation.w = self._qw
        self.pose_pub.publish(pose)

        # Publish TF (world -> virtual_hand_base_link for RViz)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'virtual_hand_base_link'
        tf.transform.translation.x = self._smooth_x
        tf.transform.translation.y = self._smooth_y
        tf.transform.translation.z = self._smooth_z
        tf.transform.rotation.x = self._qx
        tf.transform.rotation.y = self._qy
        tf.transform.rotation.z = self._qz
        tf.transform.rotation.w = self._qw
        self._tf_broadcaster.sendTransform(tf)

        # Publish finger joint states
        poses = GESTURE_POSES.get(self._gesture, GESTURE_POSES["NEUTRAL"])
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = L25_JOINT_NAMES
        js.position = [float(v) for v in poses]
        self.joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = HandMirrorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

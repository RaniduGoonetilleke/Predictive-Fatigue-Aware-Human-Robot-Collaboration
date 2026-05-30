#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import csv
import os
import time

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')
        
        # 1. Setting up the file
        # We save data to: ~/ros2_ws/data/captured_gesture.csv
        home = os.path.expanduser('~')
        self.data_dir = os.path.join(home, 'ros2_ws', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create a unique file name based on time so we don't overwrite old data
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filename = os.path.join(self.data_dir, f'gesture_recording_{timestamp}.csv')
        
        # 2. Open the CSV file
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write the Header Row
        # "Time" + 63 columns (x,y,z for 21 landmarks)
        header = ['timestamp'] + [f'lm_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        self.csv_writer.writerow(header)
        
        # 3. Subscribe to the "Clean" Data
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/vision_input/hand_landmarks',
            self.listener_callback,
            10
        )
        
        self.get_logger().info(f'RECORDER STARTED. Saving to: {self.filename}')
        self.get_logger().info('Perform your gesture now! Press Ctrl+C to stop.')

    def listener_callback(self, msg):
        # This function runs every time the camera sees a VALID hand
        row = [time.time()] + msg.data.tolist()
        self.csv_writer.writerow(row)
        # Print a dot to show it's working
        print(".", end="", flush=True)

    def __del__(self):
        self.csv_file.close()
        self.get_logger().info('File closed. Data saved successfully.')

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

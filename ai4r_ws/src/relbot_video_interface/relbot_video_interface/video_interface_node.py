#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2
import json
import os

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ── Distance estimation constants ──────────────────────────────────
ASSUMED_HEIGHT_CM = 175.0    # Assumed person height in cm
STOP_DISTANCE_CM = 500.0     # 5 m — safety stop threshold
TARGET_DISTANCE_CM = 1000.0   # 10 m — desired following distance
DEFAULT_FOCAL_LENGTH_PX = 400.0  # Fallback if no calibration file found


class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        # Publisher: sends object position to the RELBot
        # Topic `/object_position` is watched by the robot controller for actuation
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        # Declare GStreamer pipeline as a parameter for flexibility
        self.declare_parameter('gst_pipeline', (
            'udpsrc address=0.0.0.0 port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            'videoconvert ! video/x-raw, format=BGR ! '
            'appsink drop=true max-buffers=1 sync=false'
        ))
        pipeline_str = self.get_parameter('gst_pipeline').value

        # Load YOLO model
        self.model = YOLO("yolo26n.pt")
        self.get_logger().info('YOLO model loaded')

        # Open webcam
        self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open GStreamer pipeline, falling back to index 0')
            self.cap = cv2.VideoCapture(0)

        # Load camera calibration for distance estimation
        self.focal_length_px = self._load_calibration()

        # Tracking variables
        self.target_id = -1
        self.frames_lost = 0
        self.max_lost_frames = 90

        # Timer: fires at ~30Hz to pull frames and publish positions
        # The period (1/30) sets how often on_timer() is called
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

    def _load_calibration(self):
        """Load focal length from calibration.json, or use default."""
        calibration_path = 'calibration.json'
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, 'r') as f:
                    data = json.load(f)
                focal_length = data['focal_length_px']
                self.get_logger().info(
                    f'Loaded camera calibration: focal_length = {focal_length:.2f} px '
                    f'(reprojection error: {data.get("reprojection_error", "N/A")})'
                )
                return focal_length
            except (json.JSONDecodeError, KeyError) as e:
                self.get_logger().warn(
                    f'Failed to read calibration file: {e}. Using default focal length.'
                )
        else:
            self.get_logger().warn(
                f'No calibration.json found. Using default focal length = {DEFAULT_FOCAL_LENGTH_PX} px. '
                f'Run calibrate_camera for accurate distance estimation.'
            )
        return DEFAULT_FOCAL_LENGTH_PX

    def _estimate_distance_cm(self, bbox_xyxy):
        bbox_height_px = bbox_xyxy[3] - bbox_xyxy[1]
        if bbox_height_px <= 0:
            return float('inf')
        distance_cm = (ASSUMED_HEIGHT_CM * self.focal_length_px) / bbox_height_px
        return distance_cm

    def _distance_to_z(self, distance_cm):
        if distance_cm <= TARGET_DISTANCE_CM:
            # Person is within 10m (or closer) — hold position / safety stop
            return 10001.0
        else:
            # Person is beyond 10m — approach
            return 10000.0 * (TARGET_DISTANCE_CM / distance_cm)

    def on_timer(self):
        # Pull the latest frame from the GStreamer appsink
        ret, frame = self.cap.read()
        if not ret:
            # No new frame available
            return

        # Display the raw input frame for debugging
        cv2.imshow('Input Stream', frame)
        cv2.waitKey(1)

        result = self.model.track(frame, conf=0.5, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
        annotated = result[0].plot()
        cv2.imshow('YOLO', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        msg = Point()
        msg.x = 160.0  # object center x-coordinate (scaled middle)

        boxes = result[0].boxes
        target_found = False

        if boxes.id is not None:
            ids = boxes.id.tolist()
            xyxys = boxes.xyxy.tolist()

            # If no target assigned yet, latch onto the first detected person
            if self.target_id == -1 and len(ids) > 0:
                self.target_id = ids[0]
                self.get_logger().info(f"Latched onto new target ID: {self.target_id}")

            # Find our target person
            for bbox, track_id in zip(xyxys, ids):
                if track_id == self.target_id:
                    target_found = True
                    self.frames_lost = 0

                    # Compute horizontal center of the person in raw pixels
                    center_x_raw = (bbox[0] + bbox[2]) / 2
                    
                    # Scale x from original frame width to 320
                    frame_width = frame.shape[1]
                    msg.x = center_x_raw * (320.0 / frame_width)

                    # Estimate distance using pinhole camera model
                    distance_cm = self._estimate_distance_cm(bbox)

                    # Map distance to controller z-value
                    msg.z = self._distance_to_z(distance_cm)

                    self.get_logger().info(
                        f'\n{"="*50}\n'
                        f'TRACKING ID: {self.target_id}\n'
                        f'    Distance: {distance_cm/100:.2f} meters ({distance_cm:.0f} cm)\n'
                        f'    Center X: {msg.x:.1f} px (scaled from {center_x_raw:.1f} px)\n'
                        f'    Command (z): {msg.z:.1f}\n'
                        f'{"="*50}\n'
                    )
                    break

        if not target_found:
            self.frames_lost += 1
            msg.z = 10001.0  # safety stop
            
            if self.frames_lost > self.max_lost_frames:
                if self.target_id != -1:
                    self.get_logger().info(f"Lost target ID {self.target_id}. Resetting.")
                self.target_id = -1
            else:
                self.get_logger().debug('Target not seen in frame, stopping robot')

        self.position_pub.publish(msg)
        self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')

    def destroy_node(self):
        # Cleanup GStreamer resources on shutdown
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Booster Robotics Robot Controller with Face Tracking

This controller supports multiple upper body control modes:
- policy: Upper body controlled by reinforcement learning policy
- teleop: Upper body controlled by VR/teleop input  
- sine: Upper body controlled by sine wave patterns

Face Tracking Features:
- When FACE_TRACKING=True and upper body mode is "teleop", the head will 
  automatically track faces detected in camera feeds
- Uses the largest detected face as the tracking target
- Smoothly moves head to center the face in the camera view
- Automatically discovers and subscribes to ROS2 camera topics
- No need to run webrtc.py - works directly with ROS2 camera streams

Robot Head Control Conventions:
- Positive head yaw → face left
- Negative head yaw → face right  
- Negative head pitch → look up
- Positive head pitch → look down

Usage:
1. Set FACE_TRACKING = True/False to enable/disable face tracking
2. Set UPPER_BODY_CONTROL_MODE to "teleop" to use face tracking
3. Ensure your camera is publishing to ROS2 topics (e.g., realsense, usb_cam)
4. Robot will automatically discover and track faces

Command Line Options:
- --no-face-tracking: Disable face tracking even if FACE_TRACKING=True
- --face-tracking-camera TOPIC: Specify preferred camera topic

Requirements for face tracking:
- opencv-python (cv2)
- rclpy (ROS2 Python client)
- sensor_msgs (ROS2 sensor messages)
- cv_bridge (ROS2 OpenCV bridge)
- Active ROS2 camera topics (automatically discovered)

Example:
    python deploy.py --config your_config.yaml
    python deploy.py --config your_config.yaml --face-tracking-camera /camera/color/image_raw
"""

import numpy as np
import time
import yaml
import logging
import threading
from typing import Optional
import cv2
import os
import argparse

# ROS2 imports for camera topic subscription
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy import Policy
from enum import Enum

# Global constants for upper body control modes
# Set one of these to the string value to enable that control mode
# Options: "policy", "teleop", "sine"
UPPER_BODY_CONTROL_MODE = "teleop"  # Default to policy control

# Face tracking control
FACE_TRACKING = True  # Enable/disable face tracking

# Control parameter for arm gains
# Lower value means smoother movements with less stiffness
# Range: 0.1 (very soft) to 1.0 (full stiffness)
ARM_STIFFNESS_FACTOR = 0.2 * 1.25 * 1.25 * 1.25 * 1.25

# Global variables for camera frames (shared with webrtc.py if running)
latest_frames = {}
frame_locks = {}

class FaceTrackingCameraSubscriber(Node):
    """ROS2 Node to subscribe to camera topics for face tracking"""
    
    def __init__(self, topic_name, topic_type='raw'):
        """
        Initialize the camera subscriber for face tracking
        
        Args:
            topic_name: Name of the ROS2 topic to subscribe to
            topic_type: Type of the topic ('raw' or 'compressed')
        """
        # Use a specific node name for face tracking
        safe_topic_name = topic_name.replace('/', '_').strip('_')
        super().__init__(f'face_tracking_camera_{safe_topic_name}')
        
        self.topic_name = topic_name
        self.bridge = CvBridge()
        self.logger = self.get_logger()
        
        # Create a lock for this topic if it doesn't exist
        if topic_name not in frame_locks:
            frame_locks[topic_name] = threading.Lock()
        
        # Subscribe to the appropriate topic type
        if topic_type == 'compressed':
            self.subscription = self.create_subscription(
                CompressedImage,
                topic_name,
                self.compressed_callback,
                10)
            self.logger.info(f"Face tracking subscribed to compressed image topic: {topic_name}")
        else:
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.raw_callback,
                10)
            self.logger.info(f"Face tracking subscribed to raw image topic: {topic_name}")
    
    def raw_callback(self, msg):
        """Process raw image messages for face tracking"""
        try:
            # Check if this is a depth image
            is_depth_image = (
                msg.encoding in ['16UC1', '32FC1', 'mono16'] 
                or 'depth' in self.topic_name.lower()
            )
            
            if is_depth_image:
                # Skip depth images for face tracking
                return
                
            # Convert ROS Image to OpenCV image
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.logger.debug(f"Error converting image from {self.topic_name}: {e}")
                return
            
            # Store the frame
            timestamp = time.time()
            with frame_locks[self.topic_name]:
                latest_frames[self.topic_name] = {
                    'image': cv_image,
                    'timestamp': timestamp,
                    'ros_timestamp': msg.header.stamp,
                    'frame_id': msg.header.frame_id,
                    'height': msg.height,
                    'width': msg.width
                }
            
        except Exception as e:
            self.logger.error(f"Error processing raw image from {self.topic_name}: {e}")
    
    def compressed_callback(self, msg):
        """Process compressed image messages for face tracking"""
        try:
            # Skip depth images
            if 'depth' in self.topic_name.lower():
                return
                
            # Convert compressed image to OpenCV image
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    self.logger.debug(f"Failed to decode compressed image from {self.topic_name}")
                    return
                    
            except Exception as e:
                self.logger.debug(f"Error decoding compressed image from {self.topic_name}: {e}")
                return
            
            # Store the frame
            timestamp = time.time()
            with frame_locks[self.topic_name]:
                latest_frames[self.topic_name] = {
                    'image': cv_image,
                    'timestamp': timestamp,
                    'ros_timestamp': msg.header.stamp,
                    'frame_id': msg.header.frame_id,
                    'height': cv_image.shape[0],
                    'width': cv_image.shape[1]
                }
            
        except Exception as e:
            self.logger.error(f"Error processing compressed image from {self.topic_name}: {e}")

def discover_camera_topics(node):
    """
    Discover available ROS2 image topics
    
    Args:
        node: ROS2 node to use for topic discovery
        
    Returns:
        List of (topic_name, topic_type) tuples
    """
    topic_names_and_types = node.get_topic_names_and_types()
    image_topics = []
    
    for name, types in topic_names_and_types:
        for type_name in types:
            # Skip depth topics for face tracking
            if 'depth' in name.lower():
                continue
                
            if type_name == 'sensor_msgs/msg/Image':
                image_topics.append((name, 'raw'))
            elif type_name == 'sensor_msgs/msg/CompressedImage':
                image_topics.append((name, 'compressed'))
    
    return image_topics

class BodyPart(Enum):
    LOWER_BODY = 0  # Legs and torso
    UPPER_BODY = 1  # Arms

class UpperBodyControlMode(Enum):
    POLICY = "policy"    # Upper body controlled by policy
    TELEOP = "teleop"    # Upper body controlled by VR/teleop
    SINE = "sine"        # Upper body controlled by sine wave

class FaceTracker:
    """Face tracking system for head control"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FaceTracker")
        
        # Initialize face detection
        try:
            # Try to use DNN face detection with OpenCV's built-in model
            # This requires opencv-contrib-python and the model files
            self.use_dnn = False  # Start with False, enable if successful
            
            # Try to load DNN face detection model
            # You can download these from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                self.use_dnn = True
                self.logger.info("Using DNN face detection model")
            else:
                raise FileNotFoundError("DNN model files not found")
                
        except Exception as e:
            self.logger.debug(f"DNN face detection not available: {e}")
            
        if not self.use_dnn:
            # Fallback to Haar cascades
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise RuntimeError("Failed to load Haar cascade classifier")
                self.logger.info("Using Haar cascade face detection")
            except Exception as e:
                self.logger.error("Failed to initialize face detection")
                raise RuntimeError(f"Face detection initialization failed: {e}")
        
        # Head control parameters
        self.target_head_yaw = 0.0
        self.target_head_pitch = 0.0
        self.head_yaw_limits = (-1.2, 1.2)  # radians, roughly ±70 degrees
        self.head_pitch_limits = (-0.8, 0.8)  # radians, roughly ±45 degrees
        
        # Fifth-order smoothing state variables (position, velocity, acceleration, jerk, snap)
        self.smoothed_head_yaw = 0.0
        self.smoothed_head_pitch = 0.0
        self.head_yaw_velocity = 0.0
        self.head_pitch_velocity = 0.0
        self.head_yaw_acceleration = 0.0
        self.head_pitch_acceleration = 0.0
        self.head_yaw_jerk = 0.0
        self.head_pitch_jerk = 0.0
        self.head_yaw_snap = 0.0
        self.head_pitch_snap = 0.0
        
        # Control gains for face centering (reduced for less aggressive movement)
        self.yaw_gain = 0.001   # Reduced from 0.002 - How aggressively to turn left/right
        self.pitch_gain = 0.001 # Reduced from 0.002 - How aggressively to look up/down
        
        # Fifth-order smoothing parameters (critically damped system)
        self.smoothing_frequency = 1.5  # Hz - how fast the system responds (reduced for smoother motion)
        self.smoothing_damping = 1.0    # Damping ratio (1.0 = critically damped)
        self.dt = 0.02  # Control loop timestep (50Hz)
        
        # Calculate fifth-order filter coefficients
        omega = 2.0 * np.pi * self.smoothing_frequency
        zeta = self.smoothing_damping
        
        # Fifth-order critically damped system coefficients (binomial series: 1, 5, 10, 10, 5, 1)
        omega_zeta = omega * zeta
        self.k1 = omega_zeta**5      # (ζω)⁵ - position term
        self.k2 = 5.0 * omega_zeta**4  # 5(ζω)⁴ - velocity term  
        self.k3 = 10.0 * omega_zeta**3 # 10(ζω)³ - acceleration term
        self.k4 = 10.0 * omega_zeta**2 # 10(ζω)² - jerk term
        self.k5 = 5.0 * omega_zeta     # 5(ζω) - snap term
        
        self.return_to_center_rate = 0.98  # Rate at which head returns to center when no face detected
        
        # Face detection parameters
        self.min_face_size = (50, 50)  # Minimum face size to consider
        self.detection_confidence = 0.8  # Increased from 0.7 - For DNN detection (higher = more confident detections only)
        
        # Movement threshold parameters - only update targets if face moves significantly
        self.movement_threshold_pixels = 15  # Minimum pixel movement to trigger head update
        self.last_face_center = None  # Store last face center (x, y)
        self.face_stability_buffer = []  # Buffer to track face stability
        self.stability_buffer_size = 5  # Number of frames to average for stability
        
        # Tracking state
        self.last_face_time = time.time()
        self.face_timeout = 2.0  # Seconds after which to start returning to center
        
        self.logger.info(f"Face tracker initialized using {'DNN' if self.use_dnn else 'Haar cascades'}")
        self.logger.info(f"Smoothing frequency: {self.smoothing_frequency} Hz, damping ratio: {self.smoothing_damping}")
    
    def detect_faces(self, frame):
        """Detect faces in the given frame
        
        Args:
            frame: OpenCV image frame (BGR format)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            if self.use_dnn:
                return self._detect_faces_dnn(frame)
            else:
                return self._detect_faces_haar(frame)
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []
    
    def _detect_faces_dnn(self, frame):
        """DNN-based face detection"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_confidence:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                face_w = x2 - x1
                face_h = y2 - y1
                
                if face_w >= self.min_face_size[0] and face_h >= self.min_face_size[1]:
                    faces.append((x1, y1, face_w, face_h))
        
        return faces
    
    def _detect_faces_haar(self, frame):
        """Haar cascade-based face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size
        )
        return faces.tolist() if len(faces) > 0 else []
    
    def get_largest_face(self, faces):
        """Get the largest face from the list of faces
        
        Args:
            faces: List of face bounding boxes [(x, y, w, h), ...]
            
        Returns:
            Largest face bounding box (x, y, w, h) or None if no faces
        """
        if not faces:
            return None
        
        # Find face with largest area
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return largest_face
    
    def calculate_head_movement(self, face, frame_width, frame_height):
        """Calculate required head movement to center the face
        
        Args:
            face: Face bounding box (x, y, w, h)
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
            
        Returns:
            (delta_yaw, delta_pitch) - changes needed for head positioning
        """
        if face is None:
            return 0.0, 0.0
        
        x, y, w, h = face
        
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate frame center
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        # Calculate error (how far face is from center)
        error_x = face_center_x - frame_center_x  # positive = face is to the right
        error_y = face_center_y - frame_center_y  # positive = face is below center
        
        # Convert pixel error to head movement
        # Robot head control conventions:
        # - Positive head yaw → face left
        # - Negative head yaw → face right  
        # - Negative head pitch → look up
        # - Positive head pitch → look down
        
        # If face is to the right (positive error_x), we need to turn right (negative yaw)
        delta_yaw = -error_x * self.yaw_gain
        # If face is below center (positive error_y), we need to look down (positive pitch)
        delta_pitch = error_y * self.pitch_gain
        
        return delta_yaw, delta_pitch
    
    def update_head_targets(self, delta_yaw, delta_pitch):
        """Update head target positions with limits
        
        Args:
            delta_yaw: Change in yaw position
            delta_pitch: Change in pitch position
        """
        # Update targets
        self.target_head_yaw += delta_yaw
        self.target_head_pitch += delta_pitch
        
        # Apply limits
        self.target_head_yaw = np.clip(self.target_head_yaw, *self.head_yaw_limits)
        self.target_head_pitch = np.clip(self.target_head_pitch, *self.head_pitch_limits)
    
    def apply_smoothing_and_centering(self):
        """Apply fifth-order smoothing for smoothing and return to center when no face detected"""
        current_time = time.time()
        time_since_face = current_time - self.last_face_time
        
        # If no face detected for a while, gradually return to center
        if time_since_face > self.face_timeout:
            self.target_head_yaw *= self.return_to_center_rate
            self.target_head_pitch *= self.return_to_center_rate
            
            # Zero out very small values to avoid drift
            if abs(self.target_head_yaw) < 0.01:
                self.target_head_yaw = 0.0
            if abs(self.target_head_pitch) < 0.01:
                self.target_head_pitch = 0.0
        
        # Apply fifth-order critically damped filter for both axes
        self._update_fifth_order_filter_yaw()
        self._update_fifth_order_filter_pitch()
    
    def _update_fifth_order_filter_yaw(self):
        """Update yaw using fifth-order critically damped system"""
        # Error between target and current position
        error = self.target_head_yaw - self.smoothed_head_yaw
        
        # Fifth-order critically damped response
        # d⁵x/dt⁵ + 5ζω(d⁴x/dt⁴) + 10(ζω)²(d³x/dt³) + 10(ζω)³(d²x/dt²) + 5(ζω)⁴(dx/dt) + (ζω)⁵x = (ζω)⁵target
        
        # Update snap (5th derivative) based on error and all lower derivatives
        snap_update = self.dt * (
            self.k1 * error - 
            self.k2 * self.head_yaw_velocity - 
            self.k3 * self.head_yaw_acceleration - 
            self.k4 * self.head_yaw_jerk - 
            self.k5 * self.head_yaw_snap
        )
        self.head_yaw_snap += snap_update
        
        # Update jerk (4th derivative) based on snap
        self.head_yaw_jerk += self.dt * self.head_yaw_snap
        
        # Update acceleration (3rd derivative) based on jerk
        self.head_yaw_acceleration += self.dt * self.head_yaw_jerk
        
        # Update velocity (2nd derivative) based on acceleration
        self.head_yaw_velocity += self.dt * self.head_yaw_acceleration
        
        # Update position (1st derivative) based on velocity
        self.smoothed_head_yaw += self.dt * self.head_yaw_velocity
    
    def _update_fifth_order_filter_pitch(self):
        """Update pitch using fifth-order critically damped system"""
        # Error between target and current position
        error = self.target_head_pitch - self.smoothed_head_pitch
        
        # Fifth-order critically damped response
        
        # Update snap (5th derivative) based on error and all lower derivatives
        snap_update = self.dt * (
            self.k1 * error - 
            self.k2 * self.head_pitch_velocity - 
            self.k3 * self.head_pitch_acceleration - 
            self.k4 * self.head_pitch_jerk - 
            self.k5 * self.head_pitch_snap
        )
        self.head_pitch_snap += snap_update
        
        # Update jerk (4th derivative) based on snap
        self.head_pitch_jerk += self.dt * self.head_pitch_snap
        
        # Update acceleration (3rd derivative) based on jerk
        self.head_pitch_acceleration += self.dt * self.head_pitch_jerk
        
        # Update velocity (2nd derivative) based on acceleration
        self.head_pitch_velocity += self.dt * self.head_pitch_acceleration
        
        # Update position (1st derivative) based on velocity
        self.smoothed_head_pitch += self.dt * self.head_pitch_velocity
    
    def process_frame_and_get_head_targets(self, frame):
        """Process camera frame and return head target positions
        
        Args:
            frame: Camera frame from latest_frames
            
        Returns:
            (head_yaw, head_pitch) target positions
        """
        if frame is None:
            # No frame available, apply smoothing and return current smoothed targets
            self.apply_smoothing_and_centering()
            return self.smoothed_head_yaw, self.smoothed_head_pitch
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        if faces:
            # Update last face detection time
            self.last_face_time = time.time()
            
            # Get largest face
            largest_face = self.get_largest_face(faces)
            x, y, w, h = largest_face
            
            # Calculate face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            current_face_center = (face_center_x, face_center_y)
            
            # Add to stability buffer for averaging
            self.face_stability_buffer.append(current_face_center)
            if len(self.face_stability_buffer) > self.stability_buffer_size:
                self.face_stability_buffer.pop(0)  # Remove oldest
            
            # Calculate averaged face center for stability
            if len(self.face_stability_buffer) >= 3:  # Need at least 3 frames for stability
                avg_x = sum(pos[0] for pos in self.face_stability_buffer) / len(self.face_stability_buffer)
                avg_y = sum(pos[1] for pos in self.face_stability_buffer) / len(self.face_stability_buffer)
                stable_face_center = (avg_x, avg_y)
            else:
                stable_face_center = current_face_center
            
            # Check if face has moved significantly since last update
            should_update_targets = True
            movement_distance = 0.0  # Initialize movement_distance
            
            if self.last_face_center is not None:
                dx = abs(stable_face_center[0] - self.last_face_center[0])
                dy = abs(stable_face_center[1] - self.last_face_center[1])
                movement_distance = (dx**2 + dy**2)**0.5
                
                if movement_distance < self.movement_threshold_pixels:
                    should_update_targets = False
                    self.logger.debug(f"Face movement too small ({movement_distance:.1f} pixels), not updating targets")
            else:
                # First time seeing a face, always update
                movement_distance = float('inf')  # Indicate first detection
            
            # Only update head targets if face moved significantly
            if should_update_targets:
                # Use stable face center for movement calculation
                stable_face = (stable_face_center[0] - w//2, stable_face_center[1] - h//2, w, h)
                
                # Calculate required head movement
                delta_yaw, delta_pitch = self.calculate_head_movement(
                    stable_face, frame_width, frame_height
                )
                
                # Update head targets
                self.update_head_targets(delta_yaw, delta_pitch)
                
                # Update last face center
                self.last_face_center = stable_face_center
                
                # Debug logging
                if movement_distance == float('inf'):
                    self.logger.debug(
                        f"First face detection, updating targets: yaw={self.target_head_yaw:.3f}, pitch={self.target_head_pitch:.3f}"
                    )
                else:
                    self.logger.debug(
                        f"Face moved {movement_distance:.1f} pixels, updating targets: yaw={self.target_head_yaw:.3f}, pitch={self.target_head_pitch:.3f}"
                    )
            else:
                # Debug logging for no update
                self.logger.debug(f"Face stable at ({stable_face_center[0]:.0f}, {stable_face_center[1]:.0f}), keeping current targets")
                
        else:
            # No faces detected - clear stability buffer and last face center
            self.face_stability_buffer.clear()
            self.last_face_center = None
            
            current_time = time.time()
            time_since_face = current_time - self.last_face_time
            if time_since_face > self.face_timeout:
                self.logger.debug(f"No face detected for {time_since_face:.1f}s, returning to center")
        
        # Apply smoothing and centering logic
        self.apply_smoothing_and_centering()
        
        self.logger.debug(
            f"Final smoothed head targets: yaw={self.smoothed_head_yaw:.3f}, pitch={self.smoothed_head_pitch:.3f}"
        )
        
        return self.smoothed_head_yaw, self.smoothed_head_pitch

    def reset_head_position(self):
        """Reset head to center position"""
        self.target_head_yaw = 0.0
        self.target_head_pitch = 0.0
        self.smoothed_head_yaw = 0.0
        self.smoothed_head_pitch = 0.0
        
        # Reset fifth-order filter states
        self.head_yaw_velocity = 0.0
        self.head_pitch_velocity = 0.0
        self.head_yaw_acceleration = 0.0
        self.head_pitch_acceleration = 0.0
        self.head_yaw_jerk = 0.0
        self.head_pitch_jerk = 0.0
        self.head_yaw_snap = 0.0
        self.head_pitch_snap = 0.0
        
        self.last_face_time = time.time()
        
        # Clear face tracking state
        self.last_face_center = None
        self.face_stability_buffer.clear()
        
        self.logger.info("Face tracker head position reset to center")

    def set_smoothing_frequency(self, frequency):
        """Set the smoothing frequency for fifth-order filtering
        
        Args:
            frequency: Smoothing frequency in Hz (0.0 = no smoothing, positive value)
        """
        self.smoothing_frequency = max(0.0, frequency)
        self.logger.info(f"Face tracking smoothing frequency set to {self.smoothing_frequency} Hz")
    
    def set_smoothing_damping(self, damping):
        """Set the damping ratio for fifth-order filtering
        
        Args:
            damping: Damping ratio (0.0 = no damping, 1.0 = critically damped)
        """
        self.smoothing_damping = np.clip(damping, 0.0, 1.0)
        self.logger.info(f"Face tracking smoothing damping set to {self.smoothing_damping}")

    def set_movement_threshold(self, threshold_pixels):
        """Set the movement threshold for face tracking updates
        
        Args:
            threshold_pixels: Minimum pixel movement to trigger head updates
        """
        self.movement_threshold_pixels = max(1, threshold_pixels)
        self.logger.info(f"Face tracking movement threshold set to {self.movement_threshold_pixels} pixels")

class Controller:
    def __init__(self, cfg_file, preferred_camera_topic=None) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store preferred camera topic for face tracking
        self.preferred_camera_topic = preferred_camera_topic

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        # Reduce stiffness for arm gains directly in the config
        # This ensures the arms always have low gains regardless of other settings
        # Arm joints are indices 2-9: shoulders and elbows
        if "common" in self.cfg and "stiffness" in self.cfg["common"]:
            for arm_joint_idx in range(2, 10):  # Indices 2-9 are arm joints
                if arm_joint_idx < len(self.cfg["common"]["stiffness"]):
                    # Apply stiffness reduction
                    self.cfg["common"]["stiffness"][arm_joint_idx] *= ARM_STIFFNESS_FACTOR
                    self.logger.info(f"Reduced stiffness for arm joint {arm_joint_idx} to {self.cfg['common']['stiffness'][arm_joint_idx]}")

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg)
        
        # Initialize ROS2 for camera topics if face tracking is enabled
        self.ros_initialized = False
        self.camera_subscribers = []
        self.ros_executor = None
        self.ros_thread = None
        
        if FACE_TRACKING:
            self._setup_ros2_camera_system()
        
        # Initialize face tracker if enabled
        self.face_tracker = None
        if FACE_TRACKING:
            try:
                self.face_tracker = FaceTracker()
                self.logger.info("Face tracking enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize face tracker: {e}")
                self.logger.info("Continuing without face tracking")
                self.face_tracker = None

        # Define joint indices for body parts based on the actual robot configuration
        # Head and arms (upper body) - NOW 10 JOINTS (NO WAIST)
        self.upper_body_indices = [
            0, 1,                # Head (yaw, pitch)
            2, 3, 4, 5,          # Left arm (shoulder pitch, roll, yaw, elbow)
            6, 7, 8, 9,          # Right arm (shoulder pitch, roll, yaw, elbow)
            # 10                    # Waist yaw - REMOVED
        ]
        
        # Legs (lower body)
        self.lower_body_indices = [
            11, 12, 13, 14, 15, 16,  # Left leg (hip pitch, roll, yaw, knee, ankle up/down)
            17, 18, 19, 20, 21, 22   # Right leg (hip pitch, roll, yaw, knee, ankle up/down)
        ]
        
        # Log joint configuration for debugging
        self.logger.info(f"Upper body indices: {self.upper_body_indices}")
        self.logger.info(f"Lower body indices: {self.lower_body_indices}")
        
        # Check for potential waist joint at index 10
        if 10 not in self.upper_body_indices and 10 not in self.lower_body_indices:
            self.logger.warning("Joint index 10 (waist) is not assigned to any body part - this could cause issues!")
        
        # Control mode for each body part
        self.body_part_control_mode = {
            BodyPart.LOWER_BODY: "policy",  # Controlled by policy
            BodyPart.UPPER_BODY: self._determine_upper_body_mode()
        }
        
        # Reference to external sine wave controller
        self.sine_controller = None
        
        # Default positions for manual control
        self.manual_upper_body_positions = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)[self.upper_body_indices]
        
        # We'll receive sine wave positions from the upper_body_controller
        # Ensure this also reflects the 10-joint setup if upper_body_indices is used for slicing, or initialize explicitly for 10.
        # If default_qpos has 11+ elements, self.upper_body_indices will correctly slice the first 10.
        self.sine_upper_body_positions = np.copy(self.manual_upper_body_positions) # Should now be 10 elements

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.remoteControlService.close()
        
        # Cleanup ROS2 camera system
        if self.ros_initialized:
            try:
                if self.ros_executor:
                    self.ros_executor.shutdown()
                if self.ros_thread and self.ros_thread.is_alive():
                    self.ros_thread.join(timeout=1.0)
                for subscriber in self.camera_subscribers:
                    subscriber.destroy_node()
                self.logger.info("ROS2 camera system cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up ROS2 camera system: {e}")
        
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        
        # Initialize manual upper body positions from current robot state to prevent jumps
        self._initialize_upper_body_from_current_state()
        
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def _initialize_upper_body_from_current_state(self):
        """Initialize manual upper body positions from current robot joint positions"""
        if hasattr(self, 'dof_pos_latest') and len(self.dof_pos_latest) > 0:
            for i, idx in enumerate(self.upper_body_indices):
                if idx < len(self.dof_pos_latest):
                    self.manual_upper_body_positions[i] = self.dof_pos_latest[idx]
            self.logger.info("Initialized upper body positions from current robot state")
            self.logger.info(f"Upper body positions: {self.manual_upper_body_positions}")
            
            # Also initialize face tracker from current head position
            if self.face_tracker is not None and len(self.dof_pos_latest) >= 2:
                self.face_tracker.target_head_yaw = self.dof_pos_latest[0]
                self.face_tracker.target_head_pitch = self.dof_pos_latest[1]
                self.face_tracker.smoothed_head_yaw = self.dof_pos_latest[0]
                self.face_tracker.smoothed_head_pitch = self.dof_pos_latest[1]
                
                # Initialize fifth-order filter states to zero (no initial motion)
                self.face_tracker.head_yaw_velocity = 0.0
                self.face_tracker.head_pitch_velocity = 0.0
                self.face_tracker.head_yaw_acceleration = 0.0
                self.face_tracker.head_pitch_acceleration = 0.0
                self.face_tracker.head_yaw_jerk = 0.0
                self.face_tracker.head_pitch_jerk = 0.0
                self.face_tracker.head_yaw_snap = 0.0
                self.face_tracker.head_pitch_snap = 0.0
                
                self.logger.info(f"Initialized face tracker from current head position: yaw={self.dof_pos_latest[0]:.3f}, pitch={self.dof_pos_latest[1]:.3f}")
        else:
            self.logger.warning("Could not initialize from current state, using config defaults")

    def _determine_upper_body_mode(self):
        """Determine the upper body control mode based on global constant"""
        print(f"\nDEBUG - UPPER_BODY_CONTROL_MODE = {UPPER_BODY_CONTROL_MODE}\n")
        
        if UPPER_BODY_CONTROL_MODE == "teleop":
            print("Setting mode to TELEOP")
            return UpperBodyControlMode.TELEOP.value
        elif UPPER_BODY_CONTROL_MODE == "sine":
            print("Setting mode to SINE")
            return UpperBodyControlMode.SINE.value
        else:
            # Default to policy control for any other value
            print("Setting mode to POLICY (default)")
            return UpperBodyControlMode.POLICY.value
            
    def set_upper_body_positions(self, positions):
        """Set the upper body joint positions for teleop control mode.
        
        Args:
            positions: Array of joint positions for upper body joints
                      Can be either 10 joints (head + arms) or 11 joints (head + arms + waist)
        """
        with self.publish_lock:  # Use lock to avoid race conditions
            if len(positions) == len(self.upper_body_indices):
                # If we have all 11 joints, copy them directly
                self.manual_upper_body_positions = np.copy(positions)
            elif len(positions) == 10:  # If we have 10 joints (head + arms, no waist)
                # Copy the first 10 joints and keep the waist position unchanged
                self.manual_upper_body_positions[:10] = np.copy(positions[:10])
                # Note: waist position at index 10 remains unchanged
            else:
                self.logger.warning(f"Received positions array of length {len(positions)}, expected 10 or 11")
                return
    
    def set_body_part_control_mode(self, body_part, mode):
        """Set the control mode for a specific body part.
        
        Args:
            body_part: BodyPart enum (UPPER_BODY or LOWER_BODY)
            mode: Control mode ("policy", "teleop", "sine")
        """
        if mode in [UpperBodyControlMode.POLICY.value, UpperBodyControlMode.TELEOP.value, UpperBodyControlMode.SINE.value]:
            self.body_part_control_mode[body_part] = mode
            self.logger.info(f"Set {body_part} control mode to {mode}")
        else:
            self.logger.warning(f"Invalid control mode: {mode}")
    
    def set_sine_controller(self, controller):
        """Set the external sine wave controller.
        
        Args:
            controller: The sine wave controller that provides joint positions
        """
        self.sine_controller = controller
        
    def set_sine_upper_body_positions(self, positions):
        """Set the sine wave positions for the upper body from external controller.
        
        Args:
            positions: Array of joint positions for upper body joints
        """
        if len(positions) == len(self.upper_body_indices):
            with self.publish_lock:  # Use lock to avoid race conditions
                self.sine_upper_body_positions = np.copy(positions)
        else:
            self.logger.warning(f"Received sine positions array of length {len(positions)}, expected {len(self.upper_body_indices)}")
    
    def get_camera_frame(self, preferred_topic=None):
        """Get the latest camera frame for face tracking
        
        Args:
            preferred_topic: Preferred camera topic name, if None uses first available
            
        Returns:
            OpenCV image frame or None if no frame available
        """
        global latest_frames, frame_locks
        
        if not latest_frames:
            return None
        
        # Try to get preferred topic first
        if preferred_topic and preferred_topic in latest_frames:
            topic_name = preferred_topic
        else:
            # Get first available topic, prefer topics with 'color' or 'rgb' in name
            color_topics = [t for t in latest_frames.keys() if 'color' in t.lower() or 'rgb' in t.lower()]
            if color_topics:
                topic_name = color_topics[0]
            else:
                topic_name = next(iter(latest_frames))
        
        # Get frame with thread safety
        if topic_name in frame_locks:
            with frame_locks[topic_name]:
                if topic_name in latest_frames:
                    frame_data = latest_frames[topic_name]
                    if 'image' in frame_data:
                        return frame_data['image'].copy()
        
        return None
    
    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()
        
        # Store previous lower body targets for comparison
        prev_lower_body_targets = [self.dof_target[i] for i in self.lower_body_indices] if hasattr(self, 'dof_target') else None
        
        # Create copies of dof_pos and dof_vel for policy inference
        masked_dof_pos = np.copy(self.dof_pos)
        masked_dof_vel = np.copy(self.dof_vel)
        
        # If in sine mode or teleop mode, mask upper body positions and velocities with defaults
        # so the policy thinks the arms/head are in their default positions and not moving
        # This prevents upper body movements from affecting lower body policy decisions
        if (self.body_part_control_mode[BodyPart.UPPER_BODY] == UpperBodyControlMode.SINE.value or
            self.body_part_control_mode[BodyPart.UPPER_BODY] == UpperBodyControlMode.TELEOP.value):
            # Get default positions from policy
            default_positions = self.policy.default_dof_pos
            
            # Replace upper body positions with default positions
            for i in self.upper_body_indices:
                masked_dof_pos[i] = default_positions[i]
                masked_dof_vel[i] = 0.0  # Zero velocity
            
            self.logger.debug(f"Masking upper body for policy (mode: {self.body_part_control_mode[BodyPart.UPPER_BODY]})")

        # Get policy inference for all joints
        policy_targets = self.policy.inference(
            time_now=time_now,
            dof_pos=masked_dof_pos,  # Use masked positions
            dof_vel=masked_dof_vel,  # Use masked velocities
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )
        
        # Apply policy targets to lower body
        if self.body_part_control_mode[BodyPart.LOWER_BODY] == "policy":
            for i in self.lower_body_indices:
                self.dof_target[i] = policy_targets[i]
        
        # Check for significant lower body changes that might indicate ankle swaying
        if prev_lower_body_targets is not None:
            current_lower_body_targets = [self.dof_target[i] for i in self.lower_body_indices]
            max_change = max(abs(curr - prev) for curr, prev in zip(current_lower_body_targets, prev_lower_body_targets))
            
            if max_change > 0.05:  # Threshold for significant change
                self.logger.warning(f"Significant lower body change detected: max_change={max_change:.3f}")
                self.logger.warning(f"IMU: base_ang_vel={self.base_ang_vel}, projected_gravity={self.projected_gravity}")
                self.logger.warning(f"Commands: vx={self.remoteControlService.get_vx_cmd()}, vy={self.remoteControlService.get_vy_cmd()}, vyaw={self.remoteControlService.get_vyaw_cmd()}")
                
                # Check if face tracking is active
                face_tracking_active = (FACE_TRACKING and 
                                      self.face_tracker is not None and 
                                      self.body_part_control_mode[BodyPart.UPPER_BODY] == UpperBodyControlMode.TELEOP.value)
                self.logger.warning(f"Face tracking active: {face_tracking_active}")
        
        # Apply appropriate control to upper body based on mode
        upper_body_mode = self.body_part_control_mode[BodyPart.UPPER_BODY]
        
        if upper_body_mode == UpperBodyControlMode.TELEOP.value:
            # Apply manual/teleop targets to upper body
            for i, idx in enumerate(self.upper_body_indices):
                self.dof_target[idx] = self.manual_upper_body_positions[i]
            
            # Override head control with face tracking if enabled
            if FACE_TRACKING and self.face_tracker is not None:
                camera_frame = self.get_camera_frame(self.preferred_camera_topic)
                if camera_frame is not None:
                    # Store previous head targets for comparison
                    prev_head_yaw = self.dof_target[0]
                    prev_head_pitch = self.dof_target[1]
                    
                    # Get face tracking head targets
                    head_yaw_target, head_pitch_target = self.face_tracker.process_frame_and_get_head_targets(camera_frame)
                    
                    # Override head joint targets (indices 0 and 1 in upper_body_indices)
                    self.dof_target[0] = head_yaw_target   # Head yaw
                    self.dof_target[1] = head_pitch_target # Head pitch
                    
                    # Check for significant head movement that might affect stability
                    head_yaw_delta = abs(head_yaw_target - prev_head_yaw)
                    head_pitch_delta = abs(head_pitch_target - prev_head_pitch)
                    
                    if head_yaw_delta > 0.1 or head_pitch_delta > 0.1:
                        self.logger.warning(f"Large head movement detected: yaw_delta={head_yaw_delta:.3f}, pitch_delta={head_pitch_delta:.3f}")
                        self.logger.warning(f"IMU state: base_ang_vel={self.base_ang_vel}, projected_gravity={self.projected_gravity}")
                        self.logger.warning(f"Lower body targets: {[self.dof_target[i] for i in self.lower_body_indices]}")
                    
                    self.logger.debug(f"Face tracking: head_yaw={head_yaw_target:.3f}, head_pitch={head_pitch_target:.3f}")
                
        elif upper_body_mode == UpperBodyControlMode.SINE.value:
            # Apply sine wave targets to upper body
            # The sine positions should be updated externally by the upper_body_controller
            for i, idx in enumerate(self.upper_body_indices):
                self.dof_target[idx] = self.sine_upper_body_positions[i]
                
        else:  # Default to policy control
            # Use policy for upper body
            for i in self.upper_body_indices:
                self.dof_target[i] = policy_targets[i]

        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            # Apply different filtering based on body part
            # Lower body (policy controlled) - standard filtering
            for i in self.lower_body_indices:
                self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.8 + self.dof_target[i] * 0.2
            
            # Upper body (VR controlled) - more filtering for smoothness
            for i in self.upper_body_indices:
                joint_index = self.upper_body_indices.index(i)  # Get the position in upper_body_indices
                
                # Special case: Head joints (indices 0 and 1) with face tracking need extra smoothing
                if (joint_index in [0, 1] and  # Head yaw (0) and pitch (1) joints
                    FACE_TRACKING and 
                    self.face_tracker is not None and
                    self.body_part_control_mode[BodyPart.UPPER_BODY] == "teleop"):
                    # Much more aggressive filtering for head joints during face tracking
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.99 + self.dof_target[i] * 0.01
                elif self.body_part_control_mode[BodyPart.UPPER_BODY] == "teleop":
                    # Stronger filtering for teleop control to be smoother
                    # Higher first coefficient (0.9) means more of the previous value is retained
                    # resulting in smoother, less jerky movements
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.9 + self.dof_target[i] * 0.1
                else:
                    # Standard filtering for policy control
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.8 + self.dof_target[i] * 0.2

            # print(f"Filtered dof target: {self.filtered_dof_target}")
            # Set position targets for all joints
            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            # Use series-parallel conversion for torque to avoid non-linearity
            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                
                # Calculate torque - stiffness values for arms are already reduced in the config
                self.low_cmd.motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * 
                    self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                self.low_cmd.motor_cmd[i].kp = 0.0

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    def _setup_ros2_camera_system(self):
        """Setup ROS2 camera system for face tracking"""
        try:
            # Initialize ROS2 if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create a temporary node for topic discovery
            temp_node = rclpy.create_node('face_tracking_topic_discovery')
            
            # Discover camera topics
            camera_topics = discover_camera_topics(temp_node)
            
            # Clean up temp node
            temp_node.destroy_node()
            
            if not camera_topics:
                self.logger.warning("No camera topics found for face tracking")
                return
                
            self.logger.info(f"Found {len(camera_topics)} camera topics for face tracking:")
            for topic_name, topic_type in camera_topics:
                self.logger.info(f"  - {topic_name} ({topic_type})")
            
            # Filter topics based on preference
            selected_topics = []
            if self.preferred_camera_topic:
                # Look for the preferred topic
                for topic_name, topic_type in camera_topics:
                    if self.preferred_camera_topic in topic_name:
                        selected_topics.append((topic_name, topic_type))
                        break
                        
            if not selected_topics:
                # Use color/rgb topics if available, otherwise use first available
                color_topics = [(name, ttype) for name, ttype in camera_topics 
                               if 'color' in name.lower() or 'rgb' in name.lower()]
                if color_topics:
                    selected_topics = [color_topics[0]]  # Use first color topic
                else:
                    selected_topics = [camera_topics[0]]  # Use first available topic
            
            # Create camera subscribers
            self.camera_subscribers = []
            for topic_name, topic_type in selected_topics:
                try:
                    subscriber = FaceTrackingCameraSubscriber(topic_name, topic_type)
                    self.camera_subscribers.append(subscriber)
                    self.logger.info(f"Created camera subscriber for {topic_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create subscriber for {topic_name}: {e}")
            
            if not self.camera_subscribers:
                self.logger.warning("No camera subscribers created")
                return
                
            # Create ROS2 executor and thread
            self.ros_executor = MultiThreadedExecutor()
            for subscriber in self.camera_subscribers:
                self.ros_executor.add_node(subscriber)
                
            self.ros_thread = threading.Thread(target=self.ros_executor.spin, daemon=True)
            self.ros_thread.start()
            
            self.ros_initialized = True
            self.logger.info("ROS2 camera system for face tracking initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup ROS2 camera system: {e}")
            self.ros_initialized = False

    def reset_face_tracking(self):
        """Reset face tracking head position to center"""
        if self.face_tracker is not None:
            self.face_tracker.reset_head_position()
        else:
            self.logger.warning("Face tracker not initialized")
    
    def set_face_tracking_enabled(self, enabled):
        """Enable or disable face tracking dynamically
        
        Args:
            enabled: True to enable face tracking, False to disable
        """
        global FACE_TRACKING
        
        if enabled and self.face_tracker is None:
            # Try to initialize face tracker if not already done
            try:
                self.face_tracker = FaceTracker()
                FACE_TRACKING = True
                self.logger.info("Face tracking enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize face tracker: {e}")
                FACE_TRACKING = False
        elif not enabled:
            FACE_TRACKING = False
            self.logger.info("Face tracking disabled")
        else:
            FACE_TRACKING = enabled
            self.logger.info(f"Face tracking {'enabled' if enabled else 'disabled'}")

    def get_actual_hardware_joint_angles(self) -> Optional[np.ndarray]:
        """
        Retrieves the current actual joint angles for the 11 upper body joints
        from the latest low_state data.

        The order of joints returned is:
        Head Yaw, Head Pitch,
        L Shoulder Pitch, L Shoulder Roll, L Shoulder Yaw, L Elbow,
        R Shoulder Pitch, R Shoulder Roll, R Shoulder Yaw, R Elbow,
        Waist Yaw

        Returns:
            np.ndarray: An array of 11 joint angles in radians, or None if
                        upper_body_indices is not properly defined.
        """
        if not hasattr(self, 'upper_body_indices') or not self.upper_body_indices:
            self.logger.error("upper_body_indices not defined in Controller. Cannot get actual hardware joint angles.")
            return None
        if not hasattr(self, 'dof_pos_latest'):
            self.logger.error("dof_pos_latest not available in Controller.")
            return None

        actual_angles = np.zeros(len(self.upper_body_indices), dtype=np.float32)
        with self.publish_lock: # Protect access to dof_pos_latest
            for i, sdk_idx in enumerate(self.upper_body_indices):
                if 0 <= sdk_idx < len(self.dof_pos_latest):
                    actual_angles[i] = self.dof_pos_latest[sdk_idx]
                else:
                    self.logger.error(f"SDK index {sdk_idx} for upper body joint {i} is out of bounds for dof_pos_latest (len: {len(self.dof_pos_latest)}).")
                    # Return None or partial data, or raise error? For now, fills with 0 and logs.
                    actual_angles[i] = 0.0 # Or handle error more strictly
        
        # print(f"Actual hardware joint angles: {actual_angles}")
        return actual_angles

if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    parser.add_argument("--no-face-tracking", action="store_true", help="Disable face tracking even if FACE_TRACKING=True")
    parser.add_argument("--face-tracking-camera", type=str, default=None, help="Preferred camera topic for face tracking")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)
    
    # Override face tracking if disabled via command line
    if args.no_face_tracking:
        FACE_TRACKING = False

    print(f"Starting custom controller, connecting to {args.net} ...")
    print(f"Upper body control mode: {UPPER_BODY_CONTROL_MODE}")
    print(f"Face tracking: {'ENABLED' if FACE_TRACKING else 'DISABLED'}")
    if FACE_TRACKING:
        print("  - Face tracking will override head control in teleop mode")
        print("  - Robot will track the largest detected face")
        print("  - Automatically discovers ROS2 camera topics")
        if args.face_tracking_camera:
            print(f"  - Preferred camera topic: {args.face_tracking_camera}")
        else:
            print("  - Will use first available color/RGB camera topic")
    
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file, args.face_tracking_camera) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()

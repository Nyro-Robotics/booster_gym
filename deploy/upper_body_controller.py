#!/usr/bin/env python3
"""
Upper Body Controller for Booster Robot

This controller provides multiple modes for controlling the robot's upper body:
1. Policy control - Both upper and lower body controlled by the policy
2. Sine wave control - Rhythmic sinusoidal movements for the upper body
3. VR/Teleop control - Manual control via VR hand tracking (mock data for now)

The lower body (legs) is always controlled by the policy for locomotion.
"""

import numpy as np
import time
import argparse
import os
import signal
import sys
import json
import threading
import websocket
import yaml
from deploy import Controller, BodyPart, UpperBodyControlMode, UPPER_BODY_CONTROL_MODE

from booster_robotics_sdk_python import ChannelFactory

# Control mode parameters
# Don't overwrite this, but if you must options are: "policy", "sine", "teleop"
DEFAULT_CONTROL_MODE = UPPER_BODY_CONTROL_MODE  # Default from deploy.py

# WebSocket parameters
DEFAULT_WEBSOCKET_URL = "ws://localhost:8765"  # Default WebSocket URL for arm tracking
USE_MOCK_TRACKING = True if UPPER_BODY_CONTROL_MODE == "sine" else False  # Whether to use mock sine wave tracking instead of actual VR teleop via WebSocket

# Sine wave control parameters
SINE_CONTROL_AMPLITUDE = 0.15      # Base amplitude for sine wave movements
SINE_CONTROL_FREQUENCY = 1.5      # Base frequency for sine wave movements (Hz)

class WebSocketArmTrackingClient:
    """WebSocket client for receiving arm tracking commands.
    
    This client connects to a WebSocket server to receive arm joint positions
    that have already had inverse kinematics applied.
    """
    def __init__(self, websocket_url="ws://localhost:8765"):
        self.websocket_url = websocket_url
        # Store positions for all 11 upper body joints (head, arms, waist)
        # But we'll only receive 10 from WebSocket (head + arms, no waist)
        self.joint_positions = np.zeros(11)  
        self.connected = False
        self.ws = None
        self.ws_thread = None
        self.lock = threading.Lock()
        
        # Initialize with default positions
        self._set_default_positions()
    
    def _set_default_positions(self):
        """Set default joint positions when no data is available."""
        # Default neutral position for all joints
        with self.lock:
            self.joint_positions = np.zeros(11)  # All 11 upper body joints
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages with joint positions."""
        try:
            # Parse the JSON message
            data = json.loads(message)
            
            # Expected format: {"joint_positions": [j0, j1, j2, ...]} 
            # Where the array contains the 8 arm joint positions (4 for each arm)
            # plus any additional joints like head and waist
            if "joint_positions" in data and isinstance(data["joint_positions"], list):
                arm_positions = data["joint_positions"]
                
                # Update joint positions with lock to avoid race conditions
                with self.lock:
                    # Handle different formats of incoming data
                    
                    # If we have exactly 10 joints (head + arms)
                    if len(arm_positions) == 10:  
                        # Copy the 10 joints (head + arms) to our array
                        # Keep the waist position unchanged (index 10)
                        self.joint_positions[:10] = np.array(arm_positions)
                        
                    # If we have just 8 arm joints (no head)
                    elif len(arm_positions) == 8:  
                        # Update only arm positions (indices 2-9)
                        # Keep head and waist positions unchanged
                        self.joint_positions[2:10] = np.array(arm_positions)
                        
                    # If we have just head joints (2)
                    elif len(arm_positions) == 2:
                        # Update only head positions
                        # Keep arm and waist positions unchanged
                        self.joint_positions[0:2] = np.array(arm_positions)
                        
                    else:
                        print(f"Warning: Received unexpected number of joint positions: {len(arm_positions)}")
                    
                    print(f"Received joint positions: {self.joint_positions}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON message: {message}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        print(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.connected = False
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        print(f"WebSocket connection established to {self.websocket_url}")
        self.connected = True
    
    def _websocket_thread(self):
        """Run the WebSocket client in a separate thread."""
        self.ws = websocket.WebSocketApp(self.websocket_url,
                                        on_open=self._on_open,
                                        on_message=self._on_message,
                                        on_error=self._on_error,
                                        on_close=self._on_close)
        self.ws.run_forever()
    
    def connect(self):
        """Connect to the WebSocket server."""
        if self.ws_thread is None or not self.ws_thread.is_alive():
            print(f"Connecting to WebSocket server at {self.websocket_url}...")
            self.ws_thread = threading.Thread(target=self._websocket_thread)
            self.ws_thread.daemon = True
            self.ws_thread.start()
    
    def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.ws is not None:
            self.ws.close()
        if self.ws_thread is not None and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=1.0)
        self.connected = False
    
    def get_joint_positions(self):
        """Get the current joint positions.
        
        Returns:
            Array of joint positions for upper body joints (head, arms, waist)
        """
        with self.lock:
            return np.copy(self.joint_positions)


class MockArmTrackingSystem:
    """Mock arm tracking system for testing without WebSocket.
    
    This provides simulated arm joint positions when no WebSocket
    connection is available.
    """
    def __init__(self, config=None):
        self.joint_positions = np.zeros(11)  # 11 upper body joints
        self.start_time = time.time()
        
        # Use the pre-loaded config if provided
        if config is not None and isinstance(config, dict):
            if 'common' in config and 'default_qpos' in config['common']:
                # Get the upper body joint positions (first 11 values)
                self.default_positions = np.array(config['common']['default_qpos'][:11], dtype=np.float32)
                print(f"Using pre-loaded default positions: {self.default_positions}")
            else:
                print("Warning: Could not find default_qpos in provided config, using hardcoded defaults")
                # Fallback to hardcoded defaults
                self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0])
        else:
            # Raise error
            raise ValueError("No config provided and could not find default_qpos in config")
        
        # Initialize with default positions
        self.joint_positions = np.copy(self.default_positions)
        
        # Sine wave parameters
        self.sine_params = {
            "amplitudes": {
                "head_yaw": SINE_CONTROL_AMPLITUDE * 0.7,      # Head yaw amplitude
                "head_pitch": SINE_CONTROL_AMPLITUDE * 0.3,    # Head pitch amplitude
                "left_shoulder_pitch": SINE_CONTROL_AMPLITUDE * 2.0,  # Left shoulder pitch amplitude
                "left_shoulder_roll": SINE_CONTROL_AMPLITUDE * 0.2,   # Left shoulder roll amplitude
                "left_shoulder_yaw": SINE_CONTROL_AMPLITUDE * 0.6,    # Left shoulder yaw amplitude
                "left_elbow": SINE_CONTROL_AMPLITUDE * 0.7,           # Left elbow amplitude
                "right_shoulder_pitch": SINE_CONTROL_AMPLITUDE   * 1.0,           # Right arm amplitude
                "right_shoulder_roll": SINE_CONTROL_AMPLITUDE * 0.2,           # Right arm amplitude
                "right_shoulder_yaw": SINE_CONTROL_AMPLITUDE * 0.6,           # Right arm amplitude
                "right_elbow": SINE_CONTROL_AMPLITUDE * 0.7,           # Right arm amplitude
                "waist": SINE_CONTROL_AMPLITUDE * 0.5,         # Waist amplitude (disabled)
            },
            "frequencies": {
                "head_yaw": SINE_CONTROL_FREQUENCY * 0.7,       # Head yaw frequency (Hz)
                "head_pitch": SINE_CONTROL_FREQUENCY * 0.5,     # Head pitch frequency (Hz)
                "left_shoulder_pitch": SINE_CONTROL_FREQUENCY * 1.0,  # Left arm frequency (Hz)
                "left_shoulder_roll": SINE_CONTROL_FREQUENCY * 0.8,   # Right arm frequency (Hz)
                "left_shoulder_yaw": SINE_CONTROL_FREQUENCY * 0.6,    # Waist frequency (Hz)
                "left_elbow": SINE_CONTROL_FREQUENCY * 0.7,           # Left elbow frequency (Hz)
                "right_shoulder_pitch": SINE_CONTROL_FREQUENCY * 1.0,  # Left arm frequency (Hz)
                "right_shoulder_roll": SINE_CONTROL_FREQUENCY * 0.8,   # Right arm frequency (Hz)
                "right_shoulder_yaw": SINE_CONTROL_FREQUENCY * 0.6,    # Waist frequency (Hz)
                "right_elbow": SINE_CONTROL_FREQUENCY * 0.7,           # Left elbow frequency (Hz)
                "waist": SINE_CONTROL_FREQUENCY * 0.5,          # Waist frequency (Hz)
            },
            "phase_shifts": {
                "head_yaw": 0.0,       # Head yaw phase shift
                "head_pitch": 0.5,     # Head pitch phase shift
                "left_arm": 0.0,       # Left arm phase shift
                "right_arm": np.pi,    # Right arm phase shift (opposite to left)
                "waist": 0.25,         # Waist phase shift
            }
        }
        
    def update(self):
        """Update mock joint positions with a sine wave pattern using parameters."""
        t = time.time() - self.start_time
        params = self.sine_params
        
        # Helper function to calculate sine wave position
        def calc_sine_pos(joint_idx, param_name, phase_key, phase_offset=0.0):
            amp = params["amplitudes"][param_name]
            freq = params["frequencies"][param_name]
            phase = params["phase_shifts"][phase_key] + phase_offset
            return self.default_positions[joint_idx] + amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Head movements
        self.joint_positions[0] = calc_sine_pos(0, "head_yaw", "head_yaw")
        self.joint_positions[1] = calc_sine_pos(1, "head_pitch", "head_pitch")
        
        # Left arm joints
        self.joint_positions[2] = calc_sine_pos(2, "left_shoulder_pitch", "left_arm")
        self.joint_positions[3] = calc_sine_pos(3, "left_shoulder_roll", "left_arm", 0.2)
        self.joint_positions[4] = calc_sine_pos(4, "left_shoulder_yaw", "left_arm", 0.4)
        self.joint_positions[5] = calc_sine_pos(5, "left_elbow", "left_arm", 0.6)
        
        # Right arm joints
        self.joint_positions[6] = calc_sine_pos(6, "right_shoulder_pitch", "right_arm")
        self.joint_positions[7] = calc_sine_pos(7, "right_shoulder_roll", "right_arm", 0.2)
        self.joint_positions[8] = calc_sine_pos(8, "right_shoulder_yaw", "right_arm", 0.4)
        self.joint_positions[9] = calc_sine_pos(9, "right_elbow", "right_arm", 0.6)
        
        # Waist movement
        self.joint_positions[10] = calc_sine_pos(10, "waist", "waist")
    
    def get_joint_positions(self):
        """Get the current joint positions.
        
        Returns:
            Array of joint positions for upper body joints (head, arms, waist)
        """
        return np.copy(self.joint_positions)


def signal_handler(sig, frame):
    print("\nShutting down...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")

    args = parser.parse_args()
    
    cfg_file = os.path.join("configs", args.config)
    control_mode = DEFAULT_CONTROL_MODE

    print("="*50)
    print(f"CONTROL MODE: {control_mode.upper()}")
    print("="*50)
    print(f"Starting upper body controller in {control_mode} mode, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)
    
    # Initialize the controller and get its config
    controller = Controller(cfg_file)
    # Get access to the already loaded config from the controller
    config = controller.cfg
    
    # Initialize tracking system based on mode and arguments
    tracking_system = None
    if control_mode == "teleop":
        if USE_MOCK_TRACKING:
            print("Using mock arm tracking system")
            tracking_system = MockArmTrackingSystem(config=config)
        else:
            print(f"Connecting to WebSocket server at {DEFAULT_WEBSOCKET_URL}")
            tracking_system = WebSocketArmTrackingClient(DEFAULT_WEBSOCKET_URL)
            tracking_system.connect()
    
    # With the already initialized controller
    with controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        
        # Start in custom mode
        controller.start_custom_mode_conditionally()
        
        # Start the RL gait for lower body
        controller.start_rl_gait_conditionally()
        
        # Set upper body control mode
        controller.set_body_part_control_mode(BodyPart.UPPER_BODY, control_mode)
        
        if control_mode == "policy":
            print("Robot running with policy control for both upper and lower body")
        elif control_mode == "sine":
            print("Robot running with sine wave control for upper body")
            # Create a sine wave controller with the pre-loaded config
            sine_controller = MockArmTrackingSystem(config=config)
            # Set the sine controller in the main controller
            controller.set_sine_controller(sine_controller)
        elif control_mode == "teleop":
            print("Robot running with teleop control for upper body")
            if USE_MOCK_TRACKING:
                print("Using simulated arm movements")
            else:
                print(f"Receiving arm tracking data from WebSocket at {DEFAULT_WEBSOCKET_URL}")
                print("Ensure your tracking system is sending joint positions in the correct format")
        
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                # Handle different control modes
                if control_mode == "teleop" and tracking_system is not None:
                    # Update tracking data if using mock system
                    if USE_MOCK_TRACKING:
                        tracking_system.update()
                    
                    # Get joint positions from tracking system
                    upper_body_positions = tracking_system.get_joint_positions()
                    
                    # Update controller with new upper body positions
                    controller.set_upper_body_positions(upper_body_positions)
                    
                elif control_mode == "sine":
                    # Update sine wave positions
                    sine_controller.update()
                    
                    # Get joint positions from sine controller
                    sine_positions = sine_controller.get_joint_positions()
                    
                    # Update controller with new sine wave positions
                    controller.set_sine_upper_body_positions(sine_positions)
                
                # Run policy for lower body (and possibly upper body if in policy mode)
                controller.run()
                
                # Sleep to maintain update rate
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            
        finally:
            # Clean up resources
            if control_mode == "teleop" and tracking_system is not None and not USE_MOCK_TRACKING:
                tracking_system.disconnect()
                print("WebSocket connection closed")


if __name__ == "__main__":
    # Debug prints before anything else
    print("\n\n")
    print("*"*80)
    print("STARTING UPPER BODY CONTROLLER")
    print(f"DEFAULT_CONTROL_MODE = {DEFAULT_CONTROL_MODE}")
    print(f"UPPER_BODY_CONTROL_MODE from deploy.py = {UPPER_BODY_CONTROL_MODE}")
    print("*"*80)
    print("\n\n")
    
    signal.signal(signal.SIGINT, signal_handler)
    main()

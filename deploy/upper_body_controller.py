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
import logging
from deploy import Controller, BodyPart, UpperBodyControlMode, UPPER_BODY_CONTROL_MODE

from booster_robotics_sdk_python import ChannelFactory

# Control mode parameters
# Don't overwrite this, but if you must options are: "policy", "sine", "teleop"
DEFAULT_CONTROL_MODE = UPPER_BODY_CONTROL_MODE  # Default from deploy.py

# WebSocket parameters
DEFAULT_WEBSOCKET_URL = "ws://10.20.0.169:8765"  # Default WebSocket URL for arm tracking
USE_MOCK_TRACKING = True if UPPER_BODY_CONTROL_MODE == "sine" else False  # Whether to use mock sine wave tracking instead of actual VR teleop via WebSocket

# Sine wave control parameters
SINE_CONTROL_AMPLITUDE = 0.15      # Base amplitude for sine wave movements
SINE_CONTROL_FREQUENCY = 1.5      # Base frequency for sine wave movements (Hz)

class WebSocketArmTrackingClient:
    """WebSocket client for receiving arm tracking commands.
    
    When USE_MOCK_TRACKING is False, this class facilitates communication
    with an external controller (like HardwareInterface from teleoperator.py).
    It receives target joint positions from the external controller and
    sends observed/actual joint positions from this robot's hardware.
    """
    def __init__(self, websocket_url=" ws://10.20.0.169:8765", main_controller_ref=None, config=None):
        self.websocket_url = websocket_url
        self.main_controller_ref = main_controller_ref
        
        # self.joint_positions is now for OBSERVED positions from hardware
        self.joint_positions = np.zeros(10)
        # self.commanded_joint_positions is for TARGET positions from HardwareInterface
        self.commanded_joint_positions = np.zeros(10)
        # Set to False initially; will be set to True once valid commands are received
        self.has_received_valid_commands = False
        
        # Initialize default positions from config
        if config is not None and isinstance(config, dict):
            if 'common' in config and 'default_qpos' in config['common']:
                # Get the upper body joint positions (first 10 values)
                if len(config['common']['default_qpos']) >= 10:
                    self.default_positions = np.array(config['common']['default_qpos'][:10], dtype=np.float32)
                    logger.info(f"WebSocketArmTrackingClient: Using default positions from config: {self.default_positions}")
                else:
                    logger.warning(f"WebSocketArmTrackingClient: default_qpos in config has fewer than 10 values. Using hardcoded defaults.")
                    self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5])
            else:
                logger.warning("WebSocketArmTrackingClient: Could not find default_qpos in config, using hardcoded defaults")
                self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5])
        else:
            logger.warning("WebSocketArmTrackingClient: No config provided, using hardcoded defaults")
            self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5])
        
        # Initialize commanded positions with default positions
        self.commanded_joint_positions = np.copy(self.default_positions)
        
        self.connected = False
        self.ws = None
        self.ws_thread = None
        self.sender_thread = None # Thread for periodic sending
        self.lock = threading.Lock()
        self.last_message_time = None # Added for frequency calculation
        self.message_count = 0 # Added for periodic frequency logging
        
        # Define joint limits (in radians) for safety validation
        # Based on robot hardware specifications
        DEG_TO_RAD = np.pi / 180.0
        self.joint_limits = {
            # [min, max] for each joint in radians
            0: [-58 * DEG_TO_RAD, 58 * DEG_TO_RAD],    # Head Yaw Joint
            1: [-18 * DEG_TO_RAD, 47 * DEG_TO_RAD],    # Head Pitch Joint
            2: [-188 * DEG_TO_RAD, 68 * DEG_TO_RAD],   # Left Shoulder Pitch Joint
            3: [-94 * DEG_TO_RAD, 88 * DEG_TO_RAD],    # Left Shoulder Roll Joint
            4: [-128 * DEG_TO_RAD, 128 * DEG_TO_RAD],  # Left Shoulder Yaw Joint
            5: [-120 * DEG_TO_RAD, 2 * DEG_TO_RAD],    # Left Elbow Joint
            6: [-188 * DEG_TO_RAD, 68 * DEG_TO_RAD],   # Right Shoulder Pitch Joint
            7: [-88 * DEG_TO_RAD, 94 * DEG_TO_RAD],    # Right Shoulder Roll Joint
            8: [-128 * DEG_TO_RAD, 128 * DEG_TO_RAD],  # Right Shoulder Yaw Joint
            9: [-2 * DEG_TO_RAD, 120 * DEG_TO_RAD],    # Right Elbow Joint
        }
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages with joint positions."""
        try:
            # Calculate frequency
            current_time = time.time()
            self.message_count += 1
            if self.last_message_time is not None:
                time_delta = current_time - self.last_message_time
                if time_delta > 0:
                    frequency = 1.0 / time_delta
                    if self.message_count % 100 == 0: # Log every 100 messages
                        logger.info(f"WebSocket message frequency: {frequency:.2f} Hz")
            self.last_message_time = current_time

            logger.debug(f"Raw WebSocket message: {message}")
            data = json.loads(message)
            
            # Process TARGET commands from HardwareInterface (relayed by server)
            if data.get("type") == "control_command_list" and "target_positions" in data:
                command_input_positions = data["target_positions"]
                logger.debug(f"Received command 'target_positions' list via 'control_command_list': {command_input_positions}")
                
                if not isinstance(command_input_positions, list):
                    logger.warning(f"'target_positions' in 'control_command_list' is not a list: {type(command_input_positions)}")
                    return # Or handle error appropriately

                with self.lock:
                    num_received_commands = len(command_input_positions)
                    # Store in self.commanded_joint_positions after validation and clipping
                    # This logic handles various expected lengths of command_input_positions
                    
                    if num_received_commands == 11: # head, 8 arm, waist - This case should ideally not be hit if sender is 10-joint aware
                        logger.warning(f"Received 11 target_positions, but system is configured for 10. Will use first 10.")
                        received_commands_clipped = np.array(command_input_positions[:10], dtype=np.float64) # Take first 10
                        for i in range(10): # Process only 10
                            if i in self.joint_limits:
                                min_val, max_val = self.joint_limits[i]
                                clipped_val = np.clip(received_commands_clipped[i], min_val, max_val)
                                if clipped_val != command_input_positions[i]: # Compare with original command_input_positions[i]
                                    logger.warning(f"Warning: Command for joint {i} value {command_input_positions[i]} was outside limits [{min_val}, {max_val}], clipped to {clipped_val}")
                                received_commands_clipped[i] = clipped_val
                        self.commanded_joint_positions[:] = received_commands_clipped[:] # Assign 10 values
                        self.has_received_valid_commands = True
                    elif num_received_commands == 10: # head, 8 arm
                        received_commands_clipped = np.array(command_input_positions, dtype=np.float64)
                        for i in range(10):
                             if i in self.joint_limits: # Check if limit exists for this joint index
                                 min_val, max_val = self.joint_limits[i]
                                 clipped_val = np.clip(received_commands_clipped[i], min_val, max_val)
                                 if clipped_val != command_input_positions[i]:
                                     logger.warning(f"Warning: Command for joint {i} value {command_input_positions[i]} was outside limits [{min_val}, {max_val}], clipped to {clipped_val}")
                                 received_commands_clipped[i] = clipped_val
                        self.commanded_joint_positions[:10] = received_commands_clipped
                        self.has_received_valid_commands = True
                        # Waist joint (index 10) in self.commanded_joint_positions remains unchanged or at its default
                    elif num_received_commands == 8: # 8 arm only
                        received_commands_clipped = np.array(command_input_positions, dtype=np.float64)
                        for i in range(8):
                            joint_idx = i + 2 # Offset for arm joints
                            if joint_idx in self.joint_limits: # Check if limit exists
                                min_val, max_val = self.joint_limits[joint_idx]
                                clipped_val = np.clip(received_commands_clipped[i], min_val, max_val)
                                if clipped_val != command_input_positions[i]:
                                     logger.warning(f"Warning: Command for Arm joint {joint_idx} value {command_input_positions[i]} was outside limits [{min_val}, {max_val}], clipped to {clipped_val}")
                                received_commands_clipped[i] = clipped_val
                        self.commanded_joint_positions[2:10] = received_commands_clipped
                        self.has_received_valid_commands = True
                        # Head (0,1) and Waist (10) in self.commanded_joint_positions remain unchanged
                    elif num_received_commands == 2: # 2 head only
                        received_commands_clipped = np.array(command_input_positions, dtype=np.float64)
                        for i in range(2):
                            if i in self.joint_limits: # Check if limit exists
                                min_val, max_val = self.joint_limits[i]
                                clipped_val = np.clip(received_commands_clipped[i], min_val, max_val)
                                if clipped_val != command_input_positions[i]: # Use original for comparison
                                    logger.warning(f"Warning: Command for Head joint {i} value {command_input_positions[i]} was outside limits [{min_val}, {max_val}], clipped to {clipped_val}")
                                received_commands_clipped[i] = clipped_val
                        self.commanded_joint_positions[0:2] = received_commands_clipped
                        self.has_received_valid_commands = True
                        # Arm (2-9) and Waist (10) in self.commanded_joint_positions remain unchanged
                    else:
                        logger.warning(f"Warning: Received unexpected number of 'target_positions': {num_received_commands} in 'control_command_list'")
                    
                    logger.debug(f"Stored commanded_joint_positions (after safety clipping from 'target_positions'): {self.commanded_joint_positions.tolist()}")

            # If this client receives observed_joint_positions, it's likely an echo or from another UBC.
            # For a 2-client setup (HI <-> UBC), UBC should ignore this.
            elif "observed_joint_positions" in data:
                logger.debug(f"UBC received an 'observed_joint_positions' message. Ignoring. Message: {message[:150]}")
                pass # Explicitly do nothing if it's an observation message.
            
            else:
                logger.warning(f"UBC received WebSocket message not matching expected command structures: {message}")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        self.connected = False
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info(f"WebSocket connection established to {self.websocket_url}")
        self.connected = True # Crucial: set connected True only after ws is confirmed open
    
    def _websocket_thread(self):
        """Run the WebSocket client in a separate thread."""
        # Initialize WebSocketApp here so self.ws is valid for sender thread checks
        self.ws = websocket.WebSocketApp(self.websocket_url,
                                        on_open=self._on_open,
                                        on_message=self._on_message,
                                        on_error=self._on_error,
                                        on_close=self._on_close)
        self.ws.run_forever()
        # run_forever exits when ws is closed or an unhandled error occurs that stops it.
        # Ensure connected is False if we exit this way.
        self.connected = False 
        # logger.debug("WebSocket receiver thread stopped.") # Optional debug

    def _periodic_sender(self):
        """Periodically send a message to the server to request joint state updates."""
        # logger.debug("Periodic sender thread started.") # Optional debug
        while self.connected: # Loop while connection is intended to be active
            # Check if ws exists, ws.sock exists, and ws.sock.connected is True
            # self.connected can be True briefly before self.ws is fully set up by _on_open
            # or after self.ws.close() is called but before _on_close fully runs.
            if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
                try:
                    # If this instance is part of the SERVER logic for HardwareInterface:
                    # Periodically send OBSERVED joint positions
                    if self.main_controller_ref:
                        # The `update_observed_joint_positions_from_hardware()` method, called in the main loop,
                        # updates `self.joint_positions` using `main_controller_ref.get_actual_hardware_joint_angles()`.
                        # This sender then transmits these updated `self.joint_positions`.
                        try:
                            with self.lock: # Protect access to self.joint_positions
                                angles_to_send = np.copy(self.joint_positions)
                            
                            self.send_joint_positions(angles_to_send) # This sends {"observed_joint_positions": ...}
                            # logger.debug(f"Periodic_sender sent observed_joint_positions: {angles_to_send.tolist()}")

                        except AttributeError as e:
                            logger.error(f"WebSocketArmTrackingClient: main_controller_ref might be missing methods, or error accessing it: {e}")
                        except Exception as e:
                            logger.error(f"WebSocketArmTrackingClient: Error getting/sending actual hardware angles: {e}")
                    else:
                        # Fallback or original ping if not acting as server-part-for-HardwareInterface
                        ping_message = json.dumps({"action": "request_joint_states", "timestamp": time.time()})
                        self.ws.send(ping_message)

                except websocket.WebSocketConnectionClosedException:
                    logger.info("UBC Sender: WebSocket connection closed. Stopping periodic send.")
                    self.connected = False # Ensure loop terminates
                    break 
                except Exception as e:
                    logger.error(f"UBC Sender: Error sending periodic message: {e}. Assuming connection lost.")
                    self.connected = False # Stop trying if a persistent error occurs, flag connection as down.
                    break # Exit the sender loop
            time.sleep(0.01) # Send request every 10ms (100Hz). Adjust frequency as needed.
        # logger.debug("Periodic sender thread stopped.") # Optional debug

    def connect(self):
        """Connect to the WebSocket server and start periodic sender."""
        if not (self.ws_thread and self.ws_thread.is_alive()): # Check if already trying/connected
            logger.info(f"Connecting to WebSocket server at {self.websocket_url}...")
            
            # Reset connected state before attempting
            self.connected = False 

            self.ws_thread = threading.Thread(target=self._websocket_thread)
            self.ws_thread.daemon = True
            
            self.ws_thread.start() # This thread will run _websocket_thread, which initializes self.ws and calls _on_open
            
            # Wait for the ws_thread to establish connection and set self.connected = True via _on_open
            connection_establishment_timeout = 5.0  # seconds
            start_wait_time = time.time()
            logger.info("WebSocketArmTrackingClient: Waiting for connection to be established...")
            while not self.connected and (time.time() - start_wait_time) < connection_establishment_timeout:
                if not self.ws_thread.is_alive():
                    logger.warning("WebSocketArmTrackingClient: ws_thread terminated prematurely during connection attempt. Sender not started.")
                    self.ws = None # Ensure ws is cleared if thread died before ws was set
                    return # Exit connect method
                time.sleep(0.1) # Poll self.connected state

            if self.connected:
                logger.info("WebSocketArmTrackingClient: Connection established. Starting periodic sender.")
                self.sender_thread = threading.Thread(target=self._periodic_sender)
                self.sender_thread.daemon = True
                self.sender_thread.start() # This will loop and wait for self.connected and self.ws to be ready
            else:
                logger.error(f"WebSocketArmTrackingClient: Failed to connect to {self.websocket_url} within {connection_establishment_timeout}s. Periodic sender not started.")
                # Attempt to clean up the ws_thread if it's still alive but didn't connect
                if self.ws_thread and self.ws_thread.is_alive():
                    if self.ws:
                        try:
                            logger.info("WebSocketArmTrackingClient: Closing WebSocket as connection failed...")
                            self.ws.close() # This should trigger _on_close and let run_forever exit
                        except Exception as e:
                            logger.error(f"WebSocketArmTrackingClient: Error closing WebSocket during failed connect: {e}")
                    # Give ws_thread a chance to exit cleanly
                    self.ws_thread.join(timeout=2.0)
                    if self.ws_thread.is_alive():
                        logger.warning("WebSocketArmTrackingClient: Warning: ws_thread did not exit cleanly after connection failure.")
                self.ws = None # Ensure ws is cleared
    
    def disconnect(self):
        """Disconnect from the WebSocket server."""
        logger.info("Disconnecting WebSocketArmTrackingClient...")
        self.connected = False # Signal sender and receiver loops to stop

        if self.ws is not None:
            try:
                self.ws.close() # This should trigger _on_close in the ws_thread
            except Exception as e:
                logger.error(f"Error during WebSocket close: {e}")
        
        # Join receiver thread
        if self.ws_thread is not None and self.ws_thread.is_alive():
            # logger.debug("Joining WebSocket receiver thread...") # Optional debug
            self.ws_thread.join(timeout=2.0) # Increased timeout slightly
            if self.ws_thread.is_alive():
                logger.warning("Warning: WebSocket receiver thread did not exit cleanly.")
        
        # Join sender thread
        if self.sender_thread is not None and self.sender_thread.is_alive():
            # logger.debug("Joining WebSocket sender thread...") # Optional debug
            self.sender_thread.join(timeout=2.0)
            if self.sender_thread.is_alive():
                logger.warning("Warning: WebSocket sender thread did not exit cleanly.")
        
        self.ws = None # Clear WebSocketApp instance
        logger.info("WebSocketArmTrackingClient disconnected.")

    def send_joint_positions(self, positions):
        """Send joint positions back through the websocket.
        
        Args:
            positions: Array of joint positions for upper body joints.
                       These should be OBSERVED/ACTUAL hardware joint positions.
        """
        if self.connected and self.ws is not None and self.ws.sock is not None and self.ws.sock.connected:
            try:
                # Create a JSON message with the current joint positions
                pos_list = positions.tolist()
                message = json.dumps({
                    "observed_joint_positions": pos_list
                })
                
                # Debug log to verify positions and array length
                logger.debug(f"Streaming joint positions (length: {len(pos_list)}):\\n{pos_list}")
                
                self.ws.send(message)
            except websocket.WebSocketConnectionClosedException:
                logger.info("UBC Sender (send_joint_positions): WebSocket connection closed. Cannot send.")
                # self.connected = False # Let the main receiver/sender threads handle this
            except Exception as e:
                logger.error(f"Error sending joint positions: {e}")
        else:
            logger.debug("Cannot send joint positions: WebSocket not connected or not fully initialized.")

    def get_joint_positions(self):
        """Get the current COMMANDED joint positions.
        
        These are the target positions received from an external controller
        (e.g., HardwareInterface) and stored by _on_message.
        
        Returns:
            Array of COMMANDED joint positions for upper body joints (head, arms, waist),
            guaranteed to be within safe joint limits. If no valid commands have been
            received yet, returns the default positions from the config.
        """
        with self.lock:
            # If no valid commands have been received yet, return default positions
            if not self.has_received_valid_commands:
                logger.debug("No valid commanded positions received yet; returning default positions")
                return np.copy(self.default_positions)
            
            # Return a copy of the commanded positions
            # Safety limits were already applied in _on_message when they were stored.
            return np.copy(self.commanded_joint_positions)

    def update_observed_joint_positions_from_hardware(self):
        """
        This method should be called by the main loop of upper_body_controller.py
        to update the internal `self.joint_positions` with actual readings
        from the robot's hardware via the main_controller_ref.

        It relies on `self.main_controller_ref` (an instance of your `Controller` 
        from `deploy.py`) to have a method that provides these angles.
        """
        if self.main_controller_ref:
            try:
                # This calls the `get_actual_hardware_joint_angles()` method implemented 
                # in the `Controller` class (e.g., in `deploy.py`).
                # That method is responsible for using the robot's SDK (e.g., B1LowStateSubscriber)
                # to read motor positions and map them to the 10 standard upper body joint angles (NO WAIST).
                actual_angles_hw = self.main_controller_ref.get_actual_hardware_joint_angles()

                if actual_angles_hw is not None:
                    if len(actual_angles_hw) == 10: # EXPECTING 10 from deploy.py now
                        with self.lock:
                            self.joint_positions[:] = actual_angles_hw.astype(np.float32)[:]
                        # logger.debug(f"Updated self.joint_positions with actual hardware angles: {self.joint_positions.tolist()}")
                    else:
                        logger.warning(f"Could not update observed_joint_positions: Controller method returned {len(actual_angles_hw)} angles, expected 10.")
                else:
                    logger.warning("Could not update observed_joint_positions: Controller method returned None (perhaps no SDK data yet or error extracting angles).")

            except AttributeError as e:
                logger.error(f"WebSocketArmTrackingClient: `self.main_controller_ref` is missing the method `get_actual_hardware_joint_angles()`. You need to implement this in your Controller class (e.g., in deploy.py). Error: {e}")
            except Exception as e:
                logger.error(f"WebSocketArmTrackingClient: Error updating observed joint positions from hardware: {e}", exc_info=True)
        else:
            logger.warning("WebSocketArmTrackingClient: main_controller_ref not set. Cannot update observed joint positions from hardware.")


class MockArmTrackingSystem:
    """Mock arm tracking system for testing without WebSocket.
    
    This provides simulated arm joint positions when no WebSocket
    connection is available.
    """
    def __init__(self, config=None):
        self.joint_positions = np.zeros(10)  # 10 upper body joints (head + arms)
        self.start_time = time.time()
        
        # Use the pre-loaded config if provided
        if config is not None and isinstance(config, dict):
            if 'common' in config and 'default_qpos' in config['common']:
                # Get the upper body joint positions (first 10 values)
                # Assuming default_qpos in config might still have 11+ values, so slice [0:10]
                if len(config['common']['default_qpos']) >= 10:
                    self.default_positions = np.array(config['common']['default_qpos'][:10], dtype=np.float32)
                    logger.info(f"Using pre-loaded default positions (first 10): {self.default_positions}")
                else:
                    logger.warning(f"Warning: default_qpos in config has fewer than 10 values. Using hardcoded 10-joint defaults.")
                    self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5]) # 10 joints
            else:
                logger.warning("Warning: Could not find default_qpos in provided config, using hardcoded 10-joint defaults")
                # Fallback to hardcoded defaults (10 joints)
                self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5])
        else:
            # Raise error or use hardcoded defaults if config is absolutely necessary
            logger.warning("No config provided for MockArmTrackingSystem. Using hardcoded 10-joint defaults.")
            self.default_positions = np.array([0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5]) # 10 joints
        
        # Initialize with default positions
        self.joint_positions = np.copy(self.default_positions)
        
        # Sine wave parameters
        self.sine_params = {
            "amplitudes": {
                "head_yaw": SINE_CONTROL_AMPLITUDE * 1.7,      # Head yaw amplitude
                "head_pitch": SINE_CONTROL_AMPLITUDE * 0.5,    # Head pitch amplitude
                "left_shoulder_pitch": SINE_CONTROL_AMPLITUDE * 2.0,  # Left shoulder pitch amplitude
                "left_shoulder_roll": SINE_CONTROL_AMPLITUDE * 0.3,   # Left shoulder roll amplitude
                "left_shoulder_yaw": SINE_CONTROL_AMPLITUDE * 0.8,    # Left shoulder yaw amplitude
                "left_elbow": SINE_CONTROL_AMPLITUDE * 0.7,           # Left elbow amplitude
                "right_shoulder_pitch": SINE_CONTROL_AMPLITUDE   * 2.0,           # Right shoulder pitch amplitude
                "right_shoulder_roll": SINE_CONTROL_AMPLITUDE * 0.3,           # Right shoulder roll amplitude
                "right_shoulder_yaw": SINE_CONTROL_AMPLITUDE * 0.8,           # Right shoulder yaw amplitude
                "right_elbow": SINE_CONTROL_AMPLITUDE * 0.7,           # Right elbow amplitude
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
            },
            "phase_shifts": {
                "head_yaw": 0.0,       # Head yaw phase shift
                "head_pitch": 0.5,     # Head pitch phase shift
                "left_arm": 0.0,       # Left arm phase shift
                "right_arm": np.pi,    # Right arm phase shift (opposite to left)
            },
            "offsets": {
                "head_yaw": 0.0,                # Head yaw offset
                "head_pitch": 0.0,              # Head pitch offset
                "left_shoulder_pitch": 0.0,      # Left shoulder pitch offset
                "left_shoulder_roll": 0.0,       # Left shoulder roll offset
                "left_shoulder_yaw": 0.0,        # Left shoulder yaw offset
                "left_elbow": 0.0,               # Left elbow offset
                "right_shoulder_pitch": 0.0,      # Right shoulder pitch offset
                "right_shoulder_roll": 0.0,       # Right shoulder roll offset
                "right_shoulder_yaw": 0.0,        # Right shoulder yaw offset
                "right_elbow": 0.0,               # Right elbow offset
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
            offset = params["offsets"][param_name]
            return self.default_positions[joint_idx] + offset + amp * np.sin(2 * np.pi * freq * t + phase)
        
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
    
    def get_joint_positions(self):
        """Get the current joint positions.
        
        Returns:
            Array of joint positions for upper body joints (head, arms, waist)
        """
        return np.copy(self.joint_positions)


def signal_handler(sig, frame):
    logger.info("\nShutting down...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    args = parser.parse_args()
    
    # Configure logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    # No need to check if numeric_level is int, argparse choices handle validation
    # if not isinstance(numeric_level, int):
    #     raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    # Make logger global for use in classes - declared in the previous edit, ensure it's accessible
    # global logger # This was added in the previous edit if it's truly needed globally.
    # For class methods, it's better to pass the logger or use logging.getLogger(__name__) within the class if not passed.
    # However, to keep changes minimal and assuming the previous `global logger` in main was intentional for simplicity:
    global logger 
    logger = logging.getLogger(__name__) # This will get the logger configured by basicConfig

    cfg_file = os.path.join("configs", args.config)
    control_mode = DEFAULT_CONTROL_MODE

    logger.info("="*50)
    logger.info(f"CONTROL MODE: {control_mode.upper()}")
    logger.info("="*50)
    logger.info(f"Starting upper body controller in {control_mode} mode, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)
    
    # Initialize the controller and get its config
    controller = Controller(cfg_file)
    # Get access to the already loaded config from the controller
    config = controller.cfg
    
    # Initialize tracking system based on mode and arguments
    tracking_system = None
    if control_mode == "teleop":
        if USE_MOCK_TRACKING:
            logger.info("Using mock arm tracking system")
            tracking_system = MockArmTrackingSystem(config=config)
        else:
            logger.info(f"Connecting to WebSocket server at {DEFAULT_WEBSOCKET_URL}")
            # Pass the main controller instance to WebSocketArmTrackingClient
            tracking_system = WebSocketArmTrackingClient(DEFAULT_WEBSOCKET_URL, main_controller_ref=controller, config=config)
            tracking_system.connect()
    
    # With the already initialized controller
    with controller:
        time.sleep(2)  # Wait for channels to initialize
        logger.info("Initialization complete.")
        
        # Start in custom mode
        controller.start_custom_mode_conditionally()
        
        # Start the RL gait for lower body
        controller.start_rl_gait_conditionally()
        
        # Set upper body control mode
        controller.set_body_part_control_mode(BodyPart.UPPER_BODY, control_mode)
        
        if control_mode == "policy":
            logger.info("Robot running with policy control for both upper and lower body")
        elif control_mode == "sine":
            logger.info("Robot running with sine wave control for upper body")
            # Create a sine wave controller with the pre-loaded config
            sine_controller = MockArmTrackingSystem(config=config)
            # Set the sine controller in the main controller
            controller.set_sine_controller(sine_controller)
        elif control_mode == "teleop":
            logger.info("Robot running with teleop control for upper body")
            if USE_MOCK_TRACKING:
                logger.info("Using simulated arm movements")
            else:
                logger.info(f"Receiving arm tracking data from WebSocket at {DEFAULT_WEBSOCKET_URL}")
                logger.info("Ensure your tracking system is sending joint positions in the correct format")
        
        logger.info("Press Ctrl+C to exit")
        
        try:
            while True:
                # Handle different control modes
                if control_mode == "teleop" and tracking_system is not None:
                    # Update tracking data if using mock system
                    if USE_MOCK_TRACKING:
                        tracking_system.update()
                    else:
                        # If not mock tracking, update the observed positions from actual hardware
                        # This ensures self.joint_positions in WebSocketArmTrackingClient is fresh
                        # before the _periodic_sender sends it out.
                        tracking_system.update_observed_joint_positions_from_hardware()
                    
                    # Get COMMANDED joint positions from tracking system (received from HardwareInterface)
                    upper_body_positions_commanded = tracking_system.get_joint_positions()
                    
                    # Update controller with new COMMANDED upper body positions
                    # controller.set_upper_body_positions expects 10 joints (head + 8 arms)
                    # commanded_joint_positions has 11, so we might need to slice it if set_upper_body_positions takes 10
                    if upper_body_positions_commanded is not None and len(upper_body_positions_commanded) >= 10 :
                        controller.set_upper_body_positions(upper_body_positions_commanded[:10]) 
                    else:
                        logger.warning("Not enough commanded positions received or tracking_system not ready for set_upper_body_positions")
                        
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
            logger.info("\nShutting down...")
            
        finally:
            # Clean up resources
            if control_mode == "teleop" and tracking_system is not None and not USE_MOCK_TRACKING:
                tracking_system.disconnect()
                logger.info("WebSocket connection closed")


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

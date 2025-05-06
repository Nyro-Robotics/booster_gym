#!/usr/bin/env python3
"""
WebSocket Server Spoofer for Booster Robot Upper Body Joints

This script creates a WebSocket server that sends mock joint position data
for the 10 upper body joints (2 head + 8 arm joints) of the Booster robot.
It allows testing the reception side of the upper body controller without
needing actual VR/teleop input.

The server sends JSON messages in the format:
{"joint_positions": [j0, j1, j2, ..., j9]}

Joint mapping:
0: Head Yaw Joint
1: Head Pitch Joint
2: Left Shoulder Pitch Joint
3: Left Shoulder Roll Joint
4: Left Shoulder Yaw Joint
5: Left Elbow Joint
6: Right Shoulder Pitch Joint
7: Right Shoulder Roll Joint
8: Right Shoulder Yaw Joint
9: Right Elbow Joint
"""

import asyncio
import websockets
import json
import numpy as np
import time
import threading
import argparse
import signal
import sys

# Default values
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_UPDATE_RATE = 100  # Hz
DEFAULT_MOVEMENT_TYPE = "sine"  # Options: sine, random, fixed

# Default joint position values (neutral position)
DEFAULT_JOINT_POSITIONS = [
    0.0,    # Head Yaw
    0.0,    # Head Pitch
    0.2,    # Left Shoulder Pitch
    -1.35,  # Left Shoulder Roll
    0.0,    # Left Shoulder Yaw
    -0.5,   # Left Elbow
    0.2,    # Right Shoulder Pitch
    1.35,   # Right Shoulder Roll
    0.0,    # Right Shoulder Yaw
    0.5     # Right Elbow
]

# Sine wave parameters - match the ones from upper_body_controller.py
SINE_CONTROL_AMPLITUDE = 0.15      # Base amplitude for sine wave movements
SINE_CONTROL_FREQUENCY = 1.5      # Base frequency for sine wave movements (Hz)

# Random movement parameters
RANDOM_MOVEMENT_RANGE = 0.2
RANDOM_MOVEMENT_SPEED = 0.1

class JointPositionGenerator:
    """Generates joint positions based on selected movement type."""
    
    def __init__(self, movement_type="sine", update_rate=30):
        self.movement_type = movement_type
        self.update_rate = update_rate
        self.start_time = time.time()
        self.joint_positions = np.array(DEFAULT_JOINT_POSITIONS, dtype=np.float32)
        self.target_positions = np.copy(self.joint_positions)
        self.last_random_update = time.time()
        
        # Define movement ranges for each joint (converted from degrees to radians)
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
            9: [-2 * DEG_TO_RAD, 120 * DEG_TO_RAD]     # Right Elbow Joint
        }
        
        # Sine wave parameters - exactly matching those from MockArmTrackingSystem
        self.amplitudes = {
            0: SINE_CONTROL_AMPLITUDE * 1.7,             # Head yaw amplitude
            1: SINE_CONTROL_AMPLITUDE * 0.5,             # Head pitch amplitude
            2: SINE_CONTROL_AMPLITUDE * 2.0,             # Left shoulder pitch amplitude
            3: SINE_CONTROL_AMPLITUDE * 0.3,             # Left shoulder roll amplitude
            4: SINE_CONTROL_AMPLITUDE * 0.8,             # Left shoulder yaw amplitude
            5: SINE_CONTROL_AMPLITUDE * 0.7,             # Left elbow amplitude
            6: SINE_CONTROL_AMPLITUDE * 2.0,             # Right shoulder pitch amplitude
            7: SINE_CONTROL_AMPLITUDE * 0.3,             # Right shoulder roll amplitude
            8: SINE_CONTROL_AMPLITUDE * 0.8,             # Right shoulder yaw amplitude
            9: SINE_CONTROL_AMPLITUDE * 0.7              # Right elbow amplitude
        }
        
        self.frequencies = {
            0: SINE_CONTROL_FREQUENCY * 0.7,             # Head yaw frequency (Hz)
            1: SINE_CONTROL_FREQUENCY * 0.5,             # Head pitch frequency (Hz)
            2: SINE_CONTROL_FREQUENCY * 1.0,             # Left shoulder pitch frequency (Hz)
            3: SINE_CONTROL_FREQUENCY * 0.8,             # Left shoulder roll frequency (Hz)
            4: SINE_CONTROL_FREQUENCY * 0.6,             # Left shoulder yaw frequency (Hz)
            5: SINE_CONTROL_FREQUENCY * 0.7,             # Left elbow frequency (Hz)
            6: SINE_CONTROL_FREQUENCY * 1.0,             # Right shoulder pitch frequency (Hz)
            7: SINE_CONTROL_FREQUENCY * 0.8,             # Right shoulder roll frequency (Hz)
            8: SINE_CONTROL_FREQUENCY * 0.6,             # Right shoulder yaw frequency (Hz)
            9: SINE_CONTROL_FREQUENCY * 0.7              # Right elbow frequency (Hz)
        }
        
        self.phase_shifts = {
            0: 0.0,                                       # Head yaw phase shift
            1: 0.5,                                       # Head pitch phase shift
            2: 0.0,                                       # Left shoulder pitch phase shift (left arm)
            3: 0.0,                                       # Left shoulder roll phase shift (left arm)
            4: 0.0,                                       # Left shoulder yaw phase shift (left arm)
            5: 0.0,                                       # Left elbow phase shift (left arm)
            6: np.pi,                                     # Right shoulder pitch phase shift (right arm, opposite to left)
            7: np.pi,                                     # Right shoulder roll phase shift (right arm, opposite to left) 
            8: np.pi,                                     # Right shoulder yaw phase shift (right arm, opposite to left)
            9: np.pi                                      # Right elbow phase shift (right arm, opposite to left)
        }
        
        self.offsets = {
            0: 0.0,                                       # Head yaw offset
            1: 0.0,                                       # Head pitch offset
            2: 0.0,                                       # Left shoulder pitch offset
            3: 1.0,                                       # Left shoulder roll offset
            4: 0.0,                                       # Left shoulder yaw offset
            5: 0.0,                                       # Left elbow offset
            6: 0.0,                                       # Right shoulder pitch offset
            7: -1.0,                                      # Right shoulder roll offset
            8: 0.0,                                       # Right shoulder yaw offset
            9: 0.0                                        # Right elbow offset
        }
    
    def set_movement_type(self, movement_type):
        """Set the movement type."""
        self.movement_type = movement_type
        print(f"Movement type set to: {movement_type}")
    
    def get_joint_positions(self):
        """Get joint positions based on current movement type."""
        if self.movement_type == "sine":
            return self._get_sine_positions()
        elif self.movement_type == "random":
            return self._get_random_positions()
        else:  # "fixed"
            return DEFAULT_JOINT_POSITIONS
    
    def _get_sine_positions(self):
        """Generate sinusoidal joint positions using parameters exactly matching MockArmTrackingSystem."""
        current_time = time.time() - self.start_time
        positions = np.copy(self.joint_positions)
        
        for i in range(len(positions)):
            # Get parameters for each joint
            amplitude = self.amplitudes[i]
            frequency = self.frequencies[i]
            phase = self.phase_shifts[i]
            offset = self.offsets[i]
            
            # Calculate sine wave position with offset
            # Note: This exactly matches the formula from MockArmTrackingSystem
            positions[i] = offset + amplitude * np.sin(2 * np.pi * frequency * current_time + phase)
            
            # Ensure positions stay within joint limits
            min_val, max_val = self.joint_limits[i]
            positions[i] = np.clip(positions[i], min_val, max_val)
        
        return positions.tolist()
    
    def _get_random_positions(self):
        """Generate smoothly changing random joint positions."""
        current_time = time.time()
        
        # Update target positions occasionally
        if current_time - self.last_random_update > 3.0:  # Every 3 seconds
            self.last_random_update = current_time
            
            for i in range(len(self.joint_positions)):
                min_val, max_val = self.joint_limits[i]
                # Generate a new random target within limits
                self.target_positions[i] = np.random.uniform(min_val, max_val)
        
        # Smooth interpolation towards target
        delta = self.target_positions - self.joint_positions
        self.joint_positions += delta * min(1.0, RANDOM_MOVEMENT_SPEED * (1.0 / self.update_rate))
        
        return self.joint_positions.tolist()

class WebSocketSpoofer:
    """WebSocket server that sends mock joint position data."""
    
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, 
                 update_rate=DEFAULT_UPDATE_RATE, movement_type=DEFAULT_MOVEMENT_TYPE):
        self.host = host
        self.port = port
        self.update_rate = update_rate
        self.connected_clients = set()
        self.running = True
        self.position_generator = JointPositionGenerator(movement_type, update_rate)
        
    async def send_joint_positions(self):
        """Send joint positions to all connected clients periodically."""
        while self.running:
            if self.connected_clients:
                positions = self.position_generator.get_joint_positions()
                message = json.dumps({"joint_positions": positions})
                
                # Send to all connected clients
                await asyncio.gather(
                    *[client.send(message) for client in self.connected_clients],
                    return_exceptions=True
                )
                
            # Sleep based on update rate
            await asyncio.sleep(1.0 / self.update_rate)
    
    async def handler(self, websocket):
        """Handle new WebSocket connections."""
        try:
            print(f"Client connected from {websocket.remote_address}")
            self.connected_clients.add(websocket)
            
            # Keep connection open and handle any incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle any incoming commands
                    if "command" in data:
                        if data["command"] == "set_movement_type" and "value" in data:
                            self.position_generator.set_movement_type(data["value"])
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed from {websocket.remote_address}")
        finally:
            self.connected_clients.remove(websocket)
    
    async def start_server(self):
        """Start the WebSocket server."""
        server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket server started at ws://{self.host}:{self.port}")
        
        # Start the position update task
        update_task = asyncio.create_task(self.send_joint_positions())
        
        # Keep server running until stopped
        try:
            await server.wait_closed()
        finally:
            self.running = False
            await update_task

def signal_handler(sig, frame):
    """Handle interrupt signal."""
    print("\nShutting down WebSocket server...")
    sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket server for spoofing Booster robot joint positions")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind server (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind server (default: {DEFAULT_PORT})")
    parser.add_argument("--rate", type=int, default=DEFAULT_UPDATE_RATE, 
                        help=f"Update rate in Hz (default: {DEFAULT_UPDATE_RATE})")
    parser.add_argument("--movement", choices=["sine", "random", "fixed"], default=DEFAULT_MOVEMENT_TYPE,
                        help=f"Movement type (default: {DEFAULT_MOVEMENT_TYPE})")
    
    args = parser.parse_args()
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start server
    spoofer = WebSocketSpoofer(args.host, args.port, args.rate, args.movement)
    
    print("\n========== Booster Robot Joint Position Spoofer ==========")
    print(f"Server: ws://{args.host}:{args.port}")
    print(f"Movement Type: {args.movement}")
    print(f"Update Rate: {args.rate} Hz")
    print("Joint Mapping:")
    print("  0: Head Yaw")
    print("  1: Head Pitch")
    print("  2-5: Left Arm (Shoulder Pitch, Roll, Yaw, Elbow)")
    print("  6-9: Right Arm (Shoulder Pitch, Roll, Yaw, Elbow)")
    print("==========================================================")
    print("Press Ctrl+C to stop the server")
    
    # Run the async event loop
    asyncio.run(spoofer.start_server())

if __name__ == "__main__":
    main()

from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, GripperControlMode, Position, Orientation, Posture, GripperMotionParameter, Quaternion, Frame, Transform, DexterousFingerParameter, LowCmd, LowCmdType, MotorCmd, B1LowCmdPublisher, B1LowStateSubscriber, B1LowHandDataScriber
import sys, time, random, math
import asyncio
import websockets
import json
import threading
import argparse
from typing import Dict, Any, Optional
from collections import deque
import numpy as np

# Global variables for teleoperation
teleop_active = False
teleop_thread = None
latest_joint_data = {}
latest_hand_data = {}
robot_client = None
current_positions = [0.0] * 29
lower_body_positions = [0.0] * 12  # Joints 17-28
arm_stiffness_factor = 1.5  # Downscale arm stiffness by 0.8

# Joint position smoothing parameters
joint_smoothing_factor = 0.8  # How much to keep of previous value (0.0 = no smoothing, 0.9 = heavy smoothing)
filtered_joint_positions = {}  # Store smoothed joint positions

# Latency tracking variables
latency_window_size = 100  # Track last 100 messages for latency stats
message_latencies = deque(maxlen=latency_window_size)
joint_position_latencies = deque(maxlen=latency_window_size)
total_latency_sum = 0.0
total_latency_count = 0
min_latency = float('inf')
max_latency = 0.0
last_latency_report_time = 0.0

def prepare_robot(client: B1LocoClient):
    """Send robot to a stable prepare pose before switching to custom mode"""
    global current_positions
    
    # First, read current robot state
    print("Reading current robot position...")
    current_positions = [0.0] * 29
    positions_received = False
    
    def state_handler(low_state_msg):
        global current_positions
        nonlocal positions_received
        if len(low_state_msg.motor_state_serial) >= 29:
            for i in range(29):
                current_positions[i] = low_state_msg.motor_state_serial[i].q
            positions_received = True
    
    # Subscribe to get current state
    state_subscriber = B1LowStateSubscriber(state_handler)
    state_subscriber.InitChannel()
    
    # Wait for state data
    timeout = 5.0  # 5 second timeout
    start_time = time.time()
    while not positions_received and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    state_subscriber.CloseChannel()
    
    if not positions_received:
        print("Warning: Could not read current positions, using zeros as starting point")
        current_positions = [0.0] * 29
    else:
        print("Successfully read current robot positions!")
    
    # Store lower body positions to maintain them during teleoperation
    global lower_body_positions
    lower_body_positions = current_positions[17:29]  # Joints 17-28
    
    # Create low-level command publisher
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    
    # Create and initialize the command
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    
    # Initialize motor commands - total of 29 joints
    motor_cmds = [MotorCmd() for _ in range(29)]
    low_cmd.motor_cmd = motor_cmds
    
    # Initialize all motor commands to zero
    for i in range(29):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].weight = 0.0
    
    # Basic gains for holding position
    kps = [
        2.0, 2.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100., 
        350., 350., 180., 350., 450., 450.,
        350., 350., 180., 350., 450., 450.,
    ]
    
    kds = [
        0.3, 0.3,                                   # Head
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Left arm
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Right arm
        5.0,
        7.5, 7.5, 3., 5.5, 0.5, 0.5,
        7.5, 7.5, 3., 5.5, 0.5, 0.5,
    ]
    
    # Joint names for debugging
    joint_names = [
        "Head_Yaw", "Head_Pitch",
        "L_Shoulder_Pitch", "L_Shoulder_Roll", "L_Elbow_Pitch", "L_Elbow_Yaw", "L_Wrist_Pitch", "L_Wrist_Yaw", "L_Hand_Roll",
        "R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Elbow_Pitch", "R_Elbow_Yaw", "R_Wrist_Pitch", "R_Wrist_Yaw", "R_Hand_Roll",
        "Waist",
        "L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee_Pitch", "L_Crank_Up", "L_Crank_Down",
        "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw", "R_Knee_Pitch", "R_Crank_Up", "R_Crank_Down"
    ]
    
    print("BASIC TEST: Commanding robot to hold current positions...")
    print("Current positions:")
    for i, (name, pos) in enumerate(zip(joint_names, current_positions)):
        print(f"  {name}: {pos:.3f} rad ({pos * 180 / 3.14159:.1f}°)")
    
    # BASIC TEST: Just command the robot to hold its current positions
    for i in range(29):
        low_cmd.motor_cmd[i].q = current_positions[i]  # Hold current position
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = kps[i]
        low_cmd.motor_cmd[i].kd = kds[i]
        low_cmd.motor_cmd[i].tau = 0.0
    
    print("Sending current position commands...")
    low_cmd_publisher.Write(low_cmd)
    
    print("Switching to custom mode...")
    res = client.ChangeMode(RobotMode.kCustom)
    if res != 0:
        print(f"Failed to switch to custom mode: error = {res}")
        low_cmd_publisher.CloseChannel()
        return False
    
    print("Successfully switched to custom mode!")
    
    # Send the hold command once more after mode switch (like deploy.py)
    print("Sending final hold-position command...")
    low_cmd_publisher.Write(low_cmd)
    
    print("BASIC TEST COMPLETE!")
    print("Robot should now be holding its current positions in custom mode.")
    print("If this works, we can add movement logic later.")
    
    # Clean up
    low_cmd_publisher.CloseChannel()
    
    return True

def read_hand_data(hand_index=B1HandIndex.kRightHand, duration=2.0):
    """Read hand data for the specified hand for a given duration"""
    print(f"Reading hand data for {'right' if hand_index == B1HandIndex.kRightHand else 'left'} hand...")
    
    def handler(hand_data_msg):
        print("Received hand message:")
        for i, data in enumerate(hand_data_msg.hand_data):
            print(f" seq:{data.seq} angle:{data.angle}, force:{data.force}, current:{data.current}, status:{data.status}, temp:{data.temp}, error:{data.error}")
        print(f" hand index:{hand_data_msg.hand_index} hand type:{hand_data_msg.hand_type}")
        print("done")
    
    # Create subscriber
    channel_subscriber = B1LowHandDataScriber(handler)
    channel_subscriber.InitChannel()
    print("Hand data subscriber initialized")
    
    # Read data for specified duration
    start_time = time.time()
    while (time.time() - start_time) < duration:
        time.sleep(0.1)
    
    # Clean up
    channel_subscriber.CloseChannel()
    print("Hand data reading completed")

def hand_oscillation(client: B1LocoClient):
    # 定义一个 名为 finger_params 的数组，用于存储每个手指的参数
    print("Reading hand data for both hands...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)

    # Base positions for each finger (from original scissor position)
    base_positions = [200, 400, 600, 800, 1000, 100]
    
    # Sinusoidal parameters
    amplitude = 300  # How much the fingers oscillate
    frequency = 1.0  # Hz - oscillations per second
    duration = 10.0  # Total duration in seconds
    
    print(f"Starting sinusoidal hand movement on BOTH hands for {duration} seconds...")
    print("Using maximum speed, minimum force, with increased delay")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < duration:
            current_time = time.time() - start_time
            
            finger_params = []
            
            # Calculate sinusoidal angles for each finger
            for i in range(6):
                # Add phase offset for each finger to create wave-like motion
                phase_offset = i * 0.5  # Different phase for each finger
                
                # Calculate sinusoidal angle
                sin_value = amplitude * math.sin(2 * math.pi * frequency * current_time + phase_offset)
                target_angle = base_positions[i] + sin_value
                
                # Clamp angle to reasonable bounds (0-1200)
                target_angle = max(0, min(1200, target_angle))
                
                finger_param = DexterousFingerParameter()
                finger_param.seq = i
                finger_param.angle = int(target_angle)
                finger_param.force = 1000  # Minimum force
                finger_param.speed = 1000  # Maximum speed
                finger_params.append(finger_param)
            
            # Send command to RIGHT hand
            res_right = client.ControlDexterousHand(finger_params, B1HandIndex.kRightHand)
            if res_right != 0:
                print(f"Right hand sinusoidal command failed: error = {res_right}")
            
            # Send command to LEFT hand
            res_left = client.ControlDexterousHand(finger_params, B1HandIndex.kLeftHand)
            if res_left != 0:
                print(f"Left hand sinusoidal command failed: error = {res_left}")
            
            # Break if both hands failed
            if res_right != 0 and res_left != 0:
                break
            
            # Print current angles every 0.5 seconds
            if int(current_time * 2) != int((current_time - 0.2) * 2):
                angles_str = ", ".join([f"{p.angle}" for p in finger_params])
                print(f"t={current_time:.1f}s: BOTH HANDS [{angles_str}]")
            
            # Delay between commands (100Hz)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nSinusoidal motion interrupted by user")
    
    print("Sinusoidal hand movement completed on both hands")
    print("Reading final hand data...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)

def map_joint_name_to_index(joint_name: str) -> Optional[int]:
    """Map joint name to robot joint index"""
    joint_mapping = {
        # Head joints
        "AAHead_yaw": 0, "Head_pitch": 1,
        # Left arm joints  
        "Left_Shoulder_Pitch": 2, "Left_Shoulder_Roll": 3, "Left_Elbow_Pitch": 4, 
        "Left_Elbow_Yaw": 5, "Left_Wrist_Pitch": 6, "Left_Wrist_Yaw": 7, "Left_Hand_Roll": 8,
        # Right arm joints
        "Right_Shoulder_Pitch": 9, "Right_Shoulder_Roll": 10, "Right_Elbow_Pitch": 11, 
        "Right_Elbow_Yaw": 12, "Right_Wrist_Pitch": 13, "Right_Wrist_Yaw": 14, "Right_Hand_Roll": 15,
        # Waist joint
        "Trunk_Waist": 16
    }
    return joint_mapping.get(joint_name)

def map_hand_data_to_finger_params(hand_data: Dict[str, float]) -> list:
    """Map hand joint data to finger parameters"""
    finger_params = []
    
    # Map hand joint names to finger sequences
    # According to hands_doc.md: 0=little, 1=ring, 2=middle, 3=index, 4=thumb_bend, 5=thumb_rot
    # From the example data, we have specific MCP joint names
    finger_mapping = {
        # Left hand
        "L_pinky_MCP_joint": 0,      # Little finger
        "L_ring_MCP_joint": 1,       # Ring finger  
        "L_middle_MCP_joint": 2,     # Middle finger
        "L_index_MCP_joint": 3,      # Index finger
        "L_thumb_MCP_joint2": 4,     # Thumb bending
        "L_thumb_MCP_joint1": 5,     # Thumb rotation
        # Right hand
        "R_pinky_MCP_joint": 0,      # Little finger
        "R_ring_MCP_joint": 1,       # Ring finger
        "R_middle_MCP_joint": 2,     # Middle finger
        "R_index_MCP_joint": 3,      # Index finger
        "R_thumb_MCP_joint2": 4,     # Thumb bending
        "R_thumb_MCP_joint1": 5,     # Thumb rotation
    }
    
    for joint_name, joint_value in hand_data.items():
        finger_seq = finger_mapping.get(joint_name)
        
        if finger_seq is not None:
            # Convert joint value to angle (0-1000 range)
            # The incoming data appears to be normalized (0.0-1.0), so scale to 0-1000
            angle = max(0, min(1000, int(joint_value * 1000)))
            
            finger_param = DexterousFingerParameter()
            finger_param.seq = finger_seq
            finger_param.angle = angle
            finger_param.force = 200  # Low force as requested
            finger_param.speed = 1000  # Maximum speed as requested
            finger_params.append(finger_param)
    
    # If no specific mapping found, create default finger params
    if not finger_params:
        for i in range(6):
            finger_param = DexterousFingerParameter()
            finger_param.seq = i
            finger_param.angle = 500  # Middle position
            finger_param.force = 200
            finger_param.speed = 1000
            finger_params.append(finger_param)
    
    return finger_params

async def websocket_client_handler(host: str = "localhost", port: int = 8765):
    """Handle WebSocket connection and message processing"""
    global latest_joint_data, latest_hand_data, teleop_active
    global message_latencies, joint_position_latencies, total_latency_sum, total_latency_count, min_latency, max_latency
    
    uri = f"ws://{host}:{port}"
    
    try:
        print(f"🔌 Connecting to WebSocket server at {uri}")
        
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to WebSocket server successfully!")
            print(f"🎯 Listening for teleoperation commands...")
            
            async for message in websocket:
                if not teleop_active:
                    break
                    
                try:
                    # Capture receive time immediately for latency calculation
                    receive_time = time.time()
                    
                    # Parse JSON message
                    data = json.loads(message)
                    message_type = data.get('type', 'unknown')
                    
                    # Calculate latency if message has timestamp
                    latency_ms = None
                    message_timestamp = data.get('timestamp')
                    if message_timestamp is not None:
                        latency_s = receive_time - message_timestamp
                        latency_ms = latency_s * 1000  # Convert to milliseconds
                        
                        # Update global latency statistics
                        message_latencies.append(latency_ms)
                        total_latency_sum += latency_ms
                        total_latency_count += 1
                        min_latency = min(min_latency, latency_ms)
                        max_latency = max(max_latency, latency_ms)
                        
                        # Track joint position specific latency
                        if message_type == 'joint_positions':
                            joint_position_latencies.append(latency_ms)
                    
                    if message_type == 'joint_positions':
                        # Extract joint data
                        message_data = data.get('data', {})
                        all_joints = message_data.get('all_joints', {})
                        organized = message_data.get('organized', {})
                        
                        # Update latest joint data with organized structure
                        latest_joint_data = organized.copy() if organized else all_joints.copy()
                        
                        # Extract hand data from organized data
                        if organized:
                            latest_hand_data = {
                                'left_hand': organized.get('left_hand', {}),
                                'right_hand': organized.get('right_hand', {})
                            }
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
    except ConnectionRefusedError:
        print(f"❌ Connection refused to {uri}")
        print(f"   Make sure the WebSocket relay server is running")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print(f"🔌 Disconnected from WebSocket server")

def teleoperation_control_loop(client: B1LocoClient):
    """Main teleoperation control loop running at 100Hz"""
    global teleop_active, latest_joint_data, latest_hand_data, current_positions, lower_body_positions, arm_stiffness_factor
    global joint_smoothing_factor, filtered_joint_positions
    
    # Create low-level command publisher
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    
    # Create and initialize the command
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    
    # Initialize motor commands
    motor_cmds = [MotorCmd() for _ in range(29)]
    low_cmd.motor_cmd = motor_cmds
    
    # Base gains for teleoperation
    base_kps = [
        2.0, 2.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100.,                                        # Waist
        350., 350., 180., 350., 450., 450.,          # Left leg
        350., 350., 180., 350., 450., 450.,          # Right leg
    ]
    
    base_kds = [
        0.3, 0.3,                                   # Head
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Left arm
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Right arm
        5.0,                                        # Waist
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Left leg
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Right leg
    ]
    
    # Apply arm stiffness factor to arm joints only (joints 2-8 and 9-15)
    kps = base_kps.copy()
    kds = base_kds.copy()
    
    # Apply arm stiffness factor to left arm (joints 2-8)
    for i in range(2, 9):
        kps[i] *= arm_stiffness_factor
        kds[i] *= arm_stiffness_factor
    
    # Apply arm stiffness factor to right arm (joints 9-15)  
    for i in range(9, 16):
        kps[i] *= arm_stiffness_factor
        kds[i] *= arm_stiffness_factor
    
    print("🚀 Starting teleoperation control loop at 100Hz")
    print("   Upper body joints will follow WebSocket commands")
    print("   Lower body joints will maintain their current positions")
    print(f"   Arm stiffness factor: {arm_stiffness_factor} (arms only)")
    print(f"   Joint smoothing factor: {joint_smoothing_factor} (0.0=no smoothing, 0.9=heavy smoothing)")
    print("   Press Ctrl+C to stop teleoperation")
    
    start_time = time.time()
    command_count = 0
    
    try:
        while teleop_active:
            loop_start_time = time.time()
            
            # Initialize all motor commands
            for i in range(29):
                low_cmd.motor_cmd[i].q = current_positions[i]
                low_cmd.motor_cmd[i].dq = 0.0
                low_cmd.motor_cmd[i].kp = kps[i]
                low_cmd.motor_cmd[i].kd = kds[i]
                low_cmd.motor_cmd[i].tau = 0.0
            
            # Apply joint commands from WebSocket using organized data with smoothing
            if latest_joint_data:
                # Use organized upper_body data if available
                if 'upper_body' in latest_joint_data:
                    upper_body_joints = latest_joint_data['upper_body']
                    for joint_name, joint_value in upper_body_joints.items():
                        joint_index = map_joint_name_to_index(joint_name)
                        if joint_index is not None and joint_index <= 16:  # Upper body only
                            # Apply smoothing filter
                            if joint_name in filtered_joint_positions:
                                # filteredjoint_value[i] = self.filteredjoint_value[i] * 0.8 + self.joint_value[i] * 0.2
                                filtered_joint_positions[joint_name] = (
                                    filtered_joint_positions[joint_name] * joint_smoothing_factor + 
                                    joint_value * (1.0 - joint_smoothing_factor)
                                )
                            else:
                                # Initialize with current value
                                filtered_joint_positions[joint_name] = joint_value
                            
                            # Use filtered value for robot command
                            low_cmd.motor_cmd[joint_index].q = filtered_joint_positions[joint_name]
                else:
                    # Fallback to all_joints if organized data not available
                    for joint_name, joint_value in latest_joint_data.items():
                        joint_index = map_joint_name_to_index(joint_name)
                        if joint_index is not None and joint_index <= 16:  # Upper body only
                            # Apply smoothing filter
                            if joint_name in filtered_joint_positions:
                                filtered_joint_positions[joint_name] = (
                                    filtered_joint_positions[joint_name] * joint_smoothing_factor + 
                                    joint_value * (1.0 - joint_smoothing_factor)
                                )
                            else:
                                # Initialize with current value
                                filtered_joint_positions[joint_name] = joint_value
                            
                            # Use filtered value for robot command
                            low_cmd.motor_cmd[joint_index].q = filtered_joint_positions[joint_name]
            
            # Keep lower body at original positions (joints 17-28)
            for i in range(17, 29):
                if i - 17 < len(lower_body_positions):
                    low_cmd.motor_cmd[i].q = lower_body_positions[i - 17]
            
            # Send joint commands
            low_cmd_publisher.Write(low_cmd)
            
            # Send hand commands
            if latest_hand_data:
                # Control left hand
                if 'left_hand' in latest_hand_data and latest_hand_data['left_hand']:
                    left_finger_params = map_hand_data_to_finger_params(latest_hand_data['left_hand'])
                    if left_finger_params:
                        client.ControlDexterousHand(left_finger_params, B1HandIndex.kLeftHand)
                
                # Control right hand
                if 'right_hand' in latest_hand_data and latest_hand_data['right_hand']:
                    right_finger_params = map_hand_data_to_finger_params(latest_hand_data['right_hand'])
                    if right_finger_params:
                        client.ControlDexterousHand(right_finger_params, B1HandIndex.kRightHand)
            
            command_count += 1
            
            # Print statistics every 10 seconds
            if command_count % 1000 == 0:  # Every 10 seconds at 100Hz
                elapsed_time = time.time() - start_time
                actual_hz = command_count / elapsed_time
                
                # Get current latency statistics
                latency_stats = get_latency_statistics()
                latency_report = format_latency_report(latency_stats, detailed=False)
                
                print(f"📊 Teleoperation stats: {actual_hz:.1f}Hz, {command_count} commands sent")
                print(f"   Latest joint data keys: {list(latest_joint_data.keys()) if latest_joint_data else 'None'}")
                print(f"   Latest hand data: L={len(latest_hand_data.get('left_hand', {}))}, R={len(latest_hand_data.get('right_hand', {}))}")
                print(f"   Arm stiffness factor: {arm_stiffness_factor}")
                print(f"   Joint smoothing factor: {joint_smoothing_factor}")
                print(f"   Filtered joints: {len(filtered_joint_positions)}")
                print(f"   {latency_report}")
                
                # Network quality assessment for teleoperation
                if 'current_avg_ms' in latency_stats:
                    avg_latency = latency_stats['current_avg_ms']
                    if avg_latency > 50:
                        print(f"   ⚠️ Warning: High latency may affect teleoperation quality")
                    elif avg_latency < 15:
                        print(f"   ✅ Good latency for real-time teleoperation")
            
            # Maintain 100Hz loop rate
            loop_time = time.time() - loop_start_time
            target_loop_time = 0.01  # 100Hz = 10ms
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)
                
    except KeyboardInterrupt:
        print("\n🛑 Teleoperation interrupted by user")
    finally:
        print("🔌 Cleaning up teleoperation resources")
        low_cmd_publisher.CloseChannel()
        
        # Final statistics
        elapsed_time = time.time() - start_time
        actual_hz = command_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get final latency statistics
        final_latency_stats = get_latency_statistics()
        detailed_latency_report = format_latency_report(final_latency_stats, detailed=True)
        
        print(f"📊 Final teleoperation stats:")
        print(f"   Duration: {elapsed_time:.1f}s")
        print(f"   Commands sent: {command_count}")
        print(f"   Average frequency: {actual_hz:.1f}Hz")
        print(f"   Arm stiffness factor used: {arm_stiffness_factor}")
        print(f"   Joint smoothing factor used: {joint_smoothing_factor}")
        print(f"   Total filtered joints: {len(filtered_joint_positions)}")
        
        # Detailed latency analysis
        print(f"\n🕐 FINAL LATENCY ANALYSIS:")
        if final_latency_stats:
            print(f"   {detailed_latency_report}")
            
            # Network quality assessment for teleoperation
            if 'current_avg_ms' in final_latency_stats:
                avg_latency = final_latency_stats['current_avg_ms']
                jitter = final_latency_stats['current_std_ms']
                
                print(f"\n   🌐 NETWORK QUALITY ASSESSMENT:")
                if avg_latency < 10 and jitter < 5:
                    print(f"     ✅ Excellent network quality - Perfect for real-time teleoperation")
                elif avg_latency < 20 and jitter < 10:
                    print(f"     ✅ Good network quality - Well suited for teleoperation")
                elif avg_latency < 50 and jitter < 25:
                    print(f"     ⚠️  Fair network quality - Acceptable for teleoperation")
                else:
                    print(f"     ❌ Poor network quality - May affect teleoperation performance")
                    print(f"       Consider optimizing network connection or reducing data rate")
                    
                # Specific recommendations for teleoperation
                print(f"\n   📋 TELEOPERATION RECOMMENDATIONS:")
                if avg_latency < 15:
                    print(f"     ✅ Latency is excellent for real-time control")
                elif avg_latency < 30:
                    print(f"     ✅ Latency is good for teleoperation tasks")
                elif avg_latency < 50:
                    print(f"     ⚠️  Latency is acceptable but may affect precision tasks")
                else:
                    print(f"     ❌ Latency is too high for optimal teleoperation")
                    print(f"       Consider: faster network, local processing, or reduced update rate")
                    
                if jitter > 20:
                    print(f"     ⚠️  High jitter detected - network stability may be poor")
                    print(f"       Consider: wired connection, QoS settings, or network optimization")
        else:
            print(f"   No latency data available (messages may not include timestamps)")
        
        # Clear filtered positions for next session
        filtered_joint_positions.clear()

def reset_latency_tracking():
    """Reset all latency tracking variables for a new session"""
    global message_latencies, joint_position_latencies, total_latency_sum, total_latency_count, min_latency, max_latency, last_latency_report_time
    
    message_latencies.clear()
    joint_position_latencies.clear()
    total_latency_sum = 0.0
    total_latency_count = 0
    min_latency = float('inf')
    max_latency = 0.0
    last_latency_report_time = time.time()

def start_teleoperation(client: B1LocoClient, host: str = "localhost", port: int = 8765):
    """Start teleoperation mode"""
    global teleop_active, teleop_thread, robot_client
    
    if teleop_active:
        print("❌ Teleoperation is already active")
        return False
    
    robot_client = client
    teleop_active = True
    
    # Reset latency tracking for new session
    reset_latency_tracking()
    
    print("🚀 Starting teleoperation mode...")
    print(f"   WebSocket server: {host}:{port}")
    print("   Control frequency: 100Hz")
    print("   Upper body: Following WebSocket commands")
    print("   Lower body: Maintaining current positions")
    print("   Latency tracking: Enabled (requires timestamps in messages)")
    
    # Start WebSocket client in a separate thread
    websocket_thread = threading.Thread(
        target=lambda: asyncio.run(websocket_client_handler(host, port)),
        daemon=True
    )
    websocket_thread.start()
    
    # Start control loop in main thread
    try:
        teleoperation_control_loop(client)
    except Exception as e:
        print(f"❌ Error in teleoperation: {e}")
    finally:
        teleop_active = False
        print("🔌 Teleoperation stopped")
    
    return True

def stop_teleoperation():
    """Stop teleoperation mode"""
    global teleop_active
    
    if not teleop_active:
        print("❌ Teleoperation is not active")
        return False
    
    print("🛑 Stopping teleoperation mode...")
    teleop_active = False
    
    # Give some time for threads to clean up
    time.sleep(1.0)
    
    print("✅ Teleoperation stopped successfully")
    return True

def get_latency_statistics() -> Dict[str, Any]:
    """Calculate current latency statistics"""
    global message_latencies, joint_position_latencies, total_latency_sum, total_latency_count, min_latency, max_latency
    
    stats = {}
    
    # Overall latency statistics from recent window
    if len(message_latencies) > 5:
        recent_latencies = list(message_latencies)
        stats['current_avg_ms'] = np.mean(recent_latencies)
        stats['current_std_ms'] = np.std(recent_latencies)
        stats['current_min_ms'] = np.min(recent_latencies)
        stats['current_max_ms'] = np.max(recent_latencies)
        stats['current_median_ms'] = np.median(recent_latencies)
        stats['window_size'] = len(recent_latencies)
    
    # Joint position specific latency
    if len(joint_position_latencies) > 5:
        joint_latencies = list(joint_position_latencies)
        stats['joint_avg_ms'] = np.mean(joint_latencies)
        stats['joint_std_ms'] = np.std(joint_latencies)
        stats['joint_min_ms'] = np.min(joint_latencies)
        stats['joint_max_ms'] = np.max(joint_latencies)
        stats['joint_window_size'] = len(joint_latencies)
    
    # Lifetime statistics
    if total_latency_count > 0:
        stats['lifetime_avg_ms'] = total_latency_sum / total_latency_count
        stats['lifetime_min_ms'] = min_latency if min_latency != float('inf') else 0
        stats['lifetime_max_ms'] = max_latency
        stats['lifetime_count'] = total_latency_count
    
    return stats

def format_latency_report(stats: Dict[str, Any], detailed: bool = False) -> str:
    """Format latency statistics for display"""
    if not stats:
        return "🕐 Latency: No timestamp data available"
    
    lines = []
    
    # Current window statistics
    if 'current_avg_ms' in stats:
        avg = stats['current_avg_ms']
        std = stats['current_std_ms']
        
        # Quality assessment
        if avg < 10:
            quality = "✅ Excellent"
        elif avg < 20:
            quality = "✅ Good"
        elif avg < 50:
            quality = "⚠️ Fair"
        else:
            quality = "❌ High"
        
        lines.append(f"🕐 Latency: {avg:.1f}ms avg (±{std:.1f}ms) - {quality}")
        
        if detailed:
            lines.append(f"   Recent window: {stats['current_min_ms']:.1f}-{stats['current_max_ms']:.1f}ms range, {stats['current_median_ms']:.1f}ms median")
            
            # Joint position specific
            if 'joint_avg_ms' in stats:
                joint_avg = stats['joint_avg_ms']
                joint_std = stats['joint_std_ms']
                lines.append(f"   Joint positions: {joint_avg:.1f}ms avg (±{joint_std:.1f}ms)")
            
            # Lifetime statistics
            if 'lifetime_avg_ms' in stats:
                lifetime_avg = stats['lifetime_avg_ms']
                lifetime_min = stats['lifetime_min_ms']
                lifetime_max = stats['lifetime_max_ms']
                lifetime_count = stats['lifetime_count']
                lines.append(f"   Lifetime: {lifetime_avg:.1f}ms avg, {lifetime_min:.1f}-{lifetime_max:.1f}ms range ({lifetime_count} messages)")
    
    return "\n".join(lines) if lines else "🕐 Latency: Calculating..."

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface [--ws-host HOST] [--ws-port PORT]")
        sys.exit(-1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot control with teleoperation')
    parser.add_argument('network_interface', help='Network interface for robot communication')
    parser.add_argument('--ws-host', type=str, default='localhost',
                       help='WebSocket server host for teleoperation (default: localhost)')
    parser.add_argument('--ws-port', type=int, default=8765,
                       help='WebSocket server port for teleoperation (default: 8765)')
    
    args = parser.parse_args()
    
    # Initialize channel with network interface
    ChannelFactory.Instance().Init(0, args.network_interface)

    client = B1LocoClient()
    client.Init()
    x, y, z, yaw, pitch = 0.0, 0.0, 0.0, 0.0, 0.0
    res = 0
    hand_action_count = 0

    print("=" * 60)
    print("ROBOT CONTROL COMMANDS:")
    print("=" * 60)
    print("Mode Commands:")
    print("  mc      - Prepare robot and switch to Custom mode")
    print()
    print("Teleoperation Commands:")
    print(f"  teleop  - Start teleoperation mode (connects to {args.ws_host}:{args.ws_port})")
    print("  stop_teleop - Stop teleoperation mode")
    print()
    print("Movement Commands (in Walking mode):")
    print("  w/a/s/d/q/e - Move robot")
    print("  stop        - Stop movement")
    print()
    print("Head Commands:")
    print("  hd/hu/hr/hl - Move head down/up/right/left")  
    print("  ho          - Center head")
    print()
    print("Hand Commands:")
    print("  hand - Hand oscillation")
    print()
    print("TELEOPERATION WORKFLOW:")
    print("  1. First run 'mc' to prepare robot and switch to custom mode")
    print("  2. Then run 'teleop' to start receiving WebSocket commands")
    print(f"  3. Make sure WebSocket server is running on {args.ws_host}:{args.ws_port}")
    print("  4. Upper body joints will follow WebSocket joint commands")
    print("  5. Hand joints will follow WebSocket hand commands")
    print("  6. Lower body joints will maintain their current positions")
    print("  7. Latency tracking monitors network performance (requires timestamps)")
    print("=" * 60)
    print()
    print("📊 TELEOPERATION FEATURES:")
    print("  • Real-time joint position control at 100Hz")
    print("  • Hand/finger control with low force settings") 
    print("  • Joint position smoothing to reduce jitter")
    print("  • Configurable arm stiffness scaling")
    print("  • Network latency monitoring and quality assessment")
    print("  • Periodic statistics reporting every 10 seconds")
    print("=" * 60)

    try:
        while True:
            need_print = False
            input_cmd = input().strip()
            if input_cmd:
                if input_cmd == 'mc':
                    print("Preparing robot for custom mode...")
                    success = prepare_robot(client)
                    if not success:
                        print("Failed to prepare robot for custom mode")
                elif input_cmd == 'teleop':
                    print(f"Starting teleoperation mode (connecting to {args.ws_host}:{args.ws_port})...")
                    success = start_teleoperation(client, args.ws_host, args.ws_port)
                    if not success:
                        print("Failed to start teleoperation mode")
                elif input_cmd == 'stop_teleop':
                    print("Stopping teleoperation mode...")
                    success = stop_teleoperation()
                    if not success:
                        print("Failed to stop teleoperation mode")
                elif input_cmd == "stop":
                    x, y, z = 0.0, 0.0, 0.0
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "w":
                    x, y, z = 0.8, 0.0, 0.0
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "a":
                    x, y, z = 0.0, 0.2, 0.0
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "s":
                    x, y, z = -0.2, 0.0, 0.0
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "d":
                    x, y, z = 0.0, -0.2, 0.0
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "q":
                    x, y, z = 0.0, 0.0, 0.2
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "e":
                    x, y, z = 0.0, 0.0, -0.2
                    need_print = True
                    res = client.Move(x, y, z)
                elif input_cmd == "hd":
                    yaw, pitch = 0.0, 1.0
                    need_print = True
                    res = client.RotateHead(pitch, yaw)
                elif input_cmd == "hu":
                    yaw, pitch = 0.0, -0.3
                    need_print = True
                    res = client.RotateHead(pitch, yaw)
                elif input_cmd == "hr":
                    yaw, pitch = -0.785, 0.0
                    need_print = True
                    res = client.RotateHead(pitch, yaw)
                elif input_cmd == "hl":
                    yaw, pitch = 0.785, 0.0
                    need_print = True
                    res = client.RotateHead(pitch, yaw)
                elif input_cmd == "ho":
                    yaw, pitch = 0.0, 0.0
                    need_print = True
                    res = client.RotateHead(pitch, yaw)
                elif input_cmd == "mhel":
                    tar_posture = Posture()
                    tar_posture.position = Position(0.35, 0.25, 0.1)
                    tar_posture.orientation = Orientation(-1.57, -1.57, 0.0)
                    res = client.MoveHandEndEffectorV2(tar_posture, 2000, B1HandIndex.kLeftHand)
                elif input_cmd == "gopenl":
                    motion_param = GripperMotionParameter()
                    motion_param.position = 500
                    motion_param.force = 100
                    motion_param.speed = 100
                    res = client.ControlGripper(motion_param, GripperControlMode.kPosition, B1HandIndex.kLeftHand)
                elif input_cmd == "hcm-start":
                    res = client.SwitchHandEndEffectorControlMode(True)
                elif input_cmd == "hcm-stop":
                    res = client.SwitchHandEndEffectorControlMode(False)
                elif input_cmd == "hand":
                    hand_oscillation(client)

                if need_print:
                    print(f"Param: {x} {y} {z}")
                    print(f"Head param: {pitch} {yaw}")

                if res != 0:
                    print(f"Request failed: error = {res}")

    except KeyboardInterrupt:
        print("\nStopping...")
        if teleop_active:
            stop_teleoperation()

if __name__ == "__main__":
    main()
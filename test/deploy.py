from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, GripperControlMode, Position, Orientation, Posture, GripperMotionParameter, Quaternion, Frame, Transform, DexterousFingerParameter, LowCmd, LowCmdType, MotorCmd, B1LowCmdPublisher, B1LowStateSubscriber, B1LowHandDataScriber
import sys, time, random, math
import asyncio
import threading
import argparse
from typing import Dict, Any, Optional
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

# Import utility modules
from utils.websocket_client import WebSocketClient, send_initialization_message_websocket, create_robot_joint_initialization_message
from utils.zeromq_client import ZeroMQClient, send_initialization_message_zeromq
from utils.hand_controls import (
    hand_oscillation, close_hands, open_hands, read_hand_data,
    map_hand_data_to_finger_params, apply_current_reduction_to_finger_command
)

# Global variables for teleoperation
teleop_active = False
teleop_thread = None
latest_joint_data = {}
latest_hand_data = {}
robot_client = None
current_positions = [0.0] * 29
lower_body_positions = [0.0] * 12  # Joints 17-28
arm_stiffness_factor = 2.5 

# Stiffness ramping system for smooth teleop initialization
stiffness_ramp_active = False
stiffness_ramp_start_time = 0.0
stiffness_ramp_duration = 5.0  # 5 seconds to ramp from low to target stiffness
min_stiffness_factor = 0.1  # Start at 10% of target stiffness (very low but not zero)
target_stiffness_factor = 1.0  # Target stiffness factor

# Joint position smoothing parameters
joint_smoothing_factor = 0.1  # How much to keep of previous value (0.0 = no smoothing, 0.9 = heavy smoothing)
filtered_joint_positions = {}  # Store smoothed joint positions

# Simple finger command timing
last_finger_command_time = 0.0
finger_command_interval = 0.05  # 20Hz - faster than before but slower than 100Hz

# Latency tracking variables
latency_window_size = 100  # Track last 100 messages for latency stats
message_latencies = deque(maxlen=latency_window_size)
joint_position_latencies = deque(maxlen=latency_window_size)
total_latency_sum = 0.0
total_latency_count = 0
min_latency = float('inf')
max_latency = 0.0
last_latency_report_time = 0.0

def prepare_robot(client: B1LocoClient, communication_mode="websocket", ws_host="localhost", ws_port=8765, zmq_address="tcp://localhost:5555"):
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
    
    # Send current joint positions to teleoperator for initialization
    send_robot_joint_positions_to_teleoperator(current_positions, communication_mode, ws_host, ws_port, zmq_address)
    
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
    
    # Base gains for teleoperation
    kps = [
        5.0, 5.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100.,                                        # Waist
        350., 350., 180., 350., 450., 450.,          # Left leg
        350., 350., 180., 350., 450., 450.,          # Right leg
    ]
    
    kds = [
        0.1, 0.1,                                   # Head
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,         # Left arm
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,         # Right arm
        5.0,                                        # Waist
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Left leg
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Right leg
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

def calculate_current_stiffness_factor() -> float:
    """Calculate current stiffness factor based on ramping progress"""
    global stiffness_ramp_active, stiffness_ramp_start_time, stiffness_ramp_duration
    global min_stiffness_factor, target_stiffness_factor
    
    if not stiffness_ramp_active:
        return target_stiffness_factor
    
    # Calculate ramp progress (0.0 to 1.0)
    current_time = time.time()
    elapsed_time = current_time - stiffness_ramp_start_time
    ramp_progress = min(1.0, elapsed_time / stiffness_ramp_duration)
    
    # End ramping when complete
    if ramp_progress >= 1.0:
        stiffness_ramp_active = False
        print(f"🎯 Stiffness ramping complete - now at target stiffness ({target_stiffness_factor:.1f})")
    
    # Smooth ramping using a sine curve for gentler transitions
    smooth_progress = 0.5 * (1.0 - math.cos(math.pi * ramp_progress))
    current_factor = min_stiffness_factor + (target_stiffness_factor - min_stiffness_factor) * smooth_progress
    
    return current_factor

def start_stiffness_ramping():
    """Start the stiffness ramping process"""
    global stiffness_ramp_active, stiffness_ramp_start_time, arm_stiffness_factor
    
    stiffness_ramp_active = True
    stiffness_ramp_start_time = time.time()
    
    print(f"🔄 Starting stiffness ramping:")
    print(f"   Duration: {stiffness_ramp_duration:.1f} seconds")
    print(f"   Start factor: {min_stiffness_factor:.1f} ({min_stiffness_factor*100:.0f}% of base values)")
    print(f"   Target factor: {target_stiffness_factor:.1f} ({target_stiffness_factor*100:.0f}% of target values)")
    print(f"   Arm stiffness factor: {arm_stiffness_factor:.1f} (applied to target values)")
    print(f"   Example: Base arm kp=15.0 → Target arm kp={15.0*arm_stiffness_factor:.1f}")
    print(f"   Ramping: {15.0*min_stiffness_factor:.1f} → {15.0*arm_stiffness_factor:.1f} over {stiffness_ramp_duration:.1f}s")
    print(f"   Curve: Smooth sine-based ramping for gentle transitions")

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
        "Waist": 16
    }
    return joint_mapping.get(joint_name)

def teleoperation_control_loop(client: B1LocoClient):
    """Main teleoperation control loop running at 100Hz"""
    global teleop_active, latest_joint_data, latest_hand_data, current_positions, lower_body_positions, arm_stiffness_factor
    global joint_smoothing_factor, filtered_joint_positions, last_finger_command_time, finger_command_interval
    global stiffness_ramp_active, stiffness_ramp_start_time, stiffness_ramp_duration
    global min_stiffness_factor, target_stiffness_factor

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
        5.0, 5.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100.,                                        # Waist
        350., 350., 180., 350., 450., 450.,          # Left leg
        350., 350., 180., 350., 450., 450.,          # Right leg
    ]
    
    base_kds = [
        0.1, 0.1,                                   # Head
        1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,         # Left arm
        1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,         # Right arm
        5.0,                                        # Waist
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Left leg
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Right leg
    ]
    
    # Calculate target stiffness values that include arm_stiffness_factor
    target_kps = base_kps.copy()
    target_kds = base_kds.copy()
    
    # Apply arm_stiffness_factor to arm joints (2-8 and 9-15) and waist (16)
    for i in range(2, 17):  # Arms and waist
        target_kps[i] = base_kps[i] * arm_stiffness_factor
        target_kds[i] = base_kds[i]  # Damping doesn't get arm_stiffness_factor
    
    print(f"🎯 Target stiffness values (including arm_stiffness_factor={arm_stiffness_factor}):")
    print(f"   Left arm kp: {target_kps[2]:.1f}, {target_kps[3]:.1f}, {target_kps[4]:.1f}, {target_kps[5]:.1f}, {target_kps[6]:.1f}, {target_kps[7]:.1f}, {target_kps[8]:.1f}")
    print(f"   Right arm kp: {target_kps[9]:.1f}, {target_kps[10]:.1f}, {target_kps[11]:.1f}, {target_kps[12]:.1f}, {target_kps[13]:.1f}, {target_kps[14]:.1f}, {target_kps[15]:.1f}")
    print(f"   Waist kp: {target_kps[16]:.1f}")
    
    # Start stiffness ramping for smooth teleop initialization
    start_stiffness_ramping()
    
    start_time = time.time()
    command_count = 0
    
    try:
        while teleop_active:
            loop_start_time = time.time()
            
            # Calculate current stiffness factor (ramps up over time)
            current_stiffness_factor = calculate_current_stiffness_factor()
            
            # Calculate dynamic gains by ramping from minimum to target values
            dynamic_kps = base_kps.copy()
            dynamic_kds = base_kds.copy()
            
            # For joints that have modified target values, ramp from min to target
            for i in range(29):
                if i < len(target_kps):
                    # Ramp from (base * min_factor) to target value
                    min_kp = base_kps[i] * min_stiffness_factor
                    dynamic_kps[i] = min_kp + (target_kps[i] - min_kp) * current_stiffness_factor
                    
                    min_kd = base_kds[i] * min_stiffness_factor  
                    dynamic_kds[i] = min_kd + (target_kds[i] - min_kd) * current_stiffness_factor
            
            # Initialize all motor commands with dynamic gains
            for i in range(29):
                low_cmd.motor_cmd[i].q = current_positions[i]
                low_cmd.motor_cmd[i].dq = 0.0
                low_cmd.motor_cmd[i].kp = dynamic_kps[i] if i < len(dynamic_kps) else base_kps[i]
                low_cmd.motor_cmd[i].kd = dynamic_kds[i] if i < len(dynamic_kds) else base_kds[i]
                low_cmd.motor_cmd[i].tau = 0.0
            
            # Apply joint commands from WebSocket using organized data with smoothing
            if latest_joint_data:
                # Use organized upper_body data (no fallback needed)
                if 'upper_body' in latest_joint_data:
                    upper_body_joints = latest_joint_data['upper_body']
                    for joint_name, joint_value in upper_body_joints.items():
                        joint_index = map_joint_name_to_index(joint_name)
                        if joint_index is not None and joint_index <= 16:  # Upper body only
                            # Apply smoothing filter
                            if joint_name in filtered_joint_positions:
                                # filtered_joint_value[i] = self.filtered_joint_value[i] * 0.8 + self.joint_value[i] * 0.2
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
            
            # Send hand commands with conservative timing and thresholds
            current_time = time.time()
            if latest_hand_data and (current_time - last_finger_command_time) >= finger_command_interval:
                # Control left hand
                if 'left_hand' in latest_hand_data and latest_hand_data['left_hand']:
                    left_finger_params = map_hand_data_to_finger_params(latest_hand_data['left_hand'], 'left')
                    if left_finger_params:
                        print(f"🤚 LEFT HAND: Sending {len(left_finger_params)} finger commands at t={current_time:.3f}")
                        for param in left_finger_params:
                            print(f"   → LEFT finger {param.seq}: angle={param.angle}, force={param.force}, speed={param.speed}")
                        res_left = client.ControlDexterousHand(left_finger_params, B1HandIndex.kLeftHand)
                        if res_left != 0:
                            print(f"❌ Left hand command failed: {res_left}")
                        else:
                            print(f"✅ Left hand command sent successfully")
                
                # Control right hand
                if 'right_hand' in latest_hand_data and latest_hand_data['right_hand']:
                    right_finger_params = map_hand_data_to_finger_params(latest_hand_data['right_hand'], 'right')
                    if right_finger_params:
                        print(f"🤚 RIGHT HAND: Sending {len(right_finger_params)} finger commands at t={current_time:.3f}")
                        for param in right_finger_params:
                            print(f"   → RIGHT finger {param.seq}: angle={param.angle}, force={param.force}, speed={param.speed}")
                        res_right = client.ControlDexterousHand(right_finger_params, B1HandIndex.kRightHand)
                        if res_right != 0:
                            print(f"❌ Right hand command failed: {res_right}")
                        else:
                            print(f"✅ Right hand command sent successfully")
                
                # Update timing
                last_finger_command_time = current_time
            
            command_count += 1
            
            # Print statistics every 10 seconds
            if command_count % 1000 == 0:  # Every 10 seconds at 100Hz
                elapsed_time = time.time() - start_time
                actual_hz = command_count / elapsed_time
                
                # Get current latency statistics
                latency_stats = get_latency_statistics()
                latency_report = format_latency_report(latency_stats, detailed=False)
                
                # Get current stiffness for reporting
                current_stiffness_for_report = calculate_current_stiffness_factor()
                
                print(f"📊 Teleoperation stats: {actual_hz:.1f}Hz, {command_count} commands sent")
                print(f"   Latest joint data keys: {list(latest_joint_data.keys()) if latest_joint_data else 'None'}")
                print(f"   Latest hand data: L={len(latest_hand_data.get('left_hand', {}))}, R={len(latest_hand_data.get('right_hand', {}))}")
                print(f"   Stiffness: {current_stiffness_for_report:.2f} {'(ramping)' if stiffness_ramp_active else '(stable)'}")
                print(f"   Arm stiffness factor: {arm_stiffness_factor}")
                print(f"   Joint smoothing factor: {joint_smoothing_factor}")
                print(f"   Filtered joints: {len(filtered_joint_positions)}")
                print(f"   Finger control: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) with current reduction")
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
        
        # Get final stiffness factor
        final_stiffness_factor = calculate_current_stiffness_factor()
        
        print(f"📊 Final teleoperation stats:")
        print(f"   Duration: {elapsed_time:.1f}s")
        print(f"   Commands sent: {command_count}")
        print(f"   Average frequency: {actual_hz:.1f}Hz")
        print(f"   Final stiffness factor: {final_stiffness_factor:.2f} {'(complete)' if not stiffness_ramp_active else '(ramping)'}")
        print(f"   Stiffness ramping: {min_stiffness_factor:.1f} → {target_stiffness_factor:.1f} over {stiffness_ramp_duration:.1f}s")
        print(f"   Arm stiffness factor used: {arm_stiffness_factor}")
        print(f"   Joint smoothing factor used: {joint_smoothing_factor}")
        print(f"   Total filtered joints: {len(filtered_joint_positions)}")
        print(f"   Finger timing used: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) + inversion + light smoothing + reasonable thresholds")
        
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

def start_teleoperation(client: B1LocoClient, communication_mode: str = "websocket", 
                      ws_host: str = "localhost", ws_port: int = 8765, 
                      zmq_address: str = "tcp://localhost:5555"):
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
    
    if communication_mode == "websocket":
        print(f"   WebSocket server: {ws_host}:{ws_port}")
        
        # Create WebSocket client
        ws_client = WebSocketClient(ws_host, ws_port)
        
        # Create wrapper classes for shared data (since we need to pass by reference)
        class SharedBool:
            def __init__(self, value):
                self.value = value
        
        class SharedFloat:
            def __init__(self, value):
                self.value = value
        
        teleop_active_ref = SharedBool(teleop_active)
        total_latency_sum_ref = SharedFloat(total_latency_sum)
        total_latency_count_ref = SharedFloat(total_latency_count)
        min_latency_ref = SharedFloat(min_latency)
        max_latency_ref = SharedFloat(max_latency)
        
        # Start WebSocket client in a separate thread
        websocket_thread = threading.Thread(
            target=lambda: asyncio.run(ws_client.handle_connection(
                teleop_active_ref, latest_joint_data, latest_hand_data,
                message_latencies, joint_position_latencies,
                total_latency_sum_ref, total_latency_count_ref,
                min_latency_ref, max_latency_ref
            )),
            daemon=True
        )
        websocket_thread.start()
        
    elif communication_mode == "zeromq":
        print(f"   ZeroMQ publisher: {zmq_address}")
        
        # Create ZeroMQ client
        zmq_client = ZeroMQClient(zmq_address)
        
        # Create wrapper classes for shared data (since we need to pass by reference)
        class SharedBool:
            def __init__(self, value):
                self.value = value
        
        class SharedFloat:
            def __init__(self, value):
                self.value = value
        
        teleop_active_ref = SharedBool(teleop_active)
        total_latency_sum_ref = SharedFloat(total_latency_sum)
        total_latency_count_ref = SharedFloat(total_latency_count)
        min_latency_ref = SharedFloat(min_latency)
        max_latency_ref = SharedFloat(max_latency)
        
        # Start ZeroMQ subscriber in a separate thread
        zeromq_thread = threading.Thread(
            target=lambda: zmq_client.handle_subscriber(
                teleop_active_ref, latest_joint_data, latest_hand_data,
                message_latencies, joint_position_latencies,
                total_latency_sum_ref, total_latency_count_ref,
                min_latency_ref, max_latency_ref
            ),
            daemon=True
        )
        zeromq_thread.start()
    
    else:
        print(f"❌ Unknown communication mode: {communication_mode}")
        teleop_active = False
        return False
    
    print("   Control frequency: 100Hz")
    print("   Upper body: Following commands")
    print("   Lower body: Maintaining current positions")
    print("   Latency tracking: Enabled (requires timestamps in messages)")
    print(f"   Finger control: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) with current reduction")

    print("Teleoperation Commands:")
    print(f"  teleop  - Start teleoperation mode")
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
    print("  2. Then run 'teleop' to start receiving commands")
    print("  3. Make sure the communication endpoint is running")
    print("  4. Upper body joints will follow commands")
    print("  5. Hand joints will follow hand commands")
    print("  6. Lower body joints will maintain their current positions")
    print("  7. Latency tracking monitors network performance (requires timestamps)")
    print("=" * 60)
    print()
    print("📊 TELEOPERATION FEATURES:")
    print("  • Real-time joint position control at 100Hz")
    print("  • Hand/finger control with current reduction for power efficiency") 
    print("  • Joint position smoothing to reduce jitter")
    print(f"  • Finger control: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) with current reduction")
    print("  • Per-finger timing prevents rapid-fire commands")
    print("  • Reasonable thresholds allow normal movements while filtering noise")
    print("  • Configurable arm stiffness scaling")
    print(f"  • Smart stiffness ramping: {min_stiffness_factor:.1f} → {target_stiffness_factor:.1f} over {stiffness_ramp_duration:.1f}s for smooth initialization")
    print("  • Network latency monitoring and quality assessment")
    print("  • Periodic statistics reporting every 10 seconds")
    print("=" * 60)

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

def send_robot_joint_positions_to_teleoperator(joint_positions, communication_mode="websocket", ws_host="localhost", ws_port=8765, zmq_address="tcp://localhost:5555"):
    """Send current robot joint positions to teleoperator for initialization"""
    print("📤 Sending robot joint positions to teleoperator...")
    
    # Create initialization message using utility function
    init_message = create_robot_joint_initialization_message(joint_positions)
    
    # Send via the specified communication mode
    if communication_mode == "websocket":
        send_initialization_message_websocket(init_message, ws_host, ws_port)
    elif communication_mode == "zeromq":
        send_initialization_message_zeromq(init_message, zmq_address)
    else:
        print(f"⚠️ Unknown communication mode: {communication_mode}")
        print("   Trying both WebSocket and ZeroMQ...")
        send_initialization_message_websocket(init_message, ws_host, ws_port)
        send_initialization_message_zeromq(init_message, zmq_address)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface [--communication {{'websocket', 'zeromq'}}] [--ws-host HOST] [--ws-port PORT] [--zmq-address ADDRESS]")
        sys.exit(-1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot control with teleoperation')
    parser.add_argument('network_interface', help='Network interface for robot communication')
    parser.add_argument('--communication', type=str, choices=['websocket', 'zeromq'], default='websocket',
                       help='Communication method: websocket or zeromq (default: websocket)')
    parser.add_argument('--ws-host', type=str, default='localhost',
                       help='WebSocket server host for teleoperation (default: localhost)')
    parser.add_argument('--ws-port', type=int, default=8765,
                       help='WebSocket server port for teleoperation (default: 8765)')
    parser.add_argument('--zmq-address', type=str, default='tcp://localhost:5555',
                       help='ZeroMQ address for teleoperation (default: tcp://localhost:5555)')
    
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
    if args.communication == "websocket":
        print(f"  teleop  - Start teleoperation mode (connects to {args.ws_host}:{args.ws_port})")
    elif args.communication == "zeromq":
        print(f"  teleop  - Start teleoperation mode (connects to {args.zmq_address})")
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
    print("  2. Then run 'teleop' to start receiving commands")
    print(f"  3. Make sure the communication endpoint is running")
    print("  4. Upper body joints will follow commands")
    print("  5. Hand joints will follow hand commands")
    print("  6. Lower body joints will maintain their current positions")
    print("  7. Latency tracking monitors network performance (requires timestamps)")
    print("=" * 60)
    print()
    print("📊 COMMUNICATION METHODS:")
    print("  • WebSocket: Traditional WebSocket communication")
    print("  • ZeroMQ: High-performance messaging for lower latency")
    print(f"  • Current mode: {args.communication.upper()}")
    print("  • Both methods support latency tracking for benchmarking")
    print("=" * 60)

    try:
        while True:
            need_print = False
            input_cmd = input().strip()
            if input_cmd:
                if input_cmd == 'mc':
                    print("Preparing robot for custom mode...")
                    success = prepare_robot(client, args.communication, args.ws_host, args.ws_port, args.zmq_address)
                    if not success:
                        print("Failed to prepare robot for custom mode")
                elif input_cmd == 'teleop':
                    print(f"Starting teleoperation mode ({args.communication})...")
                    success = start_teleoperation(client, args.communication, 
                                                 args.ws_host, args.ws_port, args.zmq_address)
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
                elif input_cmd == "hand":
                    hand_oscillation(client)
                elif input_cmd == "hand-open":
                    open_hands(client)
                elif input_cmd == "hand-close":
                    close_hands(client)

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

import time
import math
from typing import Dict, Optional
from booster_robotics_sdk_python import B1LocoClient, B1HandIndex, DexterousFingerParameter, B1LowHandDataScriber


# Current reduction settings for finger control
finger_force_reduction = 1.0  # Reduce force to 30% to minimize current draw
finger_speed_reduction = 1.0  # Reduce speed to 50% to minimize current draw
finger_max_force = 600  # Maximum force to prevent high current draw
finger_max_speed = 800  # Maximum speed to prevent high current draw

# Simple finger command timing
finger_command_threshold = 150  # 15% of range - allows normal movements
last_finger_positions = {'left': {}, 'right': {}}  # Track last commanded positions per finger per hand
smoothed_finger_positions = {'left': {}, 'right': {}}  # Track smoothed finger positions  
finger_smoothing_factor = 0.1  # Light smoothing to filter noise but stay responsive

# Per-finger timing to prevent rapid commands to same finger
finger_last_command_time = {'left': {}, 'right': {}}  # Track last command time per finger
finger_min_interval = 0.2  # Minimum 200ms between commands to same finger (much more reasonable)


def apply_current_reduction_to_finger_command(finger_param: DexterousFingerParameter) -> DexterousFingerParameter:
    """Apply current reduction settings to finger commands to minimize power draw"""
    global finger_force_reduction, finger_speed_reduction, finger_max_force, finger_max_speed
    
    # Reduce force and speed to minimize current draw
    finger_param.force = min(int(finger_param.force * finger_force_reduction), finger_max_force)
    finger_param.speed = min(int(finger_param.speed * finger_speed_reduction), finger_max_speed)
    
    return finger_param


def map_hand_data_to_finger_params(hand_data: Dict[str, float], hand_side: str) -> list:
    """Map hand joint data to finger parameters with simple current-reduction focused approach"""
    global finger_command_threshold, last_finger_positions, smoothed_finger_positions, finger_smoothing_factor
    global finger_last_command_time, finger_min_interval
    
    finger_params = []
    current_time = time.time()
    
    # Updated mapping for new joint names (based on the WebSocket hardware interface)
    finger_mapping = {
        "thumb_proximal_yaw_joint": 5,       # Thumb rotation
        "thumb_proximal_pitch_joint": 4,     # Thumb bending  
        "index_proximal_joint": 3,           # Index finger
        "middle_proximal_joint": 2,          # Middle finger
        "ring_proximal_joint": 1,            # Ring finger
        "pinky_proximal_joint": 0,           # Little finger
    }
    
    # Initialize hand tracking if not exists
    if hand_side not in last_finger_positions:
        last_finger_positions[hand_side] = {}
    if hand_side not in smoothed_finger_positions:
        smoothed_finger_positions[hand_side] = {}
    if hand_side not in finger_last_command_time:
        finger_last_command_time[hand_side] = {}
    
    # Process finger data with simple approach - focus on current reduction
    for joint_name, joint_value in hand_data.items():
        finger_seq = finger_mapping.get(joint_name)
        
        if finger_seq is not None:
            # INVERT the finger command: 1000-x (as requested)
            inverted_value = 1000 - joint_value
            raw_angle = max(0, min(1000, int(inverted_value)))
            
            # Apply light smoothing only
            if finger_seq in smoothed_finger_positions[hand_side]:
                smoothed_angle = (
                    smoothed_finger_positions[hand_side][finger_seq] * finger_smoothing_factor + 
                    raw_angle * (1.0 - finger_smoothing_factor)
                )
            else:
                # Initialize with current value
                smoothed_angle = raw_angle
            
            # Update smoothed position
            smoothed_finger_positions[hand_side][finger_seq] = smoothed_angle
            target_angle = int(smoothed_angle)
            
            # Check per-finger timing - don't command same finger too frequently
            last_cmd_time = finger_last_command_time[hand_side].get(finger_seq, 0)
            if (current_time - last_cmd_time) < finger_min_interval:
                # Too soon since last command to this finger - skip
                continue
            
            # Simple threshold check
            last_angle = last_finger_positions[hand_side].get(finger_seq, 500)
            change_amount = abs(target_angle - last_angle)
            
            if change_amount >= finger_command_threshold:
                # Create finger command with REDUCED force and speed for current reduction
                finger_param = DexterousFingerParameter()
                finger_param.seq = finger_seq
                finger_param.angle = target_angle
                finger_param.force = 600  # Will be reduced by apply_current_reduction_to_finger_command
                finger_param.speed = 800  # Will be reduced by apply_current_reduction_to_finger_command
                
                # Apply current reduction settings
                finger_param = apply_current_reduction_to_finger_command(finger_param)
                finger_params.append(finger_param)
                
                # Update tracking for this finger
                last_finger_positions[hand_side][finger_seq] = target_angle
                finger_last_command_time[hand_side][finger_seq] = current_time
                
                print(f"🤚 {hand_side.upper()} finger {finger_seq}: MOVE {last_angle}→{target_angle} (change: {change_amount}) [force: {finger_param.force}, speed: {finger_param.speed}] - CURRENT REDUCED")
    
    return finger_params


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
    """Perform sinusoidal hand movement on both hands"""
    # 定义一个 名为 finger_params 的数组，用于存储每个手指的参数
    print("Reading hand data for both hands...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)

    # Base positions for each finger - starting with everything open (high angles)
    base_positions = [900, 850, 800, 850, 900, 950]
    
    # Sinusoidal parameters
    amplitude = 300  # How much the fingers oscillate
    frequency = 1.0  # Hz - oscillations per second
    duration = 10.0  # Total duration in seconds
    
    print(f"Starting sinusoidal hand movement on BOTH hands for {duration} seconds...")
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
                
                # Clamp angle to reasonable bounds (0-1000)
                target_angle = max(0, min(1000, target_angle))
                
                finger_param = DexterousFingerParameter()
                finger_param.seq = i
                finger_param.angle = int(target_angle)
                finger_param.force = 600  # Base force - will be reduced
                finger_param.speed = 800  # Base speed - will be reduced
                
                # Apply current reduction for consistency with teleoperation
                finger_param = apply_current_reduction_to_finger_command(finger_param)
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


def close_hands(client: B1LocoClient):
    """Close all fingers on both hands"""
    print("Reading hand data for both hands...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)
    
    # Closed positions for each finger (low angles = closed)
    closed_positions = [100, 150, 200, 150, 100, 50]  # Tight fist positions
    
    print("Closing all fingers on BOTH hands...")
    print("Using maximum speed, low force (10)")
    
    finger_params = []
    
    # Set closed angles for each finger
    for i in range(6):
        finger_param = DexterousFingerParameter()
        finger_param.seq = i
        finger_param.angle = closed_positions[i]
        finger_param.force = 600  # Base force - will be reduced
        finger_param.speed = 600  # Base speed - will be reduced
        
        # Apply current reduction for consistency with teleoperation
        finger_param = apply_current_reduction_to_finger_command(finger_param)
        finger_params.append(finger_param)
    
    # Send command to RIGHT hand
    res_right = client.ControlDexterousHand(finger_params, B1HandIndex.kRightHand)
    if res_right != 0:
        print(f"Right hand close command failed: error = {res_right}")
    else:
        print("✅ Right hand closed successfully")
    
    # Send command to LEFT hand
    res_left = client.ControlDexterousHand(finger_params, B1HandIndex.kLeftHand)
    if res_left != 0:
        print(f"Left hand close command failed: error = {res_left}")
    else:
        print("✅ Left hand closed successfully")
    
    # Print final angles
    angles_str = ", ".join([f"{p.angle}" for p in finger_params])
    print(f"Final closed positions: [{angles_str}]")
    
    print("Hand closing completed on both hands")
    print("Reading final hand data...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)


def open_hands(client: B1LocoClient):
    """Open all fingers on both hands"""
    print("Reading hand data for both hands...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0)
    
    # Open positions for each finger (high angles = open)
    open_positions = [900, 850, 800, 850, 900, 950]  # Fully open positions
    
    print("Opening all fingers on BOTH hands...")
    print("Using maximum speed, low force (10)")
    
    finger_params = []
    
    # Set open angles for each finger
    for i in range(6):
        finger_param = DexterousFingerParameter()
        finger_param.seq = i
        finger_param.angle = open_positions[i]
        finger_param.force = 600  # Base force - will be reduced
        finger_param.speed = 600  # Base speed - will be reduced
        
        # Apply current reduction for consistency with teleoperation
        finger_param = apply_current_reduction_to_finger_command(finger_param)
        finger_params.append(finger_param)
    
    # Send command to RIGHT hand
    res_right = client.ControlDexterousHand(finger_params, B1HandIndex.kRightHand)
    if res_right != 0:
        print(f"Right hand open command failed: error = {res_right}")
    else:
        print("✅ Right hand opened successfully")
    
    # Send command to LEFT hand
    res_left = client.ControlDexterousHand(finger_params, B1HandIndex.kLeftHand)
    if res_left != 0:
        print(f"Left hand open command failed: error = {res_left}")
    else:
        print("✅ Left hand opened successfully")
    
    # Print final angles
    angles_str = ", ".join([f"{p.angle}" for p in finger_params])
    print(f"Final open positions: [{angles_str}]")
    
    print("Hand opening completed on both hands")
    print("Reading final hand data...")
    read_hand_data(B1HandIndex.kRightHand, 1.0)
    read_hand_data(B1HandIndex.kLeftHand, 1.0) 
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode, B1HandIndex, GripperControlMode, Position, Orientation, Posture, GripperMotionParameter, Quaternion, Frame, Transform, DexterousFingerParameter, LowCmd, LowCmdType, MotorCmd, B1LowCmdPublisher, B1LowStateSubscriber, B1LowHandDataScriber
import sys, time, random, math

def prepare_robot(client: B1LocoClient):
    """Send robot to a stable prepare pose before switching to custom mode"""
    
    # First, read current robot state
    print("Reading current robot position...")
    current_positions = [0.0] * 29
    positions_received = False
    
    def state_handler(low_state_msg):
        nonlocal current_positions, positions_received
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

def move_right_shoulder_pitch_forward(client: B1LocoClient, target_angle_degrees=-30.0):
    """Move right shoulder pitch to a target position (forward)"""
    
    # Convert degrees to radians
    target_angle_rad = target_angle_degrees * 3.14159 / 180.0
    
    print(f"Moving right shoulder pitch to {target_angle_degrees}° ({target_angle_rad:.3f} rad)")
    
    # First, read current robot state
    current_positions = [0.0] * 29
    positions_received = False
    
    def state_handler(low_state_msg):
        nonlocal current_positions, positions_received
        if len(low_state_msg.motor_state_serial) >= 29:
            for i in range(29):
                current_positions[i] = low_state_msg.motor_state_serial[i].q
            positions_received = True
    
    # Subscribe to get current state
    state_subscriber = B1LowStateSubscriber(state_handler)
    state_subscriber.InitChannel()
    
    # Wait for state data
    timeout = 2.0
    start_time = time.time()
    while not positions_received and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    state_subscriber.CloseChannel()
    
    if not positions_received:
        print("Warning: Could not read current positions")
        return False
    
    # Create low-level command publisher
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    
    # Create and initialize the command
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    
    # Initialize motor commands
    motor_cmds = [MotorCmd() for _ in range(29)]
    low_cmd.motor_cmd = motor_cmds
    
    # Updated gains matching the prepare_robot function
    kps = [
        2.0, 2.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100.,                                        # Waist
        350., 350., 180., 350., 450., 450.,          # Left leg
        350., 350., 180., 350., 450., 450.,          # Right leg
    ]
    
    kds = [
        0.3, 0.3,                                   # Head
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Left arm
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,         # Right arm
        5.0,                                        # Waist
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Left leg
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Right leg
    ]
    
    # Set all joints to hold current positions
    for i in range(29):
        low_cmd.motor_cmd[i].q = current_positions[i]
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = kps[i]
        low_cmd.motor_cmd[i].kd = kds[i]
        low_cmd.motor_cmd[i].tau = 0.0
    
    # Override right shoulder pitch (index 9) with target position
    right_shoulder_pitch_index = 9
    old_angle = current_positions[right_shoulder_pitch_index]
    low_cmd.motor_cmd[right_shoulder_pitch_index].q = target_angle_rad
    
    print(f"Right shoulder pitch: {old_angle:.3f} rad ({old_angle * 180 / 3.14159:.1f}°) -> {target_angle_rad:.3f} rad ({target_angle_degrees:.1f}°)")
    
    # Send command
    low_cmd_publisher.Write(low_cmd)
    
    # Clean up
    low_cmd_publisher.CloseChannel()
    
    print("Right shoulder pitch movement command sent!")
    return True

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        sys.exit(-1)

    ChannelFactory.Instance().Init(0, sys.argv[1])

    client = B1LocoClient()
    client.Init()
    x, y, z, yaw, pitch = 0.0, 0.0, 0.0, 0.0, 0.0
    res = 0
    hand_action_count = 0

    print("=" * 60)
    print("ROBOT CONTROL COMMANDS:")
    print("=" * 60)
    print("Mode Commands:")
    print("  mp   - Switch to Prepare mode")
    print("  md   - Switch to Damping mode") 
    print("  mw   - Switch to Walking mode")
    print("  mc   - Prepare robot and switch to Custom mode")
    print()
    print("Movement Commands (in Walking mode):")
    print("  w/a/s/d/q/e - Move robot")
    print("  stop        - Stop movement")
    print()
    print("Head Commands:")
    print("  hd/hu/hr/hl - Move head down/up/right/left")  
    print("  ho          - Center head")
    print()
    print("Arm Commands:")
    print("  rsp         - Move right shoulder pitch forward")
    print()
    print("Hand Commands:")
    print("  hand - Hand oscillation")
    print()
    print("NOTE: 'mc' prepares robot and switches to custom mode.")
    print("      Robot should maintain pose automatically (like deploy.py).")
    print("=" * 60)
    print()

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
                elif input_cmd == "rsp":
                    print("Moving right shoulder pitch forward...")
                    success = move_right_shoulder_pitch_forward(client)
                    if not success:
                        print("Failed to move right shoulder pitch")

                if need_print:
                    print(f"Param: {x} {y} {z}")
                    print(f"Head param: {pitch} {yaw}")

                if res != 0:
                    print(f"Request failed: error = {res}")

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
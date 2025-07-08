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
import torch
import yaml
from concurrent.futures import ThreadPoolExecutor
import queue

# ===== POLICY CLASS =====
class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = (
            self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[7] = (
            self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[8] = (
            self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11:23] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[23:35] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[35:47] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets

# ===== UTILITY FUNCTIONS =====
def rotate_vector_inverse_rpy(roll, pitch, yaw, v):
    """Rotate vector by inverse RPY transformation"""
    # Simple rotation matrix inverse transformation
    # This is a simplified version - in practice you'd use proper rotation matrices
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Rotation matrix inverse (transpose of rotation matrix)
    R_inv = np.array([
        [cos_y*cos_p, sin_y*cos_p, -sin_p],
        [cos_y*sin_p*sin_r - sin_y*cos_r, sin_y*sin_p*sin_r + cos_y*cos_r, cos_p*sin_r],
        [cos_y*sin_p*cos_r + sin_y*sin_r, sin_y*sin_p*cos_r - cos_y*sin_r, cos_p*cos_r]
    ])
    
    return np.dot(R_inv, v)

def map_policy_indices_to_robot_indices(policy_indices):
    """Map joint indices from 23-joint policy space to 29-joint robot space"""
    robot_indices = []
    
    for policy_idx in policy_indices:
        if 0 <= policy_idx <= 1:
            # Head joints: direct mapping
            robot_indices.append(policy_idx)
        elif 2 <= policy_idx <= 5:
            # Left arm joints: policy 2-5 → robot 2-5 (first 4 of 7 arm joints)
            robot_indices.append(policy_idx)
        elif 6 <= policy_idx <= 9:
            # Right arm joints: policy 6-9 → robot 9-12 (first 4 of 7 arm joints)
            robot_indices.append(policy_idx + 3)  # 6→9, 7→10, 8→11, 9→12
        elif policy_idx == 10:
            # Waist joint: policy 10 → robot 16
            robot_indices.append(16)
        elif 11 <= policy_idx <= 22:
            # Leg joints: policy 11-22 → robot 17-28
            robot_indices.append(policy_idx + 6)  # 11→17, 12→18, ..., 22→28
        else:
            print(f"Warning: Invalid policy index {policy_idx}, skipping")
    
    return robot_indices

# ===== WALKING CONTROL CLASS =====
class WalkingController:
    """Handles walking control using RL policy"""
    
    def __init__(self, cfg_file):
        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Initialize policy
        self.policy = Policy(cfg=self.cfg)
        
        # State tracking for 29-joint robot
        self.dof_pos_29 = np.zeros(29, dtype=np.float32)  # 29 joints
        self.dof_vel_29 = np.zeros(29, dtype=np.float32)  # 29 joints
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        
        # Policy targets (23 joints in policy space)
        self.policy_targets_23 = np.zeros(23, dtype=np.float32)
        
        # Robot targets (29 joints in robot space)
        self.robot_targets_29 = np.zeros(29, dtype=np.float32)
        
        # Default positions for 29-joint robot (will be set from current robot state)
        self.default_29_joint_positions = np.zeros(29, dtype=np.float32)
        
        # Walking commands
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.vyaw_cmd = 0.0
        
        # Timing
        self.next_inference_time = 0.0
        self.last_time = time.time()
        
        print("Walking controller initialized")
        print(f"Policy expects 23 joints, robot has 29 joints")
        print(f"Upper body mask: joints 0-16 (head, arms, waist)")
        print(f"Leg mapping: robot joints 17-28 -> policy joints 11-22")
    
    def set_default_positions(self, positions):
        """Set default positions for the 29-joint robot"""
        if len(positions) == 29:
            self.default_29_joint_positions = np.copy(positions)
            print("Set 29-joint default positions for walking controller")
        else:
            print(f"Warning: Expected 29 positions, got {len(positions)}")
    
    def update_robot_state(self, dof_pos_29, dof_vel_29, base_ang_vel, projected_gravity):
        """Update robot state for policy inference"""
        self.dof_pos_29 = np.copy(dof_pos_29)
        self.dof_vel_29 = np.copy(dof_vel_29)
        self.base_ang_vel = np.copy(base_ang_vel)
        self.projected_gravity = np.copy(projected_gravity)
    
    def set_walking_commands(self, vx=0.0, vy=0.0, vyaw=0.0):
        """Set walking velocity commands"""
        self.vx_cmd = vx
        self.vy_cmd = vy
        self.vyaw_cmd = vyaw
        print(f"Walking commands: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
    
    def map_29_to_23_joints(self, dof_pos_29, dof_vel_29):
        """Map 29-joint robot state to 23-joint policy space with upper body masking"""
        # Create 23-joint arrays
        dof_pos_23 = np.zeros(23, dtype=np.float32)
        dof_vel_23 = np.zeros(23, dtype=np.float32)
        
        # Get policy default positions (23 joints)
        policy_defaults = self.policy.default_dof_pos
        
        # Map robot structure to policy structure:
        # Robot (29): 0-1 head, 2-8 L arm, 9-15 R arm, 16 waist, 17-28 legs  
        # Policy (23): 0-1 head, 2-5 L arm, 6-9 R arm, 10 waist, 11-22 legs
        
        # MASK UPPER BODY: Set to policy defaults (joints 0-10 in policy space)
        dof_pos_23[0:11] = policy_defaults[0:11]  # Head, arms, waist
        dof_vel_23[0:11] = 0.0  # Zero velocity for upper body
        
        # MAP LEGS: Robot legs 17-28 -> Policy legs 11-22
        dof_pos_23[11:23] = dof_pos_29[17:29]  # Map 12 leg joints
        dof_vel_23[11:23] = dof_vel_29[17:29]  # Map 12 leg velocities
        
        return dof_pos_23, dof_vel_23
    
    def map_23_to_29_targets(self, policy_targets_23):
        """Map 23-joint policy targets to 29-joint robot space"""
        # Initialize with current positions
        robot_targets_29 = np.copy(self.default_29_joint_positions)
        
        # KEEP UPPER BODY AT DEFAULT: joints 0-16 stay at default positions
        # (already set above)
        
        # MAP LEG TARGETS: Policy legs 11-22 -> Robot legs 17-28  
        robot_targets_29[17:29] = policy_targets_23[11:23]  # Map 12 leg joint targets
        
        return robot_targets_29
    
    def run_inference(self):
        """Run policy inference and return 29-joint targets"""
        current_time = time.time()
        
        # Check if it's time for inference
        if current_time < self.next_inference_time:
            return self.robot_targets_29
        
        # Update inference timing
        self.next_inference_time = current_time + self.policy.get_policy_interval()
        
        # Map 29-joint robot state to 23-joint policy space with masking
        dof_pos_23, dof_vel_23 = self.map_29_to_23_joints(self.dof_pos_29, self.dof_vel_29)
        
        # Run policy inference
        self.policy_targets_23 = self.policy.inference(
            time_now=current_time,
            dof_pos=dof_pos_23,
            dof_vel=dof_vel_23, 
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.vx_cmd,
            vy=self.vy_cmd,
            vyaw=self.vyaw_cmd
        )
        
        # Map policy targets back to 29-joint robot space
        self.robot_targets_29 = self.map_23_to_29_targets(self.policy_targets_23)
        
        return self.robot_targets_29

# ===== EXISTING GLOBALS =====
# Global variables for teleoperation
teleop_active = False
teleop_thread = None
latest_joint_data = {}
latest_hand_data = {}
robot_client = None
current_positions = [0.0] * 29
lower_body_positions = [0.0] * 12  # Joints 17-28
arm_stiffness_factor = 2.0  # Downscale arm stiffness by 0.8

# ===== NEW WALKING GLOBALS =====
walking_active = False
walking_controller = None
walking_thread = None

# ===== COMBINED CONTROL GLOBALS =====
combined_mode_active = False  # True when both walking and teleop are active
latest_walking_targets = np.zeros(29, dtype=np.float32)  # Latest walking targets for legs
latest_teleop_targets = np.zeros(29, dtype=np.float32)   # Latest teleop targets for upper body
combined_targets_lock = threading.Lock()  # Lock for thread-safe access to targets

# Joint position smoothing parameters
joint_smoothing_factor = 0.8  # How much to keep of previous value (0.0 = no smoothing, 0.9 = heavy smoothing)
filtered_joint_positions = {}  # Store smoothed joint positions

# Simple finger command timing
last_finger_command_time = 0.0
finger_command_interval = 0.05  # 20Hz - faster than before but slower than 100Hz

# More reasonable thresholds - allow normal movements but filter small noise
finger_command_threshold = 150  # 15% of range - allows normal movements
last_finger_positions = {'left': {}, 'right': {}}  # Track last commanded positions per finger per hand
smoothed_finger_positions = {'left': {}, 'right': {}}  # Track smoothed finger positions  
finger_smoothing_factor = 0.4  # Light smoothing to filter noise but stay responsive

# Per-finger timing to prevent rapid commands to same finger
finger_last_command_time = {'left': {}, 'right': {}}  # Track last command time per finger
finger_min_interval = 0.2  # Minimum 200ms between commands to same finger (much more reasonable)

# Finger feedback tracking to prevent vibrations
actual_finger_positions = {'left': {}, 'right': {}}  # Track actual finger positions from robot
finger_settled_positions = {'left': {}, 'right': {}}  # Track where fingers have settled
finger_settlement_threshold = 20  # If finger moves less than this, consider it "settled"
finger_settlement_tolerance = 50  # If commanded vs actual position is within this, stop commanding
last_hand_data_read_time = 0.0
hand_data_read_interval = 0.1  # Read hand positions every 100ms

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

    # Base positions for each finger - starting with everything open (high angles)
    base_positions = [900, 850, 800, 850, 900, 950]
    
    # Sinusoidal parameters
    amplitude = 300  # How much the fingers oscillate
    frequency = 1.0  # Hz - oscillations per second
    duration = 10.0  # Total duration in seconds
    
    print(f"Starting sinusoidal hand movement on BOTH hands for {duration} seconds...")
    print("Using maximum speed, low force (10), starting with fingers open")
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
                finger_param.force = 10  # Low force as requested
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
        finger_param.force = 10  # Low force as requested
        finger_param.speed = 1000  # Maximum speed
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
        finger_param.force = 800  # Low force as requested
        finger_param.speed = 1000  # Maximum speed
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

def update_actual_finger_positions():
    """This function is no longer needed with the simplified approach"""
    pass

def map_hand_data_to_finger_params(hand_data: Dict[str, float], hand_side: str) -> list:
    """Map hand joint data to finger parameters with conservative thresholds and per-finger timing"""
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
    
    # Process finger data with heavy filtering and conservative thresholds
    for joint_name, joint_value in hand_data.items():
        finger_seq = finger_mapping.get(joint_name)
        
        if finger_seq is not None:
            # INVERT the finger command: 1000-x (as requested)
            inverted_value = 1000 - joint_value
            raw_angle = max(0, min(1000, int(inverted_value)))
            
            # Apply heavy smoothing to filter out noise
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
            
            # Conservative threshold check - only send commands for MAJOR movements
            last_angle = last_finger_positions[hand_side].get(finger_seq, 500)
            change_amount = abs(target_angle - last_angle)
            
            if change_amount >= finger_command_threshold:
                # Create finger command (same settings as hand_oscillation)
                finger_param = DexterousFingerParameter()
                finger_param.seq = finger_seq
                finger_param.angle = target_angle
                finger_param.force = 800  # Same as hand_oscillation
                finger_param.speed = 1000  # Same as hand_oscillation
                finger_params.append(finger_param)
                
                # Update tracking for this finger
                last_finger_positions[hand_side][finger_seq] = target_angle
                finger_last_command_time[hand_side][finger_seq] = current_time
                
                print(f"🤚 {hand_side.upper()} finger {finger_seq}: BIG MOVE {last_angle}→{target_angle} (change: {change_amount})")
    
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
                        # Extract organized joint data (no fallback to all_joints)
                        message_data = data.get('data', {})
                        organized = message_data.get('organized', {})
                        
                        # Validate that we have the expected organized structure
                        if not organized:
                            print(f"⚠️ Warning: Received joint_positions message without organized data structure")
                            continue
                        
                        if 'upper_body' not in organized:
                            print(f"⚠️ Warning: Missing 'upper_body' in organized data")
                            continue
                        
                        # Update latest joint data with organized structure
                        latest_joint_data = organized.copy()
                        
                        # Extract hand data from organized data
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
    global joint_smoothing_factor, filtered_joint_positions, last_finger_command_time, finger_command_interval
    global finger_command_threshold, finger_deadband, last_finger_positions, smoothed_finger_positions, finger_smoothing_factor
    global actual_finger_positions, finger_settled_positions, finger_settlement_threshold, finger_settlement_tolerance
    global last_hand_data_read_time, hand_data_read_interval
    global combined_mode_active, latest_walking_targets, latest_teleop_targets, combined_targets_lock

    # Create low-level command publisher
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    
    # State tracking for torque control (similar to walking loop)
    robot_dof_pos_latest = np.zeros(29, dtype=np.float32)
    filtered_dof_target_teleop = np.zeros(29, dtype=np.float32)
    state_received = False
    
    def state_handler_teleop(low_state_msg):
        nonlocal robot_dof_pos_latest, state_received
        # Update latest positions for torque control
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            if i < 29:
                robot_dof_pos_latest[i] = motor.q
        state_received = True
    
    # Create state subscriber for teleop
    state_subscriber_teleop = B1LowStateSubscriber(state_handler_teleop)
    state_subscriber_teleop.InitChannel()
    
    # Initialize filtered targets
    filtered_dof_target_teleop[:] = current_positions
    
    # Create and initialize the command
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    
    # Initialize motor commands
    motor_cmds = [MotorCmd() for _ in range(29)]
    low_cmd.motor_cmd = motor_cmds
    
    # Base gains for teleoperation
    base_kps = [
        4.0, 4.0,                                    # Head
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Left arm
        15.0, 15.0, 15.0, 15.0, 10.0, 15.0, 10.0,   # Right arm  
        100.,                                        # Waist
        350., 350., 180., 350., 450., 450.,          # Left leg
        350., 350., 180., 350., 450., 450.,          # Right leg
    ]
    
    base_kds = [
        0.1, 0.1,                                   # Head
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,         # Left arm
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,         # Right arm
        5.0,                                        # Waist
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Left leg
        7.5, 7.5, 3., 5.5, 0.5, 0.5,               # Right leg
    ]
    
    # Torque limits for teleoperation (conservative values)
    torque_limits_teleop = [
        7.0, 7.0,                                   # Head
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # Left arm
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # Right arm
        30.0,                                       # Waist
        60.0, 25.0, 30.0, 60.0, 24.0, 15.0,        # Left leg
        60.0, 25.0, 30.0, 60.0, 24.0, 15.0,        # Right leg
    ]
    
    # Parallel mechanism indices for teleoperation - load from config
    try:
        # Load config file (same as walking mode)
        cfg_file = "test/T1.yaml"  # Default config file
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Get parallel mechanism indices from config and map to 29-joint space
        config_parallel_mech_indexes = cfg["mech"]["parallel_mech_indexes"]
        parallel_mech_indexes_teleop = map_policy_indices_to_robot_indices(config_parallel_mech_indexes)
        
        print(f"Teleoperation loaded parallel mechanism indices from config: {config_parallel_mech_indexes}")
        print(f"Mapped to 29-joint robot indices: {parallel_mech_indexes_teleop}")
        
    except Exception as e:
        print(f"Warning: Could not load parallel mechanism indices from config: {e}")
        # Fallback to hardcoded values
        parallel_mech_indexes_teleop = [21, 22, 27, 28]  # Ankle joints mapped to 29-joint system
        print(f"Using fallback parallel mechanism indices: {parallel_mech_indexes_teleop}")
    
    # Apply arm stiffness factor to arm joints only (joints 2-8 and 9-15)
    kps = base_kps.copy()
    kds = base_kds.copy()
    
    # Apply arm stiffness factor to left arm (joints 2-8)
    for i in range(2, 9):
        kps[i] *= arm_stiffness_factor
    
    # Apply arm stiffness factor to right arm (joints 9-15)  
    for i in range(9, 16):
        kps[i] *= arm_stiffness_factor
    
    
    start_time = time.time()
    command_count = 0
    
    try:
        while teleop_active:
            loop_start_time = time.time()
            
            # Check if combined mode is active
            is_combined_mode = walking_active and teleop_active
            
            # Wait for initial state
            if not state_received:
                time.sleep(0.001)
                continue
            
            # Initialize all motor commands with current positions
            for i in range(29):
                low_cmd.motor_cmd[i].q = current_positions[i]
                low_cmd.motor_cmd[i].dq = 0.0
                low_cmd.motor_cmd[i].kp = kps[i]
                low_cmd.motor_cmd[i].kd = kds[i]
                low_cmd.motor_cmd[i].tau = 0.0
            
            # Apply joint commands from WebSocket using organized data with smoothing
            # UPPER BODY CONTROL (joints 0-16): Always controlled by teleop
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
            
            # LOWER BODY CONTROL (joints 17-28): Depends on mode
            if is_combined_mode:
                # COMBINED MODE: Use walking targets for legs
                with combined_targets_lock:
                    walking_leg_targets = latest_walking_targets[17:29].copy()
                
                # Apply walking leg targets to lower body joints (17-28)
                for i in range(17, 29):
                    if i - 17 < len(walking_leg_targets):
                        low_cmd.motor_cmd[i].q = walking_leg_targets[i - 17]
                        
            else:
                # TELEOP ONLY MODE: Keep lower body at original positions (joints 17-28)
                for i in range(17, 29):
                    if i - 17 < len(lower_body_positions):
                        low_cmd.motor_cmd[i].q = lower_body_positions[i - 17]
            
            # Apply joint filtering for teleop targets
            for i in range(29):
                filtered_dof_target_teleop[i] = filtered_dof_target_teleop[i] * 0.8 + low_cmd.motor_cmd[i].q * 0.2
                low_cmd.motor_cmd[i].q = filtered_dof_target_teleop[i]
            
            # Apply torque control for parallel mechanism indices
            for i in parallel_mech_indexes_teleop:
                if i < 29:  # Ensure index is valid for 29-joint robot
                    low_cmd.motor_cmd[i].q = robot_dof_pos_latest[i]
                    
                    # Calculate torque
                    low_cmd.motor_cmd[i].tau = np.clip(
                        (filtered_dof_target_teleop[i] - robot_dof_pos_latest[i]) * kps[i],
                        -torque_limits_teleop[i],
                        torque_limits_teleop[i],
                    )
                    low_cmd.motor_cmd[i].kp = 0.0
                    low_cmd.motor_cmd[i].kd = 0.0
            
            # Store current teleop targets for coordination
            with combined_targets_lock:
                for i in range(29):
                    latest_teleop_targets[i] = low_cmd.motor_cmd[i].q
            
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
                
                mode_str = "🤝 COMBINED" if is_combined_mode else "🎮 TELEOP ONLY"
                print(f"📊 {mode_str} Teleoperation stats: {actual_hz:.1f}Hz, {command_count} commands sent")
                if is_combined_mode:
                    print(f"   Control split: Upper body (0-16) from teleop, legs (17-28) from walking")
                else:
                    print(f"   Control: Upper body from teleop, legs held at default positions")
                print(f"   Latest joint data keys: {list(latest_joint_data.keys()) if latest_joint_data else 'None'}")
                print(f"   Latest hand data: L={len(latest_hand_data.get('left_hand', {}))}, R={len(latest_hand_data.get('right_hand', {}))}")
                print(f"   Arm stiffness factor: {arm_stiffness_factor}")
                print(f"   Joint smoothing factor: {joint_smoothing_factor}")
                print(f"   Filtering: 0.8 smoothing factor, torque control for {len(parallel_mech_indexes_teleop)} parallel joints")
                print(f"   Filtered joints: {len(filtered_joint_positions)}")
                print(f"   Finger timing: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) + inversion + light smoothing + reasonable thresholds")
                print(f"   Finger settings: inverted commands, light smoothing: {finger_smoothing_factor}, threshold: {finger_command_threshold}")
                print(f"   Per-finger timing: min {finger_min_interval:.1f}s between commands to same finger")
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
        state_subscriber_teleop.CloseChannel()  # Clean up the state subscriber
        
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
        print(f"   Joint filtering: 0.8 smoothing factor applied to all joints")
        print(f"   Torque control: Applied to {len(parallel_mech_indexes_teleop)} parallel mechanism joints")
        print(f"   Total filtered joints: {len(filtered_joint_positions)}")
        print(f"   Finger timing used: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) + inversion + light smoothing + reasonable thresholds")
        print(f"   Finger settings: inverted commands, light smoothing: {finger_smoothing_factor}, threshold: {finger_command_threshold}")
        print(f"   Per-finger timing: min {finger_min_interval:.1f}s between commands to same finger")
        
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
        
        # Clear finger tracking for next session
        last_finger_positions.clear()
        smoothed_finger_positions.clear()
        finger_last_command_time.clear()
        
        # Reset timing
        last_finger_command_time = 0.0

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
    print("   Joint filtering: 0.8 smoothing factor applied to all joints")
    print("   Torque control: Applied to parallel mechanism joints (ankles)")
    print("   Latency tracking: Enabled (requires timestamps in messages)")
    print(f"   Finger control: {finger_command_interval:.3f}s ({1/finger_command_interval:.0f}Hz) with balanced approach")

    print("🤚 TELEOPERATION FEATURES:")
    print("  • Finger inversion: Input commands (0-1000) inverted to (1000-0) for robot")
    print("  • Light smoothing to filter out noise while staying responsive")
    print("  • Reasonable thresholds to allow normal movements")
    print("  • Per-finger timing to prevent command spam")
    print("  • Joint filtering: 0.8 smoothing factor applied to all joint commands")
    print("  • Torque control: Applied to parallel mechanism joints (ankles)")
    print()

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

# ===== WALKING CONTROL FUNCTIONS =====
def walking_control_loop(client: B1LocoClient, cfg_file: str):
    """Main walking control loop using RL policy"""
    global walking_active, walking_controller, current_positions
    global combined_mode_active, latest_walking_targets, combined_targets_lock
    
    # Initialize walking controller
    walking_controller = WalkingController(cfg_file)
    walking_controller.set_default_positions(current_positions)
    
    # Create low-level command publisher and subscriber
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    
    # State tracking variables
    robot_dof_pos = np.zeros(29, dtype=np.float32)
    robot_dof_vel = np.zeros(29, dtype=np.float32)
    robot_dof_pos_latest = np.zeros(29, dtype=np.float32)  # Latest positions for torque control
    base_ang_vel = np.zeros(3, dtype=np.float32)
    projected_gravity = np.zeros(3, dtype=np.float32)
    state_received = False
    
    # Joint filtering variables
    filtered_dof_target = np.zeros(29, dtype=np.float32)
    dof_target = np.zeros(29, dtype=np.float32)
    
    def state_handler(low_state_msg):
        nonlocal robot_dof_pos, robot_dof_vel, robot_dof_pos_latest, base_ang_vel, projected_gravity, state_received
        
        # Check for large IMU values (robot falling)
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            print(f"⚠️ WARNING: Large IMU values detected: {low_state_msg.imu_state.rpy}")
        
        # Update projected gravity
        projected_gravity[:] = rotate_vector_inverse_rpy(
            low_state_msg.imu_state.rpy[0],
            low_state_msg.imu_state.rpy[1], 
            low_state_msg.imu_state.rpy[2],
            np.array([0.0, 0.0, -1.0])
        )
        
        # Update base angular velocity
        base_ang_vel[:] = low_state_msg.imu_state.gyro
        
        # Update joint positions and velocities
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            if i < 29:
                robot_dof_pos[i] = motor.q
                robot_dof_vel[i] = motor.dq
                robot_dof_pos_latest[i] = motor.q  # For torque control
        
        state_received = True
    
    # Create state subscriber
    state_subscriber = B1LowStateSubscriber(state_handler)
    state_subscriber.InitChannel()
    
    # Create and initialize the command (only used when teleop is not active)
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    motor_cmds = [MotorCmd() for _ in range(29)]
    low_cmd.motor_cmd = motor_cmds
    
    # Load gains from config
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Extract gains (23-joint config -> 29-joint robot)
        config_kps = cfg["common"]["stiffness"] 
        config_kds = cfg["common"]["damping"]
        config_torque_limits = cfg["common"]["torque_limit"]
        config_parallel_mech_indexes = cfg["mech"]["parallel_mech_indexes"]
        
        # Map parallel mechanism indices from 23-joint to 29-joint space
        parallel_mech_indexes = map_policy_indices_to_robot_indices(config_parallel_mech_indexes)
        
        # Map gains from 23-joint config to 29-joint robot
        kps = np.zeros(29, dtype=np.float32)
        kds = np.zeros(29, dtype=np.float32)
        torque_limits = np.zeros(29, dtype=np.float32)
        
        # Head gains (0-1): direct mapping
        kps[0:2] = config_kps[0:2] 
        kds[0:2] = config_kds[0:2]
        torque_limits[0:2] = config_torque_limits[0:2]
        
        # Arm gains: use first 4 arm joint gains for all 7 arm joints  
        # Left arm (2-8): repeat config left arm gains (2-5)
        for i in range(7):
            src_idx = 2 + (i % 4)  # Cycle through config indices 2-5
            kps[2 + i] = config_kps[src_idx]
            kds[2 + i] = config_kds[src_idx] 
            torque_limits[2 + i] = config_torque_limits[src_idx]
        # Right arm (9-15): repeat config right arm gains (6-9)
        for i in range(7):
            src_idx = 6 + (i % 4)  # Cycle through config indices 6-9
            kps[9 + i] = config_kps[src_idx]
            kds[9 + i] = config_kds[src_idx]
            torque_limits[9 + i] = config_torque_limits[src_idx]
        
        # Waist gain (16): from config waist (10)
        kps[16] = config_kps[10]
        kds[16] = config_kds[10]
        torque_limits[16] = config_torque_limits[10]
        
        # Leg gains (17-28): from config legs (11-22)
        kps[17:29] = config_kps[11:23]
        kds[17:29] = config_kds[11:23]
        torque_limits[17:29] = config_torque_limits[11:23]
        
        print(f"Loaded parallel mechanism indices from config: {config_parallel_mech_indexes}")
        print(f"Mapped to 29-joint robot indices: {parallel_mech_indexes}")
        
    except Exception as e:
        raise e
    
    # Initialize filtered targets
    filtered_dof_target[:] = current_positions
    
    print("🚶 Walking control loop started")
    print(f"   Policy mode: Upper body masked, legs controlled by RL")
    print(f"   Commands: Use w/a/s/d/q/e for movement, 'stop' to halt")
    print(f"   Joint smoothing and torque control enabled")
    
    start_time = time.time()
    command_count = 0
    
    try:
        while walking_active:
            loop_start_time = time.time()
            
            # Check if combined mode is active
            combined_mode_active = teleop_active and walking_active
            
            # Wait for initial state
            if not state_received:
                time.sleep(0.001)
                continue
            
            # Update walking controller with latest robot state
            walking_controller.update_robot_state(
                robot_dof_pos, robot_dof_vel, base_ang_vel, projected_gravity
            )
            
            # Get policy targets (29 joints with upper body masked)
            target_positions = walking_controller.run_inference()
            
            # Update target positions
            dof_target[:] = target_positions
            
            if combined_mode_active:
                # COMBINED MODE: Store walking targets for teleoperation to use
                with combined_targets_lock:
                    latest_walking_targets[:] = target_positions
                
                # Don't send commands directly - teleoperation will handle it
                command_count += 1
                
            else:
                # WALKING ONLY MODE: Send commands directly with filtering and torque control
                
                # Apply filtering to all joints (like original deploy.py)
                for i in range(29):
                    filtered_dof_target[i] = filtered_dof_target[i] * 0.8 + dof_target[i] * 0.2
                
                # Set position targets for all joints
                for i in range(29):
                    low_cmd.motor_cmd[i].q = filtered_dof_target[i]
                    low_cmd.motor_cmd[i].dq = 0.0
                    low_cmd.motor_cmd[i].kp = kps[i]
                    low_cmd.motor_cmd[i].kd = kds[i]
                    low_cmd.motor_cmd[i].tau = 0.0
                
                # Use series-parallel conversion for torque to avoid non-linearity
                for i in parallel_mech_indexes:
                    if i < 29:  # Ensure index is valid for 29-joint robot
                        low_cmd.motor_cmd[i].q = robot_dof_pos_latest[i]
                        
                        # Calculate torque
                        low_cmd.motor_cmd[i].tau = np.clip(
                            (filtered_dof_target[i] - robot_dof_pos_latest[i]) * kps[i],
                            -torque_limits[i],
                            torque_limits[i],
                        )
                        low_cmd.motor_cmd[i].kp = 0.0
                        low_cmd.motor_cmd[i].kd = 0.0
                
                # Send commands
                low_cmd_publisher.Write(low_cmd)
                command_count += 1
            
            # Print stats every 5 seconds
            if command_count % 2500 == 0:  # Every 5 seconds at 500Hz
                elapsed_time = time.time() - start_time
                actual_hz = command_count / elapsed_time
                mode_str = "🤝 COMBINED" if combined_mode_active else "🚶 WALKING ONLY"
                print(f"{mode_str} Walking stats: {actual_hz:.1f}Hz, {command_count} commands processed")
                print(f"   Commands: vx={walking_controller.vx_cmd:.2f}, vy={walking_controller.vy_cmd:.2f}, vyaw={walking_controller.vyaw_cmd:.2f}")
                if combined_mode_active:
                    print(f"   Mode: Teleop controls upper body (0-16), walking controls legs (17-28)")
                else:
                    print(f"   Mode: Walking controls all joints, upper body masked to defaults")
                    print(f"   Filtering: 0.8 smoothing factor, torque control for {len(parallel_mech_indexes)} parallel joints")
            
            # Maintain ~500Hz loop rate for walking control
            loop_time = time.time() - loop_start_time
            target_loop_time = 0.002  # 500Hz = 2ms
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Walking control interrupted by user")
    finally:
        print("🔌 Cleaning up walking control resources")
        
        state_subscriber.CloseChannel()
        low_cmd_publisher.CloseChannel()
        
        # Clear combined mode
        combined_mode_active = False
        
        # Final statistics
        elapsed_time = time.time() - start_time
        actual_hz = command_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"📊 Final walking stats:")
        print(f"   Duration: {elapsed_time:.1f}s")
        print(f"   Commands processed: {command_count}")
        print(f"   Average frequency: {actual_hz:.1f}Hz")
        print(f"   Final commands: vx={walking_controller.vx_cmd:.2f}, vy={walking_controller.vy_cmd:.2f}, vyaw={walking_controller.vyaw_cmd:.2f}")

def start_walking_mode(client: B1LocoClient, cfg_file: str):
    """Start walking mode with RL policy"""
    global walking_active, walking_thread
    
    if walking_active:
        print("❌ Walking mode is already active")
        return False
    
    walking_active = True
    
    print("🚶 Starting walking mode...")
    print(f"   Config file: {cfg_file}")
    print("   Upper body: Masked to default positions (or controlled by teleop if active)")
    print("   Lower body: Controlled by RL policy")
    print("   Control frequency: ~500Hz")
    
    if teleop_active:
        print("   🤝 COMBINED MODE: Teleop controls upper body, walking controls legs")
    
    print()
    print("🎮 WALKING CONTROLS:")
    print("   w - Walk forward")
    print("   s - Walk backward")
    print("   a - Strafe left")
    print("   d - Strafe right")
    print("   q - Turn left")
    print("   e - Turn right")
    print("   stop - Stop all movement")
    print("   stop_walking - Exit walking mode")
    
    # Start walking control loop in a separate thread
    walking_thread = threading.Thread(
        target=walking_control_loop,
        args=(client, cfg_file),
        daemon=True
    )
    walking_thread.start()
    
    return True

def stop_walking_mode():
    """Stop walking mode"""
    global walking_active, walking_controller
    
    if not walking_active:
        print("❌ Walking mode is not active")
        return False
    
    print("🛑 Stopping walking mode...")
    walking_active = False
    
    # Clear walking commands
    if walking_controller:
        walking_controller.set_walking_commands(0.0, 0.0, 0.0)
    
    # Give some time for thread to clean up
    time.sleep(1.0)
    
    print("✅ Walking mode stopped successfully")
    return True

def set_walking_command(command: str):
    """Set walking velocity commands"""
    global walking_controller
    
    if not walking_active or not walking_controller:
        print("❌ Walking mode is not active")
        return False
    
    # Walking velocity parameters
    linear_speed = 0.5   # m/s for forward/backward/strafe
    angular_speed = 0.3  # rad/s for turning
    
    if command == "w":
        walking_controller.set_walking_commands(vx=linear_speed, vy=0.0, vyaw=0.0)
    elif command == "s": 
        walking_controller.set_walking_commands(vx=-linear_speed*0.5, vy=0.0, vyaw=0.0)  # Slower backward
    elif command == "a":
        walking_controller.set_walking_commands(vx=0.0, vy=linear_speed, vyaw=0.0)
    elif command == "d":
        walking_controller.set_walking_commands(vx=0.0, vy=-linear_speed, vyaw=0.0)
    elif command == "q":
        walking_controller.set_walking_commands(vx=0.0, vy=0.0, vyaw=angular_speed)
    elif command == "e":
        walking_controller.set_walking_commands(vx=0.0, vy=0.0, vyaw=-angular_speed)
    elif command == "stop":
        walking_controller.set_walking_commands(vx=0.0, vy=0.0, vyaw=0.0)
    else:
        print(f"❌ Unknown walking command: {command}")
        return False
    
    return True

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface [--ws-host HOST] [--ws-port PORT] [--config CONFIG]")
        sys.exit(-1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot control with teleoperation')
    parser.add_argument('network_interface', help='Network interface for robot communication')
    parser.add_argument('--ws-host', type=str, default='localhost',
                       help='WebSocket server host for teleoperation (default: localhost)')
    parser.add_argument('--ws-port', type=int, default=8765,
                       help='WebSocket server port for teleoperation (default: 8765)')
    parser.add_argument('--config', type=str, default='T1.yaml',
                       help='Configuration file for walking mode (default: T1.yaml)')
    
    args = parser.parse_args()
    
    # Initialize channel with network interface
    ChannelFactory.Instance().Init(0, args.network_interface)

    client = B1LocoClient()
    client.Init()
    x, y, z, yaw, pitch = 0.0, 0.0, 0.0, 0.0, 0.0
    res = 0
    hand_action_count = 0

    print("=" * 80)
    print("ROBOT CONTROL COMMANDS:")
    print("=" * 80)
    print("Setup Commands:")
    print("  mc              - Prepare robot and switch to Custom mode")
    print()
    print("Walking Mode Commands:")
    print("  walk            - Start RL-based walking mode")
    print("  stop_walking    - Stop walking mode")
    print()
    print("Walking Controls (when in walking mode):")
    print("  w/a/s/d/q/e     - Move robot (forward/left/back/right/turn_left/turn_right)")
    print("  stop            - Stop movement")
    print()
    print("Teleoperation Commands:")
    print(f"  teleop          - Start teleoperation mode (connects to {args.ws_host}:{args.ws_port})")
    print("  stop_teleop     - Stop teleoperation mode")
    print()
    print("Legacy Movement Commands (in Walking mode, for testing):")
    print("  w/a/s/d/q/e     - Move robot")
    print("  stop            - Stop movement")
    print()
    print("Head Commands:")
    print("  hd/hu/hr/hl     - Move head down/up/right/left")
    print("  ho              - Center head")
    print()
    print("Hand Commands:")
    print("  hand            - Hand oscillation")
    print("  hand-open       - Open hands")  
    print("  hand-close      - Close hands")
    print()
    print("RECOMMENDED WORKFLOW:")
    print("  1. First run 'mc' to prepare robot and switch to custom mode")
    print("  2a. FOR WALKING: Run 'walk' to start RL walking mode, then use w/a/s/d/q/e")
    print("  2b. FOR TELEOP: Run 'teleop' to start receiving WebSocket commands")
    print(f"  3. For teleop: Make sure WebSocket server is running on {args.ws_host}:{args.ws_port}")
    print("  4. Walking mode masks upper body to default positions")
    print("  5. Teleoperation allows full upper body + finger control")
    print()
    print("🤚 TELEOPERATION FEATURES:")
    print("  • Finger inversion: Input commands (0-1000) inverted to (1000-0) for robot")
    print("  • Light smoothing to filter out noise while staying responsive")
    print("  • Reasonable thresholds to allow normal movements")
    print("  • Per-finger timing to prevent command spam")
    print("  • Joint filtering: 0.8 smoothing factor applied to all joint commands")
    print("  • Torque control: Applied to parallel mechanism joints (ankles)")
    print()
    print("🤝 NEW: COMBINED WALKING + TELEOPERATION MODE:")
    print("  • Both 'walk' and 'teleop' can run simultaneously!")
    print("  • Teleoperation controls upper body (head, arms, waist, fingers)")
    print("  • Walking RL policy controls legs for stable locomotion")
    print("  • Upper body commands come from WebSocket (VR/tracking)")
    print("  • Leg commands come from RL policy with w/a/s/d/q/e movement")
    print("  • Best of both worlds: expressive upper body + stable walking")
    print()
    print("COMBINED MODE WORKFLOW:")
    print("  1. First run 'mc' to prepare robot")
    print("  2. Run 'walk' to start walking mode")
    print("  3. Run 'teleop' to start teleoperation mode")
    print("  4. Now you have combined control!")
    print("  5. Use w/a/s/d/q/e for walking + WebSocket for upper body")
    print("  6. Both systems coordinate automatically")
    print()
    print("🤖 WALKING MODE FEATURES:")
    print("  • RL policy controls legs for stable walking")
    print("  • Upper body (head, arms, waist) masked to default positions")
    print("  • Policy maps 29-joint robot to 23-joint policy space")
    print("  • Real-time velocity commands with w/a/s/d/q/e")
    print("  • ~500Hz control frequency for smooth motion")
    print("  • Joint filtering: 0.8 smoothing factor for all commands")
    print("  • Torque control: Applied to parallel mechanism joints from config")
    print("=" * 80)

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
                elif input_cmd == 'walk':
                    print(f"Starting walking mode (config: {args.config})...")
                    success = start_walking_mode(client, args.config)
                    if not success:
                        print("Failed to start walking mode")
                elif input_cmd == 'stop':
                    print("Stopping walking mode...")
                    success = stop_walking_mode()
                    if not success:
                        print("Failed to stop walking mode")
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
                elif input_cmd in ["w", "a", "s", "d", "q", "e"]:
                    # Handle walking commands if in walking mode
                    if walking_active:
                        success = set_walking_command(input_cmd)
                        if not success:
                            print("Failed to set walking command")
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
        if walking_active:
            stop_walking_mode()
        if teleop_active:
            stop_teleoperation()

if __name__ == "__main__":
    main()
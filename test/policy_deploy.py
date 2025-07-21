import sys
import time
import yaml
import argparse
import numpy as np
import onnxruntime
import threading
from termcolor import colored
from booster_robotics_sdk_python import (
    B1LocoClient, 
    ChannelFactory, 
    RobotMode, 
    B1LowCmdPublisher, 
    B1LowStateSubscriber, 
    LowCmd, 
    LowCmdType, 
    MotorCmd
)

def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the inverse of the given roll, pitch, and yaw angles.
    Copied from deploy/utils/rotate.py for proper projected gravity calculation.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x).T @ vector

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return a - b + c

def quat_rotate_inverse_numpy(q, v):
    """Alias for quat_rotate_inverse for backward compatibility"""
    return quat_rotate_inverse(q, v)

# Import the utility functions to match reference files
sys.path.append("../")
sys.path.append("./")

def rpy_to_quat(rpy):
    """Convert roll, pitch, yaw to quaternion [w, x, y, z] - matching state processor"""
    roll, pitch, yaw = rpy
    # Convert to half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    # Quaternion multiplication
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

class PolicyDeploy:
    """
    Simplified policy deployment for sim2real using the Booster T1 robot.
    Supports custom mode initialization, policy execution, and basic movement commands.
    Includes safety features from deploy.py and proper observation structure from loco_manip.py.
    """
    
    def __init__(self, config_path, model_path):
        # Load configuration
        try:
            with open(config_path) as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
        
        # Validate required config keys
        required_config_keys = [
            "NUM_JOINTS", "DEFAULT_DOF_ANGLES", "DOMAIN_ID", "NET", 
            "DESIRED_BASE_HEIGHT", "MOTOR_KP", "MOTOR_KD",
            "motor_pos_lower_limit_list", "motor_pos_upper_limit_list",
            "prepare"
        ]
        for key in required_config_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.model_path = model_path
        
        # Initialize robot parameters
        self.num_dofs = self.config["NUM_JOINTS"]
        self.num_upper_dofs = self.config.get("NUM_UPPER_BODY_JOINTS", 16)  # Use config value
        self.num_lower_dofs = self.num_dofs - self.num_upper_dofs
        self.default_dof_angles = np.array(self.config["DEFAULT_DOF_ANGLES"])
        
        # Residual upper body action configuration
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)
        self.upper_dof_indices = np.arange(self.num_upper_dofs)  # First 16 joints are upper body
        
        # Use stand keyframe positions as upper body goals (with elbow angles 1.4/-1.4 instead of 2.2/-2.2)
        stand_upper_body_pos = np.array([
            0.004, 0.009,  # Head
            0.467, -1.094, 0.024, 1.4, -0.002, 0.027, -0.009,  # Left arm 
            0.472, 1.091, 0.021, -1.4, -0.001, -0.014, 0.005   # Right arm
        ])
        self.ref_upper_dof_pos = stand_upper_body_pos.reshape(1, -1)
        
        # Initialize SDK
        ChannelFactory.Instance().Init(self.config["DOMAIN_ID"], self.config["NET"])
        
        # Initialize client and publishers
        self.client = B1LocoClient()
        self.client.Init()
        
        # Initialize policy
        self.setup_policy()
        
        # Initialize control variables
        self.custom_mode_active = False
        self.policy_active = False
        self.current_positions = np.zeros(self.num_dofs)
        self.last_policy_action = np.zeros((1, self.num_dofs))  # Match base_policy.py structure
        
        # Initialize state arrays matching booster_state_processor.py
        # 3 + 4 + num_dof (base_pos + base_quat + joint_pos)
        self._init_q = np.zeros(3 + 4 + self.num_dofs)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dofs)  # base_lin_vel + base_ang_vel + joint_vel
        self.ddq = np.zeros(3 + 3 + self.num_dofs)  # base_lin_acc + base_ang_acc + joint_acc
        self.tau_est = np.zeros(3 + 3 + self.num_dofs)  # base_lin_force + base_ang_torque + joint_torque
        
        # Safety variables
        self.robot_fallen = False
        self.imu_safety_threshold = 1.0  # Same as deploy.py
        
        # Command variables
        self.lin_vel_command = np.array([0.0, 0.0])  # [forward/backward, left/right]
        self.ang_vel_command = np.array([0.0])       # [yaw rotation]
        self.base_height_command = np.array([self.config["DESIRED_BASE_HEIGHT"]])
        self.stand_command = np.array([0.0])  # 1.0 = walking mode, 0.0 = stance mode (start in stance)
        
        # KP/KD scaling factors (can be modified during runtime)
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        
        # Get KP/KD values from config
        self.base_kps = np.array(self.config["MOTOR_KP"])
        self.base_kds = np.array(self.config["MOTOR_KD"])
        
        # Policy parameters
        self.policy_action_scale = 0.25
        self.gait_period = 0.5
        self.phase = 0.0
        
        # Observation buffers for policy
        self.obs_scales = self.config["obs_scales"]
        self.obs_dims = self.config["obs_dims"]
        self.obs_dict = self.config["obs_dict"]
        self.history_length_dict = self.config["history_length_dict"]
        self._init_obs_buffers()
        
        print("PolicyDeploy initialized successfully!")
        
    def setup_policy(self):
        """Setup ONNX policy model."""
        print(f"Loading policy from: {self.model_path}")
        
        try:
            self.onnx_policy_session = onnxruntime.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {self.model_path}: {e}")
        
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        
        if not input_names:
            raise RuntimeError(f"ONNX model has no inputs: {self.model_path}")
        if not output_names:
            raise RuntimeError(f"ONNX model has no outputs: {self.model_path}")
        
        self.onnx_input_names = input_names
        self.onnx_output_names = output_names
        
        print(f"Policy loaded successfully. Inputs: {input_names}, Outputs: {output_names}")
        
    def _init_obs_buffers(self):
        """Initialize observation buffers for policy inference."""
        # Validate required configuration
        required_keys = ['obs_scales', 'obs_dims', 'obs_dict', 'history_length_dict']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Ensure actor_obs is present in obs_dict
        if 'actor_obs' not in self.obs_dict:
            raise ValueError("Missing 'actor_obs' in obs_dict configuration")
        
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        
        # Initialize observation buffers
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
    
    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type - matching base_policy.py."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                if obs_name not in self.obs_dims:
                    raise ValueError(f"Missing dimension for observation '{obs_name}' in obs_dims")
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict
        
    def switch_to_custom_mode(self):
        """Initialize robot and switch to custom mode with smooth transition to default angles."""
        print("🔧 Preparing robot for custom mode...")
        
        # Read current robot state
        print("📖 Reading current robot positions...")
        positions_received = False
        
        def state_handler(low_state_msg):
            nonlocal positions_received
            # Safety check - same as deploy.py
            if abs(low_state_msg.imu_state.rpy[0]) > self.imu_safety_threshold or abs(low_state_msg.imu_state.rpy[1]) > self.imu_safety_threshold:
                print(f"⚠️  Warning: IMU base rpy values are too large: {low_state_msg.imu_state.rpy}")
                self.robot_fallen = True
                return
            
            if len(low_state_msg.motor_state_serial) >= self.num_dofs:
                for i in range(self.num_dofs):
                    self.current_positions[i] = low_state_msg.motor_state_serial[i].q
                positions_received = True
        
        # Subscribe to get current state
        state_subscriber = B1LowStateSubscriber(state_handler)
        state_subscriber.InitChannel()
        
        # Wait for state data
        timeout = 5.0
        start_time = time.time()
        while not positions_received and (time.time() - start_time) < timeout:
            if self.robot_fallen:
                print("❌ Robot appears to have fallen. Please check robot orientation.")
                state_subscriber.CloseChannel()
                return False
            time.sleep(0.1)
        
        state_subscriber.CloseChannel()
        
        if not positions_received:
            print("⚠️  Warning: Could not read current positions, using defaults")
            self.current_positions = self.default_dof_angles.copy()
        else:
            print("✅ Successfully read current robot positions!")
        
        # Create low-level command publisher
        low_cmd_publisher = B1LowCmdPublisher()
        low_cmd_publisher.InitChannel()
        
        print("🔄 Switching to custom mode...")
        res = self.client.ChangeMode(RobotMode.kCustom)
        if res != 0:
            print(f"❌ Failed to switch to custom mode: error = {res}")
            low_cmd_publisher.CloseChannel()
            return False
        
        print("✅ Successfully switched to custom mode!")
        
        # Smooth transition to default positions
        print("🎯 Transitioning to default joint positions...")
        transition_duration = 10.0  # seconds
        control_freq = 50  # Hz
        control_dt = 1.0 / control_freq
        num_steps = int(transition_duration / control_dt)
        
        # Get prepare gains - use higher stiffness for better position tracking during transition
        prepare_kps = np.array(self.config["prepare"]["stiffness"]) * 2.0  # Double the stiffness for transition
        prepare_kds = self.config["prepare"]["damping"]
        
        start_positions = self.current_positions.copy()
        target_positions = self.default_dof_angles.copy()
        
        try:
            for step in range(num_steps + 1):
                step_start_time = time.time()
                
                # Calculate interpolation factor (0 to 1)
                alpha = min(1.0, step / num_steps)
                
                # Linear interpolation between start and target positions
                current_targets = start_positions + alpha * (target_positions - start_positions)
                
                # Create and send command
                low_cmd = LowCmd()
                low_cmd.cmd_type = LowCmdType.SERIAL
                motor_cmds = [MotorCmd() for _ in range(self.num_dofs)]
                low_cmd.motor_cmd = motor_cmds
                
                # Set interpolated target positions
                for i in range(self.num_dofs):
                    low_cmd.motor_cmd[i].q = current_targets[i]
                    low_cmd.motor_cmd[i].dq = 0.0
                    low_cmd.motor_cmd[i].kp = prepare_kps[i]
                    low_cmd.motor_cmd[i].kd = prepare_kds[i]
                    low_cmd.motor_cmd[i].tau = 0.0
                
                # Send command
                low_cmd_publisher.Write(low_cmd)
                
                # Print progress occasionally
                if step % (num_steps // 10) == 0 or step == num_steps:
                    progress = int(alpha * 100)
                    print(f"   Transition progress: {progress}%")
                
                # Maintain control frequency
                elapsed = time.time() - step_start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n⚠️  Transition interrupted by user")
        except Exception as e:
            print(f"❌ Error during transition: {e}")
            low_cmd_publisher.CloseChannel()
            return False
        
        print("✅ Transition to default positions completed!")
        
        # Update current positions to default angles
        self.current_positions = self.default_dof_angles.copy()
        
        # Clean up
        low_cmd_publisher.CloseChannel()
        
        self.custom_mode_active = True
        return True
    
    def policy_inference(self, robot_state_data):
        """Perform policy inference to get policy actions for all joints."""
        obs = self.prepare_obs_for_policy(robot_state_data)
        
        # Run policy inference
        input_feed = {name: obs[name] for name in self.onnx_input_names}
        outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
        policy_action = outputs[0]
        
        # Ensure policy_action is 2D
        if policy_action.ndim == 1:
            policy_action = policy_action.reshape(1, -1)
        
        # Fail fast if dimensions are wrong
        if policy_action.shape[1] != self.num_dofs:
            raise ValueError(
                f"Policy output dimension mismatch: "
                f"Expected {self.num_dofs} DOFs, got {policy_action.shape[1]}. "
                f"Policy shape: {policy_action.shape}, "
                f"Expected shape: (1, {self.num_dofs})"
            )
        
        # Clip actions
        policy_action = np.clip(policy_action, -1, 1)
        
        # Store for observation buffer
        self.last_policy_action = policy_action.copy()
        
        # Scale actions
        scaled_policy_action = policy_action * self.policy_action_scale
        
        return scaled_policy_action
        
    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state - matching base_policy.py structure."""
        current_obs_buffer_dict = {}
        
        # Extract base and joint data (exactly like base_policy.py)
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3:7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7:7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[:, 7 + self.num_dofs + 6:7 + self.num_dofs + 6 + self.num_dofs]
        
        # Calculate projected gravity (matching base_policy.py)
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse_numpy(
            current_obs_buffer_dict["base_quat"], v
        )
        
        # Add command observations (these should be added here for loco_manip compatibility)
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command.reshape(1, -1)
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command.reshape(1, -1) 
        current_obs_buffer_dict["command_stand"] = self.stand_command.reshape(1, -1)
        current_obs_buffer_dict["command_base_height"] = self.base_height_command.reshape(1, -1)
        current_obs_buffer_dict["command_waist_dofs"] = np.zeros((1, 3))
        
        # Upper body reference positions
        ref_upper_dof_pos = self.default_dof_angles[:self.num_upper_dofs]
        current_obs_buffer_dict["ref_upper_dof_pos"] = ref_upper_dof_pos.reshape(1, -1)
        
        # Actions (matching loco_manip.py)
        if self.last_policy_action.ndim == 1:
            current_obs_buffer_dict["actions"] = self.last_policy_action.reshape(1, -1)
        else:
            current_obs_buffer_dict["actions"] = self.last_policy_action
        
        return current_obs_buffer_dict
    
    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary - matching base_policy.py."""
        current_obs_dict = {}
        for key in self.obs_dict:
            obs_list = sorted(self.obs_dict[key])
            current_obs_dict[key] = np.concatenate(
                [current_obs_buffer_dict[obs_name] * self.obs_scales[obs_name] for obs_name in obs_list], axis=1
            )
        return current_obs_dict
        
    def prepare_obs_for_policy(self, robot_state_data):
        """Prepare observations for policy inference - exactly matching base_policy.py structure."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        
        # Update observation buffers with history (exactly like base_policy.py)
        self.obs_buf_dict = {
            key: np.concatenate(
                (
                    self.obs_buf_dict[key][:, self.obs_dim_dict[key] : (self.obs_dim_dict[key] * self.history_length_dict[key])],
                    current_obs_dict[key],
                ),
                axis=1,
            )
            for key in self.obs_buf_dict
        }
        
        return {"actor_obs": self.obs_buf_dict["actor_obs"].astype(np.float32)}
    
    def _extract_imu_data(self, imu_state):
        """Extract IMU data from Booster state message - matching booster_state_processor.py"""
        # base quaternion
        self.q[0:3] = 0.0  # base position (assumed to be at origin)
        rpy = imu_state.rpy
        self.q[3:7] = rpy_to_quat(rpy)
        self.dq[3:6] = imu_state.gyro
        self.ddq[0:3] = imu_state.acc

    def _extract_joint_data(self, robot_joint_state):
        """Extract joint data from Booster state message - matching booster_state_processor.py"""
        for i in range(self.num_dofs):
            # For Booster SDK, motor index matches joint index directly
            self.q[7 + i] = robot_joint_state[i].q
            self.dq[6 + i] = robot_joint_state[i].dq
            self.tau_est[6 + i] = robot_joint_state[i].tau_est

    def _create_robot_state_data(self):
        """Create the final robot state data array - matching basic_state_processor.py"""
        robot_state_data = np.array(
            self.q.tolist() + self.dq.tolist() + self.tau_est.tolist() + self.ddq.tolist(), 
            dtype=np.float64
        ).reshape(1, -1)
        return robot_state_data
    
    def run_policy(self):
        """Main policy execution loop."""
        if not self.custom_mode_active:
            print("❌ Must be in custom mode first. Run 'mc' command.")
            return False
            
        print("🚀 Starting policy execution...")
        self.policy_active = True
        
        # Create state subscriber with proper IMU processing matching booster_state_processor.py
        self.robot_state_data = None
        state_received = False
        
        def state_handler(low_state_msg):
            nonlocal state_received
            
            # Safety check for falling - same as deploy.py
            if abs(low_state_msg.imu_state.rpy[0]) > self.imu_safety_threshold or abs(low_state_msg.imu_state.rpy[1]) > self.imu_safety_threshold:
                print(f"⚠️  Warning: IMU base rpy values are too large: {low_state_msg.imu_state.rpy}")
                self.robot_fallen = True
                self.policy_active = False
                return
            
            if len(low_state_msg.motor_state_serial) >= self.num_dofs:
                # Extract IMU data using the same method as booster_state_processor
                imu_state = low_state_msg.imu_state
                self._extract_imu_data(imu_state)
                
                # Extract joint data using serial motor state
                robot_joint_state = low_state_msg.motor_state_serial
                self._extract_joint_data(robot_joint_state)
                
                # Create robot state data array in the same format as the state processor
                self.robot_state_data = self._create_robot_state_data()
                state_received = True
        
        state_subscriber = B1LowStateSubscriber(state_handler)
        state_subscriber.InitChannel()
        
        # Create command publisher
        low_cmd_publisher = B1LowCmdPublisher()
        low_cmd_publisher.InitChannel()
        
        # Control loop
        rate_hz = 50  # 50 Hz control loop
        loop_time = 1.0 / rate_hz
        
        try:
            print(f"🔄 Policy running at {rate_hz}Hz. Press Ctrl+C to stop.")
            print("Commands: w/a/s/d (move), q/e (turn), space (stop), = (toggle stand/walk)")
            
            while self.policy_active:
                loop_start = time.time()
                
                # Check for safety conditions
                if self.robot_fallen:
                    raise RuntimeError("Robot has fallen! Stopping policy execution for safety.")
                
                if state_received and self.robot_state_data is not None:
                    try:
                        # Get policy actions for all joints
                        policy_actions = self.policy_inference(self.robot_state_data)
                        
                        # Calculate target joint positions
                        joint_targets = policy_actions.flatten() + self.default_dof_angles
                        
                        # Apply residual upper body action if enabled
                        if self.residual_upper_body_action:
                            joint_targets[self.upper_dof_indices] += (
                                self.ref_upper_dof_pos.flatten() - self.default_dof_angles[self.upper_dof_indices]
                            )
                        
                        # Create and send command
                        low_cmd = LowCmd()
                        low_cmd.cmd_type = LowCmdType.SERIAL
                        motor_cmds = [MotorCmd() for _ in range(self.num_dofs)]
                        low_cmd.motor_cmd = motor_cmds
                        
                        # Apply motor limits
                        motor_lower_limits = self.config["motor_pos_lower_limit_list"]
                        motor_upper_limits = self.config["motor_pos_upper_limit_list"]
                        joint_targets = np.clip(joint_targets, motor_lower_limits, motor_upper_limits)
                        
                        # Fill motor commands
                        for i in range(self.num_dofs):
                            low_cmd.motor_cmd[i].q = joint_targets[i]
                            low_cmd.motor_cmd[i].dq = 0.0
                            low_cmd.motor_cmd[i].kp = self.base_kps[i] * self.kp_scale
                            low_cmd.motor_cmd[i].kd = self.base_kds[i] * self.kd_scale
                            low_cmd.motor_cmd[i].tau = 0.0
                        
                        # breakpoint()
                        # Send command
                        low_cmd_publisher.Write(low_cmd)
                        
                    except Exception as e:
                        print(f"❌ Policy execution failed: {e}")
                        self.policy_active = False
                        raise
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < loop_time:
                    time.sleep(loop_time - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 Policy execution stopped by user")
        finally:
            self.policy_active = False
            state_subscriber.CloseChannel()
            low_cmd_publisher.CloseChannel()
            
        return True
    
    def handle_velocity_command(self, command):
        """Handle velocity commands."""
        vel_step = 0.1
        ang_step = 0.2
        
        if not self.stand_command[0]:  # Only allow movement in walk mode
            print("⚠️  Must be in walk mode. Press '=' to toggle.")
            return
            
        if command == 'w':      # Forward
            self.lin_vel_command[0] += vel_step
        elif command == 's':    # Backward  
            self.lin_vel_command[0] -= vel_step
        elif command == 'a':    # Left
            self.lin_vel_command[1] += vel_step
        elif command == 'd':    # Right
            self.lin_vel_command[1] -= vel_step
        elif command == 'q':    # Turn left
            self.ang_vel_command[0] += ang_step
        elif command == 'e':    # Turn right
            self.ang_vel_command[0] -= ang_step
        elif command == ' ':    # Stop
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0
            
        # Clip velocities to reasonable limits
        self.lin_vel_command = np.clip(self.lin_vel_command, -1.0, 1.0)
        self.ang_vel_command = np.clip(self.ang_vel_command, -1.0, 1.0)
        
        print(f"📍 Velocity: linear={self.lin_vel_command}, angular={self.ang_vel_command}")
    
    def toggle_stand_walk(self):
        """Toggle between stand and walk mode."""
        self.stand_command[0] = 1.0 - self.stand_command[0]
        if self.stand_command[0] == 0:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0
            print(colored("🚏 STANCE MODE - Robot will stand in place", "blue"))
        else:
            print(colored("🚶 WALK MODE - Robot ready to move", "green"))
    
    def adjust_gains(self, kp_change=0.0, kd_change=0.0):
        """Adjust KP/KD scaling factors."""
        self.kp_scale += kp_change
        self.kd_scale += kd_change
        
        # Keep gains positive and reasonable
        self.kp_scale = max(0.1, min(2.0, self.kp_scale))
        self.kd_scale = max(0.1, min(2.0, self.kd_scale))
        
        print(f"🔧 Gains: KP scale={self.kp_scale:.2f}, KD scale={self.kd_scale:.2f}")
    
    def print_status(self):
        """Print current system status."""
        print(f"\n📊 POLICY DEPLOY STATUS:")
        print(f"   Custom Mode: {'✅' if self.custom_mode_active else '❌'}")
        print(f"   Policy Running: {'✅' if self.policy_active else '❌'}")
        print(f"   Robot Fallen: {'❌' if self.robot_fallen else '✅'}")
        print(f"   Movement Mode: {'🚶 WALK' if self.stand_command[0] else '🚏 STANCE'}")
        print(f"   Linear Velocity: {self.lin_vel_command}")
        print(f"   Angular Velocity: {self.ang_vel_command}")
        print(f"   Base Height: {self.base_height_command[0]:.3f}m")
        print(f"   KP Scale: {self.kp_scale:.2f}, KD Scale: {self.kd_scale:.2f}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Policy deployment for Booster T1 robot')
    parser.add_argument('network_interface', help='Network interface for robot communication') 
    parser.add_argument('--config', type=str, default='ref/t1_29dof_falcon.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='ref/t1_29dof.onnx', help='ONNX model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 POLICY DEPLOY - Sim2Real for Booster T1")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Network: {args.network_interface}")
    print("=" * 60)
    
    # Create policy deploy instance
    try:
        policy_deploy = PolicyDeploy(args.config, args.model)
    except Exception as e:
        print(f"❌ Failed to initialize PolicyDeploy: {e}")
        return
    
    print("\n🎮 COMMANDS:")
    print("  mc          - Switch to custom Mode and prepare robot")
    print("  policy      - Start policy execution")  
    print("  stop        - Stop policy execution")
    print("  w/s/a/d     - Linear movement (forward/back/left/right)")
    print("  q/e         - Angular movement (turn left/right)")
    print("  space       - Stop movement (zero velocities)")
    print("  =           - Toggle stand/walk mode")
    print("  +/-         - Increase/decrease KP gains")
    print("  [/]         - Increase/decrease KD gains")
    print("  status      - Print current status")
    print("  help        - Show this help")
    print("  exit        - Exit program")
    print("=" * 60)
    
    # Start keyboard input thread
    def keyboard_handler():
        while True:
            try:
                cmd = input().strip().lower()
                if not cmd:
                    continue
                    
                if cmd == 'mc':
                    success = policy_deploy.switch_to_custom_mode()
                    if not success:
                        print("❌ Failed to switch to custom mode")
                        
                elif cmd == 'policy':
                    if not policy_deploy.policy_active:
                        # Start policy in separate thread
                        policy_thread = threading.Thread(target=policy_deploy.run_policy, daemon=True)
                        policy_thread.start()
                    else:
                        print("⚠️  Policy is already running")
                        
                elif cmd == 'stop':
                    policy_deploy.policy_active = False
                    print("🛑 Stopping policy execution...")
                    
                elif cmd in ['w', 's', 'a', 'd', 'q', 'e', ' ']:
                    policy_deploy.handle_velocity_command(cmd)
                    
                elif cmd == '=':
                    policy_deploy.toggle_stand_walk()
                    
                elif cmd == '+':
                    policy_deploy.adjust_gains(kp_change=0.1)
                elif cmd == '-':
                    policy_deploy.adjust_gains(kp_change=-0.1)
                elif cmd == '[':
                    policy_deploy.adjust_gains(kd_change=-0.1)
                elif cmd == ']':
                    policy_deploy.adjust_gains(kd_change=0.1)
                    
                elif cmd == 'status':
                    policy_deploy.print_status()
                    
                elif cmd == 'help':
                    print("\n🎮 COMMANDS:")
                    print("  mc          - Switch to custom Mode and prepare robot")
                    print("  policy      - Start policy execution")
                    print("  stop        - Stop policy execution")
                    print("  w/s/a/d     - Linear movement")
                    print("  q/e         - Angular movement") 
                    print("  space       - Stop movement")
                    print("  =           - Toggle stand/walk mode")
                    print("  +/-         - Adjust KP gains")
                    print("  [/]         - Adjust KD gains")
                    print("  status      - Print status")
                    print("  exit        - Exit program\n")
                    
                elif cmd == 'exit':
                    policy_deploy.policy_active = False
                    print("👋 Goodbye!")
                    break
                    
                else:
                    print(f"❓ Unknown command: {cmd}. Type 'help' for commands.")
                    
            except KeyboardInterrupt:
                policy_deploy.policy_active = False
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
    
    # Run keyboard handler
    keyboard_handler()

if __name__ == "__main__":
    main()
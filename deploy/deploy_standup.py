import numpy as np
import time
import yaml
import logging
import threading
from typing import Optional

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
from utils.standup_policy import StandupPolicy # Import StandupPolicy
import select # For non-blocking input
import sys # For stdin
import tty # For raw input mode
import termios # For terminal settings

# Global constants for upper body control modes
# Set one of these to the string value to enable that control mode
# Options: "policy", "teleop", "sine"
UPPER_BODY_CONTROL_MODE = "teleop"  # Default to policy control

# Control parameter for arm gains
# Lower value means smoother movements with less stiffness
# Range: 0.1 (very soft) to 1.0 (full stiffness)
ARM_STIFFNESS_FACTOR = 0.2 * 1.25 * 1.25 # * 1.25

# Standup success parameters
STANDUP_SUCCESS_DURATION_S = 10.0  # Seconds to wait in STADING_UP before checking stability
UPRIGHT_IMU_THRESHOLD_RP = 0.35 # Radians (approx 20 degrees) for roll/pitch to be considered upright
STANDUP_ATTEMPT_TIMEOUT_S = 15.0 # Max duration for a standup attempt

# Define robot states
class RobotState(Enum):
    WALKING = 0             # Normal operation, walking policy active
    FALLEN = 1              # Robot has fallen based on IMU data
    WAITING_FOR_STANDUP = 2 # Fallen, waited 5s, waiting for user input 's'
    STANDING_UP = 3         # Stand-up policy active

class BodyPart(Enum):
    LOWER_BODY = 0  # Legs and torso
    UPPER_BODY = 1  # Arms

class UpperBodyControlMode(Enum):
    POLICY = "policy"    # Upper body controlled by policy
    TELEOP = "teleop"    # Upper body controlled by VR/teleop
    SINE = "sine"        # Upper body controlled by sine wave

class Controller:
    def __init__(self, cfg_file) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        # Reduce stiffness for arm joints directly in the config
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
        self.standup_policy = StandupPolicy(cfg=self.cfg) # Initialize StandupPolicy

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
        self.state = RobotState.WALKING # Initial state
        self.fall_detected_time: Optional[float] = None # Time when fall was detected
        self.standup_start_time: Optional[float] = None # Time when standup started

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.current_imu_rpy = np.zeros(3, dtype=np.float32) # Store current IMU RPY values
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
        # Always update current IMU RPY
        with self.publish_lock: # Protect access if other threads might read it, though unlikely here
             self.current_imu_rpy[:] = low_state_msg.imu_state.rpy

        # Only process state changes if currently walking
        if self.state == RobotState.WALKING:
            # Check for fall condition (IMU roll/pitch > 1.0 rad)
            if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
                if self.state == RobotState.WALKING: # Only trigger fall detection once
                    self.logger.warning("FALL DETECTED! IMU base rpy values: {}".format(low_state_msg.imu_state.rpy))
                    self.state = RobotState.FALLEN
                    self.fall_detected_time = self.timer.get_time()
                    self.logger.info(f"State changed to {self.state} at time {self.fall_detected_time:.2f}. Waiting 5 seconds.")
                    # Optionally, stop the publisher thread or change its behavior immediately
                    # self.running = False # Decide if falling should stop the publisher thread entirely or just change policy

        # Continue processing low state data regardless of fall state for monitoring/logging
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
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

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
    
    def run(self):
        time_now = self.timer.get_time()
        
        # State machine logic
        if self.state == RobotState.FALLEN:
            if self.fall_detected_time is not None and (time_now - self.fall_detected_time >= 5.0):
                self.state = RobotState.WAITING_FOR_STANDUP
                self.logger.info(f"State changed to {self.state}. Press 's' to attempt stand-up.")
            # Do not run inference or advance timers while fallen or waiting
            time.sleep(0.01) # Prevent busy-waiting
            return
            
        if self.state == RobotState.WAITING_FOR_STANDUP:
            # Wait for external input (handled in main loop) to change state
            time.sleep(0.01) # Prevent busy-waiting
            return

        # Proceed with inference only if WALKING or STANDING_UP
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        
        self.logger.debug("-----------------------------------------------------")
        # Determine which policy interval to use
        current_policy_interval = self.policy.get_policy_interval() if self.state == RobotState.WALKING else self.standup_policy.get_policy_interval()
        self.next_inference_time += current_policy_interval
        self.logger.debug(f"State: {self.state} | Next inference time: {self.next_inference_time}")
        start_time = time.perf_counter()
        
        # Create copies of dof_pos and dof_vel for policy inference
        masked_dof_pos = np.copy(self.dof_pos)
        masked_dof_vel = np.copy(self.dof_vel)
        
        # If in sine mode, mask upper body positions and velocities with defaults
        # so the policy thinks the arms are in their default positions and not moving
        # This masking might need adjustment based on standup policy needs
        if self.body_part_control_mode[BodyPart.UPPER_BODY] == UpperBodyControlMode.SINE.value:
            # Get default positions from policy
            default_positions = self.policy.default_dof_pos
            
            # Replace upper body positions with default positions
            for i in self.upper_body_indices:
                masked_dof_pos[i] = default_positions[i]
                masked_dof_vel[i] = 0.0  # Zero velocity

        # Select the appropriate policy based on the current state
        if self.state == RobotState.WALKING:
            active_policy = self.policy
            policy_targets = active_policy.inference(
                time_now=time_now,
                dof_pos=masked_dof_pos,  # Use masked positions if necessary for walking
                dof_vel=masked_dof_vel,  # Use masked velocities if necessary
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                vx=self.remoteControlService.get_vx_cmd(),
                vy=self.remoteControlService.get_vy_cmd(),
                vyaw=self.remoteControlService.get_vyaw_cmd(),
            )
        elif self.state == RobotState.STANDING_UP:
            active_policy = self.standup_policy
            # Standup policy will not use vx, vy, vyaw or masking
            policy_targets = active_policy.inference(
                time_now=time_now,
                dof_pos=self.dof_pos, # Use actual positions for standup
                dof_vel=self.dof_vel, # Use actual velocities for standup
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                # vx, vy, vyaw are likely ignored by standup_policy.inference
            )
            # Optional: Check if standup is complete (e.g., IMU level) and transition back to WALKING or a different state
            # if self.is_standup_complete(low_state_msg): # Requires access to low_state_msg or storing relevant parts
            #     self.logger.info("Stand-up sequence complete. Transitioning back to WALKING (or IDLE).")
            #     self.state = RobotState.WALKING # Or a new IDLE/READY state
            #     # Reset policy states if needed
        else:
            # Should not happen if state logic above is correct, but good practice to handle
             self.logger.warning(f"Inference called in unexpected state: {self.state}")
             return

        # Apply policy targets to lower body (always controlled by the active policy)
        # Might need adjustment if standup policy controls arms differently
        for i in self.lower_body_indices:
             self.dof_target[i] = policy_targets[i]
        
        # Apply appropriate control to upper body based on mode
        upper_body_mode = self.body_part_control_mode[BodyPart.UPPER_BODY]
        
        # If standing up, potentially override upper body control to use standup policy targets or a fixed safe pose
        if self.state == RobotState.STANDING_UP:
             # Example: Force upper body to follow standup policy targets (indices 0-10)
             upper_body_targets = policy_targets[self.upper_body_indices] 
             for i, idx in enumerate(self.upper_body_indices):
                  self.dof_target[idx] = upper_body_targets[i]
             # Or set to a fixed safe pose during standup
             # safe_pose = self.standup_policy.default_dof_pos[self.upper_body_indices]
             # for i, idx in enumerate(self.upper_body_indices):
             #     self.dof_target[idx] = safe_pose[i]
        elif upper_body_mode == UpperBodyControlMode.TELEOP.value:
            # Apply manual/teleop targets to upper body (only if WALKING)
            for i, idx in enumerate(self.upper_body_indices):
                self.dof_target[idx] = self.manual_upper_body_positions[i]
                
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
            
            # Only advance publish time and send commands if walking or standing up
            if self.state not in [RobotState.WALKING, RobotState.STANDING_UP]:
                # Optional: Send damping command or zero torque when fallen/waiting
                # for i in range(B1JointCnt):
                #     self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i] # Hold position
                #     self.low_cmd.motor_cmd[i].kp = 0
                #     self.low_cmd.motor_cmd[i].dq = 0
                #     self.low_cmd.motor_cmd[i].kd = 1.0 # Apply some damping
                #     self.low_cmd.motor_cmd[i].tau = 0
                # self._send_cmd(self.low_cmd) 
                time.sleep(0.01) # Prevent busy-waiting
                continue
                
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
                # Apply specific filtering based on the active control source
                if self.state == RobotState.WALKING and self.body_part_control_mode[BodyPart.UPPER_BODY] == "teleop":
                    # Stronger filtering for teleop control to be smoother
                    # Higher first coefficient (0.9) means more of the previous value is retained
                    # resulting in smoother, less jerky movements
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.9 + self.dof_target[i] * 0.1
                else:
                    # Standard filtering for policy control (walking or standing up)
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.8 + self.dof_target[i] * 0.2

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
                # Apply Kp based on state? Maybe 0 Kp during standup for some joints?
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

# Helper function for non-blocking keyboard input
def listen_for_key(controller):
    """Runs in a separate thread to listen for 's' key press."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno()) # Set terminal to raw mode
        print("Input listener started. Press 's' when ready to stand up...")
        while controller.running:
            # Check if data is available on stdin (non-blocking)
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                if char == 's' and controller.state == RobotState.WAITING_FOR_STANDUP:
                    print("\n's' key pressed. Initiating stand-up sequence.")
                    controller.state = RobotState.STANDING_UP
                    controller.standup_start_time = controller.timer.get_time()
                    # Potentially reset standup policy internal state here if needed
                    # controller.standup_policy.reset() 
                elif ord(char) == 3: # CTRL+C
                     print("\nCtrl+C detected in input thread. Signalling shutdown.")
                     controller.running = False
                     break
            time.sleep(0.05) # Small sleep to prevent high CPU usage
    except Exception as e:
        print(f"\nError in keyboard listener: {e}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings) # Restore terminal settings
        print("Input listener stopped.")

if __name__ == "__main__":
    import argparse
    import signal
    import sys
    import os

    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        # Start the keyboard listener thread
        input_thread = threading.Thread(target=listen_for_key, args=(controller,), daemon=True)
        input_thread.start()

        try:
            while controller.running:
                # The run method now handles state-based execution
                controller.run()
                
                # Check for standup success or timeout
                if controller.state == RobotState.STANDING_UP and controller.standup_start_time is not None:
                    current_time = controller.timer.get_time()
                    time_since_standup_start = current_time - controller.standup_start_time

                    # Check for successful stand-up
                    if time_since_standup_start >= STANDUP_SUCCESS_DURATION_S:
                        # Read current IMU values (already updated by _low_state_handler)
                        # Ensure publish_lock is used if accessing shared IMU data, but current_imu_rpy is updated in low_state_handler
                        roll_ok = abs(controller.current_imu_rpy[0]) < UPRIGHT_IMU_THRESHOLD_RP
                        pitch_ok = abs(controller.current_imu_rpy[1]) < UPRIGHT_IMU_THRESHOLD_RP
                        
                        if roll_ok and pitch_ok:
                            controller.logger.info(
                                f"Stand-up successful! Robot is upright after {time_since_standup_start:.2f}s. "
                                f"IMU RPY: {controller.current_imu_rpy}. Transitioning to WALKING."
                            )
                            controller.state = RobotState.WALKING
                            controller.standup_start_time = None # Reset standup timer
                            # Optionally reset walking policy if needed: controller.policy.reset_state() (if such a method exists)
                        elif time_since_standup_start >= STANDUP_ATTEMPT_TIMEOUT_S:
                            # Timeout for stand-up attempt
                            controller.logger.warning(
                                f"Stand-up attempt timed out after {time_since_standup_start:.2f}s. "
                                f"IMU RPY: {controller.current_imu_rpy}. Transitioning back to FALLEN."
                            )
                            controller.state = RobotState.FALLEN
                            controller.fall_detected_time = current_time # Reset fall time for next 5s wait
                            controller.standup_start_time = None # Reset standup timer
                    
                    # Fallback: Original timeout logic (kept if success check is earlier or different)
                    # elif time_since_standup_start > STANDUP_ATTEMPT_TIMEOUT_S: # Adjusted from 15.0
                    #      print(f"Stand-up sequence timed out after {STANDUP_ATTEMPT_TIMEOUT_S} seconds.")
                    #      controller.state = RobotState.FALLEN 
                    #      controller.fall_detected_time = controller.timer.get_time() 
                    #      controller.standup_start_time = None

            # If loop exits due to controller.running = False (e.g., Ctrl+C)
            print("Main loop exited. Setting robot to Damping mode.")
            controller.client.ChangeMode(RobotMode.kDamping)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received in main thread. Cleaning up...")
            controller.running = False # Signal threads to stop
            # controller.cleanup() is handled by __exit__
        finally:
             if input_thread.is_alive():
                  print("Waiting for input thread to finish...")
                  input_thread.join(timeout=1.0) # Wait briefly for thread cleanup

    print("Program finished.")

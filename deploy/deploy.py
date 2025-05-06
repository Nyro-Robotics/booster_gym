import numpy as np
import time
import yaml
import logging
import threading

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

# Global constants for upper body control modes
# Set one of these to the string value to enable that control mode
# Options: "policy", "teleop", "sine"
UPPER_BODY_CONTROL_MODE = "teleop"  # Default to policy control

# Control parameter for arm gains
# Lower value means smoother movements with less stiffness
# Range: 0.1 (very soft) to 1.0 (full stiffness)
ARM_STIFFNESS_FACTOR = 0.05

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

        # Define joint indices for body parts based on the actual robot configuration
        # Head and arms (upper body)
        self.upper_body_indices = [
            0, 1,                # Head (yaw, pitch)
            2, 3, 4, 5,          # Left arm (shoulder pitch, roll, yaw, elbow)
            6, 7, 8, 9,          # Right arm (shoulder pitch, roll, yaw, elbow)
            10                    # Waist yaw
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
        self.sine_upper_body_positions = np.copy(self.manual_upper_body_positions)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
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
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
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
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()
        
        # Create copies of dof_pos and dof_vel for policy inference
        masked_dof_pos = np.copy(self.dof_pos)
        masked_dof_vel = np.copy(self.dof_vel)
        
        # If in sine mode, mask upper body positions and velocities with defaults
        # so the policy thinks the arms are in their default positions and not moving
        if self.body_part_control_mode[BodyPart.UPPER_BODY] == UpperBodyControlMode.SINE.value:
            # Get default positions from policy
            default_positions = self.policy.default_dof_pos
            
            # Replace upper body positions with default positions
            for i in self.upper_body_indices:
                masked_dof_pos[i] = default_positions[i]
                masked_dof_vel[i] = 0.0  # Zero velocity

        # Get policy inference for all joints
        policy_targets = self.policy.inference(
            time_now=time_now,
            dof_pos=masked_dof_pos,  # Use masked positions
            dof_vel=masked_dof_vel,  # Use masked velocities
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )
        
        # Apply policy targets to lower body
        if self.body_part_control_mode[BodyPart.LOWER_BODY] == "policy":
            for i in self.lower_body_indices:
                self.dof_target[i] = policy_targets[i]
        
        # Apply appropriate control to upper body based on mode
        upper_body_mode = self.body_part_control_mode[BodyPart.UPPER_BODY]
        
        if upper_body_mode == UpperBodyControlMode.TELEOP.value:
            # Apply manual/teleop targets to upper body
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
                if self.body_part_control_mode[BodyPart.UPPER_BODY] == "teleop":
                    # Stronger filtering for teleop control to be smoother
                    # Higher first coefficient (0.9) means more of the previous value is retained
                    # resulting in smoother, less jerky movements
                    self.filtered_dof_target[i] = self.filtered_dof_target[i] * 0.9 + self.dof_target[i] * 0.1
                else:
                    # Standard filtering for policy control
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
        
    def set_upper_body_positions(self, positions):
        """Set manual positions for upper body joints (e.g., from VR tracking)
        
        Args:
            positions: Array of joint positions for upper body joints (indices 0-10)
        """
        if len(positions) != len(self.upper_body_indices):
            self.logger.error(f"Expected {len(self.upper_body_indices)} positions, got {len(positions)}")
            return
            
        self.manual_upper_body_positions = np.array(positions, dtype=np.float32)
        
    def set_body_part_control_mode(self, body_part, mode):
        """Set the control mode for a body part
        
        Args:
            body_part: BodyPart enum (LOWER_BODY or UPPER_BODY)
            mode: Control mode ("policy", "teleop", or "sine" for upper body; "policy" for lower body)
        """
        if body_part == BodyPart.UPPER_BODY:
            valid_modes = [mode.value for mode in UpperBodyControlMode]
            if mode not in valid_modes:
                self.logger.error(f"Invalid upper body control mode: {mode}. Must be one of {valid_modes}")
                return
        elif body_part == BodyPart.LOWER_BODY:
            if mode != "policy":
                self.logger.error(f"Invalid lower body control mode: {mode}. Only 'policy' is supported")
                return
        else:
            self.logger.error(f"Invalid body part: {body_part}")
            return
            
        self.body_part_control_mode[body_part] = mode
        self.logger.info(f"Set {body_part} control mode to {mode}")


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

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()

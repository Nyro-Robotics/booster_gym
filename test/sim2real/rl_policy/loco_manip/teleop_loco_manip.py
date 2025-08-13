import os
import sys
import time
import threading
import asyncio
from typing import Dict, Any, Optional
from collections import deque

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append("../../")
sys.path.append("../../../")  # Add path to reach test/ directory

import pinocchio as pin
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from termcolor import colored

# Import teleoperation utilities from deploy.py
from utils.websocket_client import WebSocketClient, send_initialization_message_websocket, create_robot_joint_initialization_message
from utils.zeromq_client import ZeroMQClient, send_initialization_message_zeromq


class TeleopUpperBodyController:
    """Upper body controller that receives joint positions via teleoperation"""
    
    def __init__(self, config, communication_mode="websocket", ws_host="localhost", ws_port=8765, zmq_address="tcp://localhost:5555"):
        self.config = config
        self.communication_mode = communication_mode
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.zmq_address = zmq_address
        
        # Joint data storage
        self.latest_joint_data = {}
        self.latest_hand_data = {}
        self.teleop_active = False
        self.teleop_thread = None
        
        # Joint position smoothing
        self.joint_smoothing_factor = 0.1
        self.filtered_joint_positions = {}
        
        # Latency tracking
        self.message_latencies = deque(maxlen=100)
        self.joint_position_latencies = deque(maxlen=100)
        self.total_latency_sum = 0.0
        self.total_latency_count = 0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        
        # Joint mapping from teleop to robot indices
        self.joint_mapping = {
            "AAHead_yaw": 0, "Head_pitch": 1,
            "Left_Shoulder_Pitch": 2, "Left_Shoulder_Roll": 3, "Left_Elbow_Pitch": 4, 
            "Left_Elbow_Yaw": 5, "Left_Wrist_Pitch": 6, "Left_Wrist_Yaw": 7, "Left_Hand_Roll": 8,
            "Right_Shoulder_Pitch": 9, "Right_Shoulder_Roll": 10, "Right_Elbow_Pitch": 11, 
            "Right_Elbow_Yaw": 12, "Right_Wrist_Pitch": 13, "Right_Wrist_Yaw": 14, "Right_Hand_Roll": 15,
            "Waist": 16
        }
        
        self.logger = None
        
    def start_teleoperation(self):
        """Start teleoperation communication thread"""
        if self.teleop_active:
            return False
            
        self.teleop_active = True
        self.reset_latency_tracking()
        
        if self.communication_mode == "websocket":
            self._start_websocket_client()
        elif self.communication_mode == "zeromq":
            self._start_zeromq_client()
        else:
            print(f"Unknown communication mode: {self.communication_mode}")
            self.teleop_active = False
            return False
            
        return True
    
    def stop_teleoperation(self):
        """Stop teleoperation"""
        self.teleop_active = False
        if self.teleop_thread:
            self.teleop_thread.join(timeout=2.0)
        self.filtered_joint_positions.clear()
        
    def _start_websocket_client(self):
        """Start WebSocket client in separate thread"""
        ws_client = WebSocketClient(self.ws_host, self.ws_port)
        
        # Create wrapper classes for shared data
        class SharedBool:
            def __init__(self, value):
                self.value = value
        
        class SharedFloat:
            def __init__(self, value):
                self.value = value
        
        teleop_active_ref = SharedBool(self.teleop_active)
        total_latency_sum_ref = SharedFloat(self.total_latency_sum)
        total_latency_count_ref = SharedFloat(self.total_latency_count)
        min_latency_ref = SharedFloat(self.min_latency)
        max_latency_ref = SharedFloat(self.max_latency)
        
        self.teleop_thread = threading.Thread(
            target=lambda: asyncio.run(ws_client.handle_connection(
                teleop_active_ref, self.latest_joint_data, self.latest_hand_data,
                self.message_latencies, self.joint_position_latencies,
                total_latency_sum_ref, total_latency_count_ref,
                min_latency_ref, max_latency_ref
            )),
            daemon=True
        )
        self.teleop_thread.start()
    
    def _start_zeromq_client(self):
        """Start ZeroMQ client in separate thread"""
        zmq_client = ZeroMQClient(self.zmq_address)
        
        class SharedBool:
            def __init__(self, value):
                self.value = value
        
        class SharedFloat:
            def __init__(self, value):
                self.value = value
        
        teleop_active_ref = SharedBool(self.teleop_active)
        total_latency_sum_ref = SharedFloat(self.total_latency_sum)
        total_latency_count_ref = SharedFloat(self.total_latency_count)
        min_latency_ref = SharedFloat(self.min_latency)
        max_latency_ref = SharedFloat(self.max_latency)
        
        self.teleop_thread = threading.Thread(
            target=lambda: zmq_client.handle_subscriber(
                teleop_active_ref, self.latest_joint_data, self.latest_hand_data,
                self.message_latencies, self.joint_position_latencies,
                total_latency_sum_ref, total_latency_count_ref,
                min_latency_ref, max_latency_ref
            ),
            daemon=True
        )
        self.teleop_thread.start()
    
    def reset_latency_tracking(self):
        """Reset latency tracking variables"""
        self.message_latencies.clear()
        self.joint_position_latencies.clear()
        self.total_latency_sum = 0.0
        self.total_latency_count = 0
        self.min_latency = float('inf')
        self.max_latency = 0.0
    
    def get_upper_body_positions(self):
        """Extract upper body joint positions from teleoperation data
        
        Returns:
            numpy array: Upper body joint positions in format compatible with ref_upper_dof_pos
        """
        if not self.latest_joint_data or 'upper_body' not in self.latest_joint_data:
            return None
            
        upper_body_joints = self.latest_joint_data['upper_body']
        
        # Create array for upper body positions (17 joints: head + arms + waist)
        upper_body_positions = np.zeros(17)
        
        for joint_name, joint_value in upper_body_joints.items():
            joint_index = self.joint_mapping.get(joint_name)
            if joint_index is not None and joint_index <= 16:
                # Apply smoothing filter
                if joint_name in self.filtered_joint_positions:
                    self.filtered_joint_positions[joint_name] = (
                        self.filtered_joint_positions[joint_name] * self.joint_smoothing_factor + 
                        joint_value * (1.0 - self.joint_smoothing_factor)
                    )
                else:
                    self.filtered_joint_positions[joint_name] = joint_value
                
                # Use filtered value
                upper_body_positions[joint_index] = self.filtered_joint_positions[joint_name]
        
        return upper_body_positions
    
    def send_robot_initialization(self, current_positions):
        """Send current robot positions to teleoperator for initialization"""
        # Convert to list if numpy array
        if hasattr(current_positions, 'tolist'):
            positions_list = current_positions.tolist()
        else:
            positions_list = list(current_positions)
            
        init_message = create_robot_joint_initialization_message(positions_list)
        
        if self.communication_mode == "websocket":
            send_initialization_message_websocket(init_message, self.ws_host, self.ws_port)
        elif self.communication_mode == "zeromq":
            send_initialization_message_zeromq(init_message, self.zmq_address)
            
        print(f"📤 Sent robot initialization to teleoperator ({self.communication_mode})")
        print(f"    Joint positions: {len(positions_list)} joints")


class TeleopLocoManipPolicy(DecLocomotionPolicy):
    """LocoManipPolicy with integrated teleoperation upper body control"""
    
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25,
        communication_mode="websocket", ws_host="localhost", ws_port=8765, zmq_address="tcp://localhost:5555"
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)

        # Replace IK controller with teleoperation controller
        self.upper_body_controller = None
        self.teleop_controller = None
        
        # Track init completion for teleop synchronization
        self.init_completion_sent = False
        
        if self.config.get("use_teleoperation_controller", False):
            self.init_teleoperation_controller(communication_mode, ws_host, ws_port, zmq_address)

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command
        return current_obs_dict

    def init_teleoperation_controller(self, communication_mode="websocket", ws_host="localhost", ws_port=8765, zmq_address="tcp://localhost:5555"):
        """Initialize teleoperation controller"""
        self.teleop_controller = TeleopUpperBodyController(
            self.config, communication_mode, ws_host, ws_port, zmq_address
        )
        self.teleop_controller.logger = self.logger
        
        # Start teleoperation communication
        success = self.teleop_controller.start_teleoperation()
        if success:
            self.logger.info(colored("Teleoperation controller started successfully", "green"))
        else:
            self.logger.error("Failed to start teleoperation controller")
        
    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        # WBC actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        
        # Apply teleoperation controller
        if self.teleop_controller:
            # Get upper body positions from teleoperation
            teleop_positions = self.teleop_controller.get_upper_body_positions()
            if teleop_positions is not None:
                # Map teleoperation positions to ref_upper_dof_pos
                # Upper body indices in policy correspond to joints 0-16 in teleop
                for i, teleop_idx in enumerate(range(min(17, self.num_upper_dofs))):
                    if i < len(teleop_positions):
                        self.ref_upper_dof_pos[0, i] = teleop_positions[i]
                
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Updated upper body positions from teleoperation")

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
            
            # Check if init is complete and send teleop initialization
            if (self.init_count >= 500 and not self.init_completion_sent and 
                self.teleop_controller and self.teleop_controller.teleop_active):
                # Send current joint positions to teleoperator
                current_joint_positions = robot_state_data[0, 7 : 7 + self.num_dofs].copy()
                self.teleop_controller.send_robot_initialization(current_joint_positions)
                self.init_completion_sent = True
                self.logger.info(colored("Init complete - sent joint positions to teleoperator", "cyan"))
                
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = scaled_policy_action + self.default_dof_angles
        
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])

    def update_waypoints(self):
        """Update waypoints for IK controller (fallback mode)"""
        if hasattr(self, 'EE_left_R') and hasattr(self, 'EE_right_R'):
            self.waypoints_left = [
                pin.SE3(self.EE_left_R.astype(np.float64), np.array([self.EE_left_x, self.EE_left_y, self.EE_left_z]))
            ]
            self.waypoints_right = [
                pin.SE3(self.EE_right_R.astype(np.float64), np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z]))
            ]

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == "m":
            self.lin_vel_command[:, 0] = -1.0
            self.logger.info(colored(f"lin_vel_command: {self.lin_vel_command}", "green"))
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)
        elif keycode == "i":
            # Reset init completion flag when 'i' is pressed for new initialization
            self.init_completion_sent = False
            self.logger.info(colored("Init state reset - will send teleop sync when complete", "yellow"))
        elif keycode == "t":
            # Toggle teleoperation
            if self.teleop_controller:
                if self.teleop_controller.teleop_active:
                    self.teleop_controller.stop_teleoperation()
                    self.logger.info(colored("Teleoperation stopped", "red"))
                else:
                    self.teleop_controller.start_teleoperation()
                    self.logger.info(colored("Teleoperation started", "green"))

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        if cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    def _handle_base_height_control(self, keycode):
        """Handle base height control."""
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        """Handle joystick base height control."""
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")
        if self.teleop_controller:
            print(f"Teleoperation active: {self.teleop_controller.teleop_active}")
            print(f"Communication mode: {self.teleop_controller.communication_mode}")

    def cleanup(self):
        """Clean up resources"""
        if self.teleop_controller:
            self.teleop_controller.stop_teleoperation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot with Teleoperation Control")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    parser.add_argument("--communication", type=str, choices=['websocket', 'zeromq'], default='websocket',
                       help='Communication method: websocket or zeromq (default: websocket)')
    parser.add_argument("--ws-host", type=str, default='localhost',
                       help='WebSocket server host for teleoperation (default: localhost)')
    parser.add_argument("--ws-port", type=int, default=8765,
                       help='WebSocket server port for teleoperation (default: 8765)')
    parser.add_argument("--zmq-address", type=str, default='tcp://localhost:5555',
                       help='ZeroMQ address for teleoperation (default: tcp://localhost:5555)')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    # Enable teleoperation controller in config
    config["use_teleoperation_controller"] = True
    config["use_upper_body_controller"] = False  # Disable IK controller

    policy = TeleopLocoManipPolicy(
        config=config, 
        model_path=model_path, 
        rl_rate=50, 
        policy_action_scale=0.25,
        communication_mode=args.communication,
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        zmq_address=args.zmq_address
    )
    
    try:
        print(f"Starting TeleopLocoManipPolicy with {args.communication} communication")
        print(f"Press 't' to toggle teleoperation on/off")
        print(f"Communication endpoint: {args.ws_host}:{args.ws_port}" if args.communication == "websocket" else f"Communication endpoint: {args.zmq_address}")
        policy.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        policy.cleanup()
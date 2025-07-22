import os
import sys
import time

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.base_policy import BasePolicy

from termcolor import colored

class DecLocomotionPolicy(BasePolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.num_lower_dofs = self.num_dofs - self.num_upper_dofs

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action[:, :self.num_lower_dofs]
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command
        current_obs_buffer_dict["command_waist_dofs"] = self.waist_dofs_command
        current_obs_buffer_dict["phase_time"] = self._get_obs_phase_time()
        current_obs_buffer_dict["sin_phase"] = np.sin(2 * np.pi * current_obs_buffer_dict["phase_time"])
        current_obs_buffer_dict["cos_phase"] = np.cos(2 * np.pi * current_obs_buffer_dict["phase_time"])
        current_obs_buffer_dict["ref_upper_dof_pos"] = self.ref_upper_dof_pos

        return current_obs_buffer_dict
    
    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # Lower body actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale
        # Combine upper body actions
        scaled_policy_action = np.concatenate([scaled_policy_action, self.ref_upper_dof_pos], axis=1)
        
        return scaled_policy_action

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)
        
        # Locomotion-specific commands
        if keycode in ["w", "s", "a", "d"]:
            self._handle_velocity_control(keycode)
        elif keycode in ["q", "e"]:
            self._handle_angular_velocity_control(keycode)
        elif keycode == "=":
            self._handle_stand_command()
        elif keycode == "z":
            self._handle_zero_velocity()
            
        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_joystick_button(cur_key)
        
        # Locomotion-specific commands
        if cur_key == "R2":
            self._handle_stand_command()
        elif cur_key == "L2":
            self._handle_zero_velocity()

    def _handle_velocity_control(self, keycode):
        """Handle linear velocity control."""
        if not self.stand_command[0, 0]:
            return
            
        if keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1
    
    def _handle_angular_velocity_control(self, keycode):
        """Handle angular velocity control."""
        if keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1
    
    def _handle_stand_command(self):
        """Handle stand command toggle."""
        self.stand_command[0, 0] = 1 - self.stand_command[0, 0]
        if self.stand_command[0, 0] == 0:
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.logger.info(colored("Stance command", "blue"))
        else:
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("Walk command", "blue"))

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))
    
    def _print_control_status(self):
        """Print current control status."""
        print(f"Linear velocity command: {self.lin_vel_command}")
        print(f"Angular velocity command: {self.ang_vel_command}")
        print(f"Stand command: {self.stand_command}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    
    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = DecLocomotionPolicy(config, model_path)
    policy.run()

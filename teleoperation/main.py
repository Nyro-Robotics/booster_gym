"""Robot teleoperation tool for MuJoCo environments with inverse kinematics.

This module provides functionality for:
- Loading and controlling a robot in MuJoCo
- Specifying target positions for the robot's hands
- Solving inverse kinematics to move the hands to target positions
- Interactive visualization and control

python -m teleoperation.main --task=T1 --headless
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import time
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import attrs
from scipy.optimize import minimize
import pygame
from mujoco import viewer
import cv2
import glfw

from teleoperation.record import RecordingState
from teleoperation.utils import initialize_robot_from_config, initialize_robot_state, quat_rotate, setup_viewer
from teleoperation.teleop_types import RobotState
from teleoperation.teleop_ik import IKSolver

@attrs.define
class UserInterface:
    """User interface for controlling the robot."""
    robot_state: RobotState
    ik_solver: IKSolver
    step_size: float = 0.01  # Step size for position adjustments
    
    def process_keyboard_input(self) -> bool:
        """Process keyboard input for robot control.
        
        Returns:
            Boolean indicating whether to continue running
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Switch between left and right hand
                if event.key == pygame.K_TAB:
                    self.robot_state.current_mode = "right" if self.robot_state.current_mode == "left" else "left"
                    print(f"Switched to {self.robot_state.current_mode} hand control")
                
                # Toggle IK mode
                elif event.key == pygame.K_SPACE:
                    self.robot_state.ik_active = not self.robot_state.ik_active
                    mode = "enabled" if self.robot_state.ik_active else "disabled"
                    print(f"IK mode {mode}")
                
                # Toggle freeze lower body
                elif event.key == pygame.K_f:
                    self.robot_state.freeze_lower_body = not self.robot_state.freeze_lower_body
                    mode = "enabled" if self.robot_state.freeze_lower_body else "disabled"
                    print(f"Freeze lower body {mode}")
                
                # Quit
                elif event.key == pygame.K_q:
                    return False
                
                # Reset target position to current hand position
                elif event.key == pygame.K_r:
                    if self.robot_state.current_mode == "left":
                        self.robot_state.target_left_pos = self.robot_state.get_hand_position("left")
                    else:
                        self.robot_state.target_right_pos = self.robot_state.get_hand_position("right")
                    print(f"Reset {self.robot_state.current_mode} hand target to current position")
                
                # Add a key to toggle dual hand control
                elif event.key == pygame.K_b:
                    self.robot_state.control_both_hands = not self.robot_state.control_both_hands
                    mode = "enabled" if self.robot_state.control_both_hands else "disabled"
                    print(f"Dual hand control {mode}")
        
        # Get pressed keys for continuous movement
        keys = pygame.key.get_pressed()
        
        # Determine which target to modify
        target = self.robot_state.target_left_pos if self.robot_state.current_mode == "left" else self.robot_state.target_right_pos
        
        # Modify target position based on key presses
        if keys[pygame.K_w]:  # Forward (+X)
            target[0] += self.step_size
        if keys[pygame.K_s]:  # Backward (-X)
            target[0] -= self.step_size
        if keys[pygame.K_a]:  # Left (+Y)
            target[1] += self.step_size
        if keys[pygame.K_d]:  # Right (-Y)
            target[1] -= self.step_size
        if keys[pygame.K_e]:  # Up (+Z)
            target[2] += self.step_size
        if keys[pygame.K_c]:  # Down (-Z)
            target[2] -= self.step_size
        
        # Update the appropriate target
        if self.robot_state.current_mode == "left":
            self.robot_state.target_left_pos = target
        else:
            self.robot_state.target_right_pos = target
        
        return True
    
    def update_robot(self) -> None:
        """Update robot position using IK if active."""
        if self.robot_state.ik_active:
            if self.robot_state.control_both_hands:
                # Solve IK for both hands
                self.ik_solver.solve_ik("left", self.robot_state.target_left_pos)
                self.ik_solver.solve_ik("right", self.robot_state.target_right_pos)
            else:
                # Solve IK for the active hand only
                if self.robot_state.current_mode == "left":
                    self.ik_solver.solve_ik("left", self.robot_state.target_left_pos)
                else:
                    self.ik_solver.solve_ik("right", self.robot_state.target_right_pos)
    
    def render_targets(self, viewer) -> None:
        """Render target markers in the viewer."""
        try:
            # Get current hand positions
            left_pos = self.robot_state.get_hand_position("left")
            right_pos = self.robot_state.get_hand_position("right")
            
            # Add markers for targets
            viewer.add_marker(
                pos=self.robot_state.target_left_pos,
                size=np.array([0.03, 0.03, 0.03]),
                rgba=np.array([1, 0, 0, 0.7]),  # Red with transparency
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="Left Target"
            )
            
            viewer.add_marker(
                pos=self.robot_state.target_right_pos,
                size=np.array([0.03, 0.03, 0.03]),
                rgba=np.array([0, 0, 1, 0.7]),  # Blue with transparency
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="Right Target"
            )
            
            # Add markers for current hand positions
            viewer.add_marker(
                pos=left_pos,
                size=np.array([0.03, 0.03, 0.03]),
                rgba=np.array([1, 0.5, 0.5, 0.7]),  # Light red
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="Left Hand"
            )
            
            viewer.add_marker(
                pos=right_pos,
                size=np.array([0.03, 0.03, 0.03]),
                rgba=np.array([0.5, 0.5, 1, 0.7]),  # Light blue
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="Right Hand"
            )
            
            # Draw lines connecting hands to targets
            viewer.add_marker(
                pos=np.zeros(3),  # Not used for lines
                size=np.zeros(3),  # Not used for lines
                rgba=np.array([1, 0, 0, 0.4]),  # Red with transparency
                type=mujoco.mjtGeom.mjGEOM_LINE,
                label="",
                from_=left_pos,
                to=self.robot_state.target_left_pos
            )
            
            viewer.add_marker(
                pos=np.zeros(3),  # Not used for lines
                size=np.zeros(3),  # Not used for lines
                rgba=np.array([0, 0, 1, 0.4]),  # Blue with transparency
                type=mujoco.mjtGeom.mjGEOM_LINE,
                label="",
                from_=right_pos,
                to=self.robot_state.target_right_pos
            )
            
        except Exception as e:
            # Fall back to printing if visualization fails
            left_pos = self.robot_state.get_hand_position("left")
            right_pos = self.robot_state.get_hand_position("right")
            
            left_target = self.robot_state.target_left_pos
            right_target = self.robot_state.target_right_pos
            
            left_dist = np.linalg.norm(left_pos - left_target)
            right_dist = np.linalg.norm(right_pos - right_target)
            
            active = self.robot_state.current_mode
            
            if active == "left" and self.robot_state.ik_active:
                print(f"\rLeft hand: {left_pos.round(3)} → {left_target.round(3)} (dist: {left_dist:.3f})", end="")
            elif active == "right" and self.robot_state.ik_active:
                print(f"\rRight hand: {right_pos.round(3)} → {right_target.round(3)} (dist: {right_dist:.3f})", end="")

@attrs.define
class HeadlessInterface:
    """Terminal interface for controlling the robot in headless mode."""
    robot_state: RobotState
    ik_solver: IKSolver
    recording_state: RecordingState
    
    def run_simulation(self, target_fps: int = 30, duration: float = 5.0) -> None:
        """Run a headless simulation to reach the current targets.
        
        Args:
            target_fps: Target frames per second for recording
            duration: Maximum duration in seconds
        """
        self.headless_simulation_loop(
            robot_state=self.robot_state,
            recording_state=self.recording_state,
            target_fps=target_fps,
            duration=duration
        )
    
    def run(self) -> None:
        """Run the headless interface with terminal input."""
        print("\nHeadless Teleoperation Mode")
        print("Commands:")
        print("  left x y z - Set left hand target position (relative to head)")
        print("  right x y z - Set right hand target position (relative to head)")
        print("  record - Start recording")
        print("  stop - Stop recording")
        print("  status - Show current positions")
        print("  record_solve_left - Record and solve IK for left hand")
        print("  record_solve_right - Record and solve IK for right hand")
        print("  run - Run simulation to reach targets")
        print("  quit - Exit the program")
        
        running = True
        while running:
            try:
                command = input("\nEnter command: ").strip().lower()
                parts = command.split()
                
                if not parts:
                    continue
                
                if parts[0] == "quit":
                    running = False
                
                elif parts[0] == "run":
                    self.run_simulation(
                        target_fps=self.recording_state.fps,
                        duration=2.0  # Default duration
                    )
                
                elif parts[0] == "left" or parts[0] == "right":
                    if len(parts) != 4:
                        print("Error: Expected 3 coordinates. Usage: left/right x y z")
                        continue
                    
                    try:
                        # Parse coordinates
                        coords = [float(p) for p in parts[1:4]]
                        
                        # Get head position and orientation
                        head_pos, head_quat = self._get_head_pose()
                        
                        # Convert coordinates to world frame
                        world_coords = self._convert_to_world_frame(coords, head_pos, head_quat)
                        
                        # Set target position
                        if parts[0] == "left":
                            self.robot_state.target_left_pos = world_coords
                            self.robot_state.current_mode = "left"
                            print(f"Set left hand target to {coords} (relative to head)")
                            print(f"World coordinates: {world_coords.round(3)}")
                        else:
                            self.robot_state.target_right_pos = world_coords
                            self.robot_state.current_mode = "right"
                            print(f"Set right hand target to {coords} (relative to head)")
                            print(f"World coordinates: {world_coords.round(3)}")
                    
                    except ValueError:
                        print("Error: Coordinates must be numbers")
                elif parts[0] == "record_solve_left":
                    # Record and solve IK for left hand
                    print("Recording and solving IK for left hand...")
                    self.ik_solver.solve_ik("left", self.robot_state.target_left_pos, record_convergence=True)
                    self._print_status()
                
                elif parts[0] == "record_solve_right":
                    # Record and solve IK for right hand
                    print("Recording and solving IK for right hand...")
                    self.ik_solver.solve_ik("right", self.robot_state.target_right_pos, record_convergence=True)
                    self._print_status()

                elif parts[0] == "record":
                    self.recording_state.start_recording()
                
                elif parts[0] == "stop":
                    self.recording_state.stop_recording()
                
                elif parts[0] == "status":
                    self._print_status()
                
                else:
                    print(f"Unknown command: {parts[0]}")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                running = False
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _get_head_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the position and orientation of the robot's head or root body.
        
        Returns:
            Tuple of (position, quaternion)
        """
        # Try to find head body ID
        head_id = mujoco.mj_name2id(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2") # nathanjzhao: booster hardcoded head name
                 
        # If still not found, use the root body (usually at index 1, after the world body)
        if head_id == -1:
            print("Warning: Could not find head body. Check the .XML file for where the head is.\n Now using root body instead.")
            raise ValueError("Could not find head body. Check the .XML file for where the head is.")
        
        # Get position and orientation from the body
        head_pos = self.robot_state.mj_data.xpos[head_id].copy()
        head_quat = self.robot_state.mj_data.xquat[head_id].copy()  # [w, x, y, z]
        
        return head_pos, head_quat
    
    def _convert_to_world_frame(self, local_coords: List[float], origin: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Convert coordinates from head-relative to world frame.
        
        Args:
            local_coords: Coordinates in head-relative frame [x, y, z]
            origin: Origin position (head position)
            quat: Quaternion representing head orientation [w, x, y, z]
            
        Returns:
            Coordinates in world frame
        """
        # Convert to numpy array
        local_vec = np.array(local_coords)
        
        # Rotate vector using quaternion
        # MuJoCo quaternions are [w, x, y, z] but we need [x, y, z, w] for our function
        q = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        # Rotate vector
        rotated_vec = quat_rotate(q, local_vec)
        
        # Translate to world frame
        world_coords = origin + rotated_vec
        
        return world_coords
    
    def _print_status(self) -> None:
        """Print current robot status."""
        left_pos = self.robot_state.get_hand_position("left")
        right_pos = self.robot_state.get_hand_position("right")
        
        left_target = self.robot_state.target_left_pos
        right_target = self.robot_state.target_right_pos
        
        left_dist = np.linalg.norm(left_pos - left_target)
        right_dist = np.linalg.norm(right_pos - right_target)
        
        head_pos, head_quat = self._get_head_pose()
        
        # Convert world positions to head-relative
        left_rel = self._convert_to_head_frame(left_pos, head_pos, head_quat)
        right_rel = self._convert_to_head_frame(right_pos, head_pos, head_quat)
        left_target_rel = self._convert_to_head_frame(left_target, head_pos, head_quat)
        right_target_rel = self._convert_to_head_frame(right_target, head_pos, head_quat)
        
        print("\nCurrent Status:")
        print(f"Head position: {head_pos.round(3)}")
        print(f"Left hand: world={left_pos.round(3)}, head-relative={left_rel.round(3)}")
        print(f"Left target: world={left_target.round(3)}, head-relative={left_target_rel.round(3)}, distance={left_dist:.3f}")
        print(f"Right hand: world={right_pos.round(3)}, head-relative={right_rel.round(3)}")
        print(f"Right target: world={right_target.round(3)}, head-relative={right_target_rel.round(3)}, distance={right_dist:.3f}")
    
    def _convert_to_head_frame(self, world_coords: np.ndarray, head_pos: np.ndarray, head_quat: np.ndarray) -> np.ndarray:
        """Convert coordinates from world frame to head-relative frame.
        
        Args:
            world_coords: Coordinates in world frame
            head_pos: Head position in world frame
            head_quat: Head orientation quaternion [w, x, y, z]
            
        Returns:
            Coordinates in head-relative frame
        """
        # Translate to head-centered frame
        local_vec = world_coords - head_pos
        
        # Rotate vector using inverse quaternion
        # MuJoCo quaternions are [w, x, y, z] but we need [x, y, z, w] for our function
        q = np.array([head_quat[1], head_quat[2], head_quat[3], head_quat[0]])
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])  # Inverse quaternion
        
        # Rotate vector with inverse quaternion
        head_relative = quat_rotate(q_inv, local_vec)
        
        return head_relative

    def headless_simulation_loop(self, robot_state: RobotState, recording_state: RecordingState, 
                                target_fps: int = 30, duration: float = 5.0) -> None:
        """Run a headless simulation to reach the current targets."""
        print(f"Running headless simulation for {duration} seconds...")
        
        # Calculate number of steps
        dt = robot_state.mj_model.opt.timestep
        steps_per_frame = max(1, int(1.0 / (dt * target_fps)))
        total_steps = int(duration / dt)
        
        # Solve IK for both hands to get target joint angles
        ik_solver = IKSolver(robot_state=robot_state, recording_state=recording_state)
        
        # Get initial joint angles
        initial_angles = robot_state.get_joint_angles().copy()
        
        # Print initial hand positions and targets for debugging
        left_initial = robot_state.get_hand_position("left")
        right_initial = robot_state.get_hand_position("right")
        print(f"Initial left hand position: {left_initial}")
        print(f"Target left hand position: {robot_state.target_left_pos}")
        print(f"Initial right hand position: {right_initial}")
        print(f"Target right hand position: {robot_state.target_right_pos}")
        
        # Get the joint indices for both arms
        left_joint_indices = robot_state.left_arm_joint_indices
        right_joint_indices = robot_state.right_arm_joint_indices
        
        print(f"Using {len(left_joint_indices)} left arm joints and {len(right_joint_indices)} right arm joints")
        
        # Solve IK for left hand
        print("Solving IK for left hand...")
        left_target_angles = ik_solver.solve_ik("left", robot_state.target_left_pos)
        
        # Reset to initial state before solving for right hand
        robot_state.set_joint_angles(initial_angles)
        
        # Solve IK for right hand
        print("Solving IK for right hand...")
        right_target_angles = ik_solver.solve_ik("right", robot_state.target_right_pos)
        
        # Reset to initial state
        robot_state.set_joint_angles(initial_angles)
        
        # PD control parameters
        kp = 20.0  # Position gain
        kd = 2.0   # Velocity damping
        
        # Start recording if requested
        if recording_state is not None:
            recording_state.start_recording()
        
        try:
            # Run simulation
            for step in range(total_steps):
                # Get current joint positions and velocities
                current_angles = robot_state.mj_data.qpos.copy()
                current_vels = robot_state.mj_data.qvel.copy()
                
                # Apply control forces using PD control
                for i, idx in enumerate(left_joint_indices):
                    if idx < len(robot_state.mj_data.ctrl):
                        # Calculate error and derivative terms
                        pos_error = left_target_angles[idx] - current_angles[idx]
                        vel_error = -current_vels[idx]  # Damping term
                        
                        # PD control law
                        control_force = kp * pos_error + kd * vel_error
                        
                        # Apply control force
                        robot_state.mj_data.ctrl[idx] = control_force
                
                for i, idx in enumerate(right_joint_indices):
                    if idx < len(robot_state.mj_data.ctrl):
                        # Calculate error and derivative terms
                        pos_error = right_target_angles[idx] - current_angles[idx]
                        vel_error = -current_vels[idx]  # Damping term
                        
                        # PD control law
                        control_force = kp * pos_error + kd * vel_error
                        
                        # Apply control force
                        robot_state.mj_data.ctrl[idx] = control_force
                
                # Step the simulation forward
                mujoco.mj_step(robot_state.mj_model, robot_state.mj_data)
                
                # Capture frame if it's time and we're recording
                if step % steps_per_frame == 0 and recording_state is not None and recording_state.is_recording:
                    try:
                        # Get current hand positions
                        left_pos = robot_state.get_hand_position("left")
                        right_pos = robot_state.get_hand_position("right")
                        
                        # Create markers for visualization
                        markers = [
                            # Left target marker
                            {
                                "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                                "size": np.array([0.03, 0.03, 0.03]),
                                "pos": robot_state.target_left_pos,
                                "rgba": np.array([1.0, 0.0, 0.0, 0.7]),
                                "label": "Left Target"
                            },
                            # Right target marker
                            {
                                "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                                "size": np.array([0.03, 0.03, 0.03]),
                                "pos": robot_state.target_right_pos,
                                "rgba": np.array([0.0, 0.0, 1.0, 0.7]),
                                "label": "Right Target"
                            },
                            # Left hand marker
                            {
                                "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                                "size": np.array([0.03, 0.03, 0.03]),
                                "pos": left_pos,
                                "rgba": np.array([1.0, 0.5, 0.5, 0.7]),
                                "label": "Left Hand"
                            },
                            # Right hand marker
                            {
                                "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                                "size": np.array([0.03, 0.03, 0.03]),
                                "pos": right_pos,
                                "rgba": np.array([0.5, 0.5, 1.0, 0.7]),
                                "label": "Right Hand"
                            }
                        ]
                        
                        # Capture frame
                        frame = capture_mujoco_frame(
                            model=robot_state.mj_model,
                            data=robot_state.mj_data,
                            width=recording_state.width,
                            height=recording_state.height,
                            text_overlay=[
                                (f"Simulation Step: {step}/{total_steps}", 
                                 (10, 30), 0.7, (255, 255, 255), 2),
                                (f"Left hand distance: {np.linalg.norm(left_pos - robot_state.target_left_pos):.3f}", 
                                 (10, 60), 0.7, (255, 255, 255), 2),
                                (f"Right hand distance: {np.linalg.norm(right_pos - robot_state.target_right_pos):.3f}", 
                                 (10, 90), 0.7, (255, 255, 255), 2)
                            ],
                            markers=markers
                        )
                        
                        # Add the frame to the recording
                        recording_state.frames.append(frame.copy())
                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                
                # Print progress occasionally
                if step % 100 == 0 or step == total_steps - 1:
                    progress = step / total_steps * 100
                    left_pos = robot_state.get_hand_position("left")
                    right_pos = robot_state.get_hand_position("right")
                    left_dist = np.linalg.norm(left_pos - robot_state.target_left_pos)
                    right_dist = np.linalg.norm(right_pos - robot_state.target_right_pos)
                    print(f"Simulation progress: {progress:.1f}%")
                    print(f"Left hand distance to target: {left_dist:.3f}")
                    print(f"Right hand distance to target: {right_dist:.3f}")
                    
                    # Print joint angles for debugging
                    if step == 0 or step == total_steps - 1:
                        print("Joint angles:")
                        for i, idx in enumerate(left_joint_indices):
                            name = [n for n, j in robot_state.joint_indices.items() if j == idx][0]
                            print(f"  {name}: {np.degrees(robot_state.mj_data.qpos[idx]):.2f}°")
            
            print(f"Simulation complete.")
            
            # Stop recording if we started it
            if recording_state is not None and recording_state.is_recording:
                recording_state.stop_recording()
            
        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()

def main() -> None:
    """Main function to run the teleoperation interface."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Teleoperate a robot in MuJoCo with inverse kinematics")
    parser.add_argument("--task", type=str, help="Name of the task to run (loads config from envs/{task}.yaml)")
    parser.add_argument("--model", type=str, help="Path to MuJoCo XML model file (overrides config)")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for position adjustments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode with terminal input")
    parser.add_argument("--output", type=str, default="robot_episode.webm", help="Output video file name")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for recording")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of simulation in seconds")
    args = parser.parse_args()
    
    # Load from task name (similar to play_mujoco.py)
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
    except Exception as e:
        raise e
    
    # Override model path if specified
    if args.model:
        config["asset"]["mujoco_serial_file"] = args.model
    
    # Initialize robot from config
    model, data, joint_config = initialize_robot_from_config(config)
    
    # Initialize robot state
    robot_state = initialize_robot_state(model, data, joint_config)
    
    # Create recording state
    recording_state = RecordingState(
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height
    )
    
    # Override recording settings from config if available
    if "recording" in config:
        if "fps" in config["recording"]:
            recording_state.fps = config["recording"]["fps"]
        if "width" in config["recording"]:
            recording_state.width = config["recording"]["width"]
        if "height" in config["recording"]:
            recording_state.height = config["recording"]["height"]
        if "output" in config["recording"]:
            recording_state.output_path = config["recording"]["output"]
    
    # Create IK solver with recording state
    ik_solver = IKSolver(robot_state=robot_state, recording_state=recording_state)
    
    # Add these lines for debugging
    print("\n=== DEBUGGING ROBOT MODEL ===")
    ik_solver.ensure_hand_sites_exist()
    ik_solver.visualize_robot_structure()
    ik_solver.verify_joint_limits()
    print("============================\n")
    
    if args.headless:
        # Create headless interface
        interface = HeadlessInterface(
            robot_state=robot_state,
            ik_solver=ik_solver,
            recording_state=recording_state
        )
        
        # Run the interface
        interface.run()
    else:
        # Initialize pygame for keyboard input
        pygame.init()
        pygame.display.init()
        # Create a small window for keyboard focus
        pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Robot Teleoperation (Keyboard Focus)")
        
        # Create user interface
        ui = UserInterface(robot_state=robot_state, ik_solver=ik_solver, step_size=args.step_size)
        
        # Print instructions
        print("\nTeleoperation Controls:")
        print("  TAB: Switch between left and right hand control")
        print("  B: Toggle dual hand control (solve IK for both hands)")
        print("  SPACE: Toggle inverse kinematics mode")
        print("  F: Toggle freeze lower body and torso")
        print("  W/S: Move target forward/backward (+/- X)")
        print("  A/D: Move target left/right (+/- Y)")
        print("  E/C: Move target up/down (+/- Z)")
        print("  R: Reset target to current hand position")
        print("  Q: Quit")
        
        # Main loop with MuJoCo viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set initial camera position
            setup_viewer(viewer, config)
            
            running = True
            while running:
                # Process keyboard input
                running = ui.process_keyboard_input()
                if not running:
                    break
                
                # Update robot position using IK
                ui.update_robot()
                
                # Render target markers
                ui.render_targets(viewer)
                
                # Update the viewer
                viewer.sync()
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
        
        # Clean up
        pygame.quit()
        print("\nTeleoperation ended.")


if __name__ == "__main__":
    main() 
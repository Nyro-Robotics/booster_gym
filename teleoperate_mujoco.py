"""Robot teleoperation tool for MuJoCo environments with inverse kinematics.

This module provides functionality for:
- Loading and controlling a robot in MuJoCo
- Specifying target positions for the robot's hands
- Solving inverse kinematics to move the hands to target positions
- Interactive visualization and control
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

# -- Attributes --

@attrs.define
class RobotState:
    """Class to hold the current state of the robot."""
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    joint_limits: Dict[str, Dict[str, float]]
    joint_indices: Dict[str, int]
    left_hand_body_id: int
    right_hand_body_id: int
    target_left_pos: Optional[np.ndarray] = None
    target_right_pos: Optional[np.ndarray] = None
    current_mode: str = "left"  # 'left' or 'right'
    ik_active: bool = False
    control_both_hands: bool = False
    freeze_lower_body: bool = True  # Default to freezing lower body
    lower_body_joints: List[int] = attrs.field(factory=list)
    torso_joints: List[int] = attrs.field(factory=list)
    left_arm_joints: List[int] = attrs.field(factory=list)
    right_arm_joints: List[int] = attrs.field(factory=list)
    
    def __attrs_post_init__(self):
        # Initialize target positions to current hand positions
        self.target_left_pos = self.get_hand_position("left")
        self.target_right_pos = self.get_hand_position("right")
        
        # Create lists of joint indices for different body parts
        self.lower_body_joints = []
        self.torso_joints = []
        self.left_arm_joints = []
        self.right_arm_joints = []
        
        # Categorize joints based on name patterns
        for name, idx in self.joint_indices.items():
            name_lower = name.lower()
            if any(part in name_lower for part in ["hip", "knee", "ankle", "foot"]):
                self.lower_body_joints.append(idx)
            elif any(part in name_lower for part in ["torso", "spine", "waist"]):
                self.torso_joints.append(idx)
            elif "left" in name_lower and any(part in name_lower for part in ["shoulder", "elbow", "wrist"]):
                self.left_arm_joints.append(idx)
            elif "right" in name_lower and any(part in name_lower for part in ["shoulder", "elbow", "wrist"]):
                self.right_arm_joints.append(idx)

    def get_hand_position(self, hand: str) -> np.ndarray:
        """Get the current position of the specified hand.
        
        Args:
            hand: Either 'left' or 'right'
            
        Returns:
            3D position of the hand
        """
        body_id = self.left_hand_body_id if hand == "left" else self.right_hand_body_id
        return self.mj_data.site_xpos[body_id].copy()
    
    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles.
        
        Returns:
            Array of joint angles
        """
        return self.mj_data.qpos.copy()
    
    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set joint angles and update the simulation.
        
        Args:
            angles: Array of joint angles to set
        """
        self.mj_data.qpos[:] = angles
        mujoco.mj_forward(self.mj_model, self.mj_data)


@attrs.define
class IKSolver:
    """Inverse Kinematics solver with smoothing and body part freezing."""
    robot_state: RobotState
    # Control parameters for smoothing
    smoothing_factor: float = 0.1  # Smoothing factor for joint angle changes
    
    def solve_ik(self, hand: str, target_pos: np.ndarray, max_iterations: int = 500) -> np.ndarray:
        """Solve inverse kinematics to reach target position with smoothing.
        
        Args:
            hand: Either 'left' or 'right'
            target_pos: Target 3D position
            max_iterations: Maximum optimization iterations (increased)
            
        Returns:
            Optimized joint angles
        """
        # Find joints that match the patterns
        joint_indices = []
        valid_joint_names = []
        
        # Joints we are utilizing for IK for each hand
        if hand == "left":
            standard_names = ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", 
                             "Left_Shoulder_Yaw", "Left_Elbow"]
        else:
            standard_names = ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", 
                             "Right_Shoulder_Yaw", "Right_Elbow"]
        
        for name in standard_names:
            if name in self.robot_state.joint_indices:
                joint_indices.append(self.robot_state.joint_indices[name])
                valid_joint_names.append(name)
        
        # If no valid joints found, print error and return
        if not joint_indices:
            raise ValueError(f"No valid joints found for {hand} hand. Check joint names in your model.")
        
        # Print which joints we're controlling (for debugging)
        print(f"Controlling {hand} hand using joints: {valid_joint_names}")
        
        # Create a backup of the current state
        backup_qpos = self.robot_state.mj_data.qpos.copy()
        backup_qvel = self.robot_state.mj_data.qvel.copy()
        
        # Get current hand position
        current_pos = self.robot_state.get_hand_position(hand)
        print(f"Current {hand} hand position: {current_pos}")
        print(f"Target {hand} hand position: {target_pos}")
        print(f"Distance to target: {np.linalg.norm(current_pos - target_pos)}")
        
        # Try a grid search approach for better initial values
        best_distance = float('inf')
        best_angles = None
        
        # Define the objective function for optimization
        def objective(x):
            # Restore from backup to ensure clean state for each evaluation
            self.robot_state.mj_data.qpos[:] = backup_qpos
            self.robot_state.mj_data.qvel[:] = backup_qvel
            mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
            
            # Create a copy of the current joint angles
            test_angles = self.robot_state.mj_data.qpos.copy()
            
            # Update ONLY the relevant joint angles
            for i, idx in enumerate(joint_indices):
                test_angles[idx] = x[i]
            
            # Set the joint angles directly for evaluation
            self.robot_state.mj_data.qpos[:] = test_angles
            mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
            
            # Get the resulting hand position
            body_id = self.robot_state.left_hand_body_id if hand == "left" else self.robot_state.right_hand_body_id
            current_pos = self.robot_state.mj_data.site_xpos[body_id]
            
            # Calculate the distance to the target
            distance = np.linalg.norm(current_pos - target_pos)
            
            return distance
        
        # Try some random initializations to avoid local minima
        print("Trying multiple initializations to find best starting point...")
        num_tries = 10
        for _ in range(num_tries):
            # Generate random angles within bounds
            random_angles = []
            for name in valid_joint_names:
                limits = self.robot_state.joint_limits[name]
                min_rad = np.radians(limits["min"])
                max_rad = np.radians(limits["max"])
                random_angles.append(np.random.uniform(min_rad, max_rad))
            
            # Evaluate this configuration
            distance = objective(random_angles)
            
            if distance < best_distance:
                best_distance = distance
                best_angles = random_angles.copy()
        
        print(f"Best initialization found with distance: {best_distance}")
        
        # Extract initial values for the joints we're optimizing
        x0 = best_angles if best_angles is not None else np.array([backup_qpos[idx] for idx in joint_indices])
        
        # Define bounds based on joint limits
        bounds = []
        for name in valid_joint_names:
            limits = self.robot_state.joint_limits[name]
            # Convert degrees to radians
            min_rad = np.radians(limits["min"])
            max_rad = np.radians(limits["max"])
            bounds.append((min_rad, max_rad))
        
        # Run the optimization with more iterations and different method
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',  # Try a different optimization method
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        # If not successful, try again with a different method
        if not result.success or result.fun > 0.1:
            print("First optimization attempt not satisfactory, trying again with different method...")
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': 1e-6}
            )
        
        # Check if optimization was successful
        if result.success:
            print(f"IK optimization successful. Final distance: {result.fun}")
        else:
            print(f"IK optimization failed: {result.message}")
        
        # Restore from backup to ensure clean state
        self.robot_state.mj_data.qpos[:] = backup_qpos
        self.robot_state.mj_data.qvel[:] = backup_qvel
        mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
        
        # Get the optimized joint angles
        optimized_angles = self.robot_state.mj_data.qpos.copy()
        
        # Print the joint changes for debugging
        print(f"Joint angle changes for {hand} hand:")
        for i, idx in enumerate(joint_indices):
            current = optimized_angles[idx]
            target = result.x[i]
            change = target - current
            print(f"  {valid_joint_names[i]}: {np.degrees(current):.2f}° → {np.degrees(target):.2f}° (change: {np.degrees(change):.2f}°)")
            
            # Apply the change directly without smoothing
            optimized_angles[idx] = target
        
        # Apply the optimized angles directly using position control
        self._apply_direct_position_control(optimized_angles, joint_indices)
        
        # Verify the new hand position
        new_pos = self.robot_state.get_hand_position(hand)
        new_dist = np.linalg.norm(new_pos - target_pos)
        print(f"After IK: {hand} hand position: {new_pos}")
        print(f"After IK: Distance to target: {new_dist}")
        
        # If the distance is still large, try a brute force approach
        if new_dist > 0.1:
            print("IK solution not satisfactory, trying brute force approach...")
            self._try_brute_force_ik(hand, target_pos, joint_indices, valid_joint_names)
        
        return optimized_angles
    
    def _try_brute_force_ik(self, hand: str, target_pos: np.ndarray, joint_indices: List[int], joint_names: List[str]) -> None:
        """Try a brute force approach to IK by testing many joint configurations.
        
        Args:
            hand: Either 'left' or 'right'
            target_pos: Target 3D position
            joint_indices: Indices of joints to control
            joint_names: Names of joints to control
        """
        # Create a backup of the current state
        backup_qpos = self.robot_state.mj_data.qpos.copy()
        backup_qvel = self.robot_state.mj_data.qvel.copy()
        
        # Get current hand position
        current_pos = self.robot_state.get_hand_position(hand)
        initial_dist = np.linalg.norm(current_pos - target_pos)
        
        # Number of steps for each joint
        steps = 5
        
        # Best configuration found
        best_dist = initial_dist
        best_angles = backup_qpos.copy()
        
        # Try different combinations of joint angles
        print(f"Trying {steps**len(joint_indices)} configurations...")
        
        # For each joint, try different angles
        for i, idx in enumerate(joint_indices):
            name = joint_names[i]
            limits = self.robot_state.joint_limits[name]
            min_rad = np.radians(limits["min"])
            max_rad = np.radians(limits["max"])
            
            # Try different angles for this joint
            for angle_pct in np.linspace(0, 1, steps):
                angle = min_rad + angle_pct * (max_rad - min_rad)
                
                # Set the joint angle
                test_angles = backup_qpos.copy()
                test_angles[idx] = angle
                
                # Apply the angles
                self.robot_state.mj_data.qpos[:] = test_angles
                mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
                
                # Check the hand position
                hand_pos = self.robot_state.get_hand_position(hand)
                dist = np.linalg.norm(hand_pos - target_pos)
                
                # If better, save it
                if dist < best_dist:
                    best_dist = dist
                    best_angles = test_angles.copy()
                    print(f"Found better configuration: {dist:.3f} (joint {name} at {np.degrees(angle):.2f}°)")
        
        # Apply the best configuration found
        self.robot_state.mj_data.qpos[:] = best_angles
        mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
        
        # Print the final distance
        final_pos = self.robot_state.get_hand_position(hand)
        final_dist = np.linalg.norm(final_pos - target_pos)
        print(f"Brute force approach: initial distance {initial_dist:.3f}, final distance {final_dist:.3f}")
    
    def _apply_direct_position_control(self, target_angles: np.ndarray, joint_indices: List[int]) -> None:
        """Apply direct position control by setting joint angles.
        
        Args:
            target_angles: Target joint angles
            joint_indices: Indices of joints to control
        """
        # Get current joint positions
        current_angles = self.robot_state.mj_data.qpos.copy()
        
        # Update only the specified joints, leaving others unchanged
        new_angles = current_angles.copy()
        for idx in joint_indices:
            new_angles[idx] = target_angles[idx]
        
        # Set the joint angles
        self.robot_state.mj_data.qpos[:] = new_angles
        
        # Zero out velocities for stability
        self.robot_state.mj_data.qvel[:] = 0.0
        
        # Forward the simulation to update positions
        mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)


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
class RecordingState:
    """Class to hold the state of video recording."""
    recording: bool = False
    frames: List[np.ndarray] = attrs.Factory(list)
    output_path: str = "robot_episode.webm"  # Changed to WebM format
    fps: int = 30
    width: int = 640
    height: int = 480
    
    def __attrs_post_init__(self):
        # Ensure width and height are reasonable
        if self.width < 320:
            self.width = 320
        if self.height < 240:
            self.height = 240
        
        # Ensure output path has .webm extension
        if not self.output_path.endswith('.webm'):
            base_name = os.path.splitext(self.output_path)[0]
            self.output_path = f"{base_name}.webm"
    
    def start_recording(self) -> None:
        """Start recording frames."""
        self.recording = True
        self.frames = []
        print(f"Recording started. Frames will be stored for video creation.")
    
    def stop_recording(self) -> None:
        """Stop recording and save the video."""
        if not self.recording:
            print("Not recording.")
            return
        
        self.recording = False
        if not self.frames:
            print("No frames recorded.")
            return
        
        try:
            import cv2
            # Use VP8 codec for WebM format
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            
            # Get dimensions from the first frame
            if self.frames:
                h, w = self.frames[0].shape[:2]
                if h != self.height or w != self.width:
                    print(f"Adjusting video dimensions to match frames: {w}x{h}")
                    self.width, self.height = w, h
            
            video_writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                self.fps, 
                (self.width, self.height)
            )
            
            if not video_writer.isOpened():
                print(f"Error: Could not create video writer for {self.output_path}")
                # Try with a different codec as fallback
                print("Trying with MP4V codec instead...")
                fallback_path = os.path.splitext(self.output_path)[0] + ".mp4"
                video_writer = cv2.VideoWriter(
                    fallback_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (self.width, self.height)
                )
                
                if not video_writer.isOpened():
                    print(f"Error: Could not create video writer with fallback codec")
                    return
                else:
                    self.output_path = fallback_path
            
            print(f"Saving {len(self.frames)} frames to {self.output_path}...")
            for frame in self.frames:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        self.frames = []


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
        headless_simulation_loop(
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
        print("  solve - Solve IK for current targets")
        print("  solve_left - Solve IK for left hand only")
        print("  solve_right - Solve IK for right hand only")
        print("  solve_both - Solve IK for both hands")
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
                        duration=5.0  # Default duration
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
                
                elif parts[0] == "solve":
                    # Solve IK for the active hand
                    print(f"Solving IK for {self.robot_state.current_mode} hand...")
                    if self.robot_state.current_mode == "left":
                        self.ik_solver.solve_ik("left", self.robot_state.target_left_pos)
                    else:
                        self.ik_solver.solve_ik("right", self.robot_state.target_right_pos)
                    self._print_status()
                
                elif parts[0] == "solve_left":
                    # Solve IK for left hand only
                    print("Solving IK for left hand...")
                    self.ik_solver.solve_ik("left", self.robot_state.target_left_pos)
                    self._print_status()
                
                elif parts[0] == "solve_right":
                    # Solve IK for right hand only
                    print("Solving IK for right hand...")
                    self.ik_solver.solve_ik("right", self.robot_state.target_right_pos)
                    self._print_status()
                
                elif parts[0] == "solve_both":
                    # Solve IK for both hands
                    print("Solving IK for both hands...")
                    self.ik_solver.solve_ik("left", self.robot_state.target_left_pos)
                    self.ik_solver.solve_ik("right", self.robot_state.target_right_pos)
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


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion in xyzw format
        v: Vector to rotate
        
    Returns:
        Rotated vector
    """
    q_w = q[3]
    q_vec = q[:3]
    
    t = 2.0 * np.cross(q_vec, v)
    return v + q_w * t + np.cross(q_vec, t)


def render_offscreen(model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int,
                    left_target: np.ndarray, right_target: np.ndarray,
                    left_hand_id: int, right_hand_id: int) -> np.ndarray:
    """Render a frame offscreen with target markers using mjr functions."""

    # Initialize GLFW to create an OpenGL context
    if not glfw.init():
        print("Could not initialize GLFW")
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Create a hidden window for the OpenGL context
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "Offscreen Rendering", None, None)
    if not window:
        glfw.terminate()
        print("Could not create GLFW window")
        return np.zeros((height, width, 3), dtype=np.uint8)
    glfw.make_context_current(window)
    glfw.swap_interval(0) # Vsync off

    context = None # Initialize context to None for cleanup
    try:
        # Create MuJoCo rendering context, scene, camera, options
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        scene = mujoco.MjvScene(model, maxgeom=1000) # Increase maxgeom if needed
        camera = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb() # Needed for mjv_updateScene

        # Configure camera (adjust as needed)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.5
        camera.azimuth = 0
        camera.elevation = -20
        camera.lookat[:] = np.mean(data.xpos, axis=0) # Look at center of model

        # Update scene data
        mujoco.mjv_updateScene(
            model, data, opt, pert, camera,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )

        # --- Add Markers Directly to Scene ---
        # Get current hand positions
        left_pos = data.site_xpos[left_hand_id].copy()
        right_pos = data.site_xpos[right_hand_id].copy()

        markers_to_add = [
            {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.03]*3, "pos": left_target, "rgba": [1, 0, 0, 0.7]},
            {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.03]*3, "pos": right_target, "rgba": [0, 0, 1, 0.7]},
            {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.05]*3, "pos": left_pos, "rgba": [1, 0.5, 0.5, 0.7]},
            {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.05]*3, "pos": right_pos, "rgba": [0.5, 0.5, 1, 0.7]},
        ]

        for marker in markers_to_add:
            if scene.ngeom >= scene.maxgeom:
                print("Warning: Max geoms reached, skipping marker.")
                break
            g = scene.geoms[scene.ngeom]
            # Use mjv_initGeom for modern MuJoCo versions if available, otherwise set manually
            if hasattr(mujoco, 'mjv_initGeom'):
                 mujoco.mjv_initGeom(
                     g,
                     marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE),
                     marker.get("size", np.array([0.01, 0.01, 0.01])),
                     marker.get("pos", np.array([0.0, 0.0, 0.0])),
                     marker.get("mat", np.eye(3).flatten()),
                     marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0])),
                 )
                 # Set label separately if needed (mjv_initGeom doesn't handle it)
                 g.label = marker.get("label", "")
            else: # Fallback for older MuJoCo versions or manual setting
                g.type = marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE)
                g.size[:] = marker.get("size", np.array([0.01, 0.01, 0.01]))
                g.pos[:] = marker.get("pos", np.array([0.0, 0.0, 0.0]))
                g.mat[:] = marker.get("mat", np.eye(3).flatten())
                g.rgba[:] = marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0]))
                g.label = marker.get("label", "")
                # Set default values as in ref2.py's legacy function
                g.dataid = -1
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                g.texid = -1
                g.texuniform = 0
                g.texrepeat[0] = 1
                g.texrepeat[1] = 1
                g.emission = 0
                g.specular = 0.5
                g.shininess = 0.5
                g.reflectance = 0

            scene.ngeom += 1
        # --- End Marker Addition ---

        # Create viewport
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Render scene to offscreen buffer
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        mujoco.mjr_render(viewport, scene, context)

        # Read pixels
        rgb_arr = np.zeros(3 * viewport.width * viewport.height, dtype=np.uint8)
        depth_arr = np.zeros(viewport.width * viewport.height, dtype=np.float32) # Need depth buffer for readPixels
        mujoco.mjr_readPixels(rgb_arr, depth_arr, viewport, context)
        img = rgb_arr.reshape((height, width, 3))
        img = np.flipud(img) # Flip vertically

        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Add text overlays using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        left_dist = np.linalg.norm(left_pos - left_target)
        right_dist = np.linalg.norm(right_pos - right_target)
        cv2.putText(img_bgr, f"Left distance: {left_dist:.3f}",
                   (10, 20), font, font_scale, (0, 0, 255), thickness)
        cv2.putText(img_bgr, f"Right distance: {right_dist:.3f}",
                   (10, 40), font, font_scale, (255, 0, 0), thickness)

        return img_bgr

    except Exception as e:
        print(f"Error during offscreen rendering: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((height, width, 3), dtype=np.uint8)

    finally:
        # Clean up MuJoCo context and GLFW
        # The MjrContext (context) should be garbage collected.
        # We only need to terminate GLFW to release the OpenGL context.
        # if context:  # No longer needed
        #     mujoco.mjr_freeContext(context) # Remove this line
        glfw.terminate()

def headless_simulation_loop(robot_state: RobotState, recording_state: RecordingState, 
                            target_fps: int = 30, duration: float = 5.0) -> None:
    """Run a headless simulation to reach the current targets."""
    print(f"Running headless simulation for {duration} seconds...")
    
    # Calculate number of steps
    dt = robot_state.mj_model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (dt * target_fps)))
    total_steps = int(duration / dt)
    
    # Solve IK for both hands to get target joint angles
    ik_solver = IKSolver(robot_state=robot_state)
    
    # Get initial joint angles
    initial_angles = robot_state.get_joint_angles().copy()
    
    # Print initial hand positions and targets for debugging
    left_initial = robot_state.get_hand_position("left")
    right_initial = robot_state.get_hand_position("right")
    print(f"Initial left hand position: {left_initial}")
    print(f"Target left hand position: {robot_state.target_left_pos}")
    print(f"Initial right hand position: {right_initial}")
    print(f"Target right hand position: {robot_state.target_right_pos}")
    
    # Use the same joint indices as the IK solver for consistency
    left_joint_indices = []
    right_joint_indices = []
    
    # Get the same joints that the IK solver uses
    for name in ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Shoulder_Yaw", "Left_Elbow"]:
        if name in robot_state.joint_indices:
            left_joint_indices.append(robot_state.joint_indices[name])
    
    for name in ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Shoulder_Yaw", "Right_Elbow"]:
        if name in robot_state.joint_indices:
            right_joint_indices.append(robot_state.joint_indices[name])
    
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
    
    # Start recording
    if not recording_state.recording:
        print("Starting recording for this simulation run...")
        recording_state.start_recording()
    
    # Use a larger smoothing factor for faster movement in headless mode
    smoothing_factor = 0.1
    
    try:
        # Run simulation
        for step in range(total_steps):
            # Get current joint positions
            current_angles = robot_state.mj_data.qpos.copy()
            
            # Create new target angles by blending current and final target
            new_angles = current_angles.copy()
            
            # Update left arm joints
            for i, idx in enumerate(left_joint_indices):
                # Apply direct movement without smoothing for testing
                if step < total_steps / 2:  # Move more aggressively in first half
                    new_angles[idx] = left_target_angles[idx]
                else:
                    # Apply smoothing in second half
                    new_angles[idx] = current_angles[idx] + smoothing_factor * (left_target_angles[idx] - current_angles[idx])
            
            # Update right arm joints
            for i, idx in enumerate(right_joint_indices):
                # Apply direct movement without smoothing for testing
                if step < total_steps / 2:  # Move more aggressively in first half
                    new_angles[idx] = right_target_angles[idx]
                else:
                    # Apply smoothing in second half
                    new_angles[idx] = current_angles[idx] + smoothing_factor * (right_target_angles[idx] - current_angles[idx])
            
            # Set the joint angles
            robot_state.mj_data.qpos[:] = new_angles
            
            # Zero out velocities for stability
            robot_state.mj_data.qvel[:] = 0.0
            
            # Forward the simulation to update positions
            mujoco.mj_forward(robot_state.mj_model, robot_state.mj_data)
            
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
                print(f"Current left hand position: {left_pos}")
                print(f"Current right hand position: {right_pos}")
                
                # Print joint angles for debugging
                if step == 0 or step == total_steps - 1:
                    print("Joint angles:")
                    for i, idx in enumerate(left_joint_indices):
                        name = [n for n, j in robot_state.joint_indices.items() if j == idx][0]
                        print(f"  {name}: {np.degrees(new_angles[idx]):.2f}°")
            
            # Record frame if needed
            if recording_state.recording and step % steps_per_frame == 0:
                frame = render_offscreen(
                    robot_state.mj_model, 
                    robot_state.mj_data, 
                    recording_state.width, 
                    recording_state.height,
                    robot_state.target_left_pos,
                    robot_state.target_right_pos,
                    robot_state.left_hand_body_id,
                    robot_state.right_hand_body_id
                )
                recording_state.frames.append(frame)
        
        print(f"Simulation complete. Recorded {len(recording_state.frames)} frames.")
        
        # Stop recording and save video
        if recording_state.recording:
            recording_state.stop_recording()
            
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def initialize_robot_from_config(config: Dict[str, Any]) -> Tuple[mujoco.MjModel, mujoco.MjData, Dict[str, Any]]:
    """Initialize the robot model and data from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, data, joint_config)
    """
    # Load model
    model_path = config["asset"]["mujoco_serial_file"]
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Set timestep if specified
    if "sim" in config and "dt" in config["sim"]:
        model.opt.timestep = config["sim"]["dt"]
    
    # Create data
    data = mujoco.MjData(model)
    
    # Set initial pose if specified
    if "init_state" in config:
        if "pos" in config["init_state"]:
            data.qpos[0:3] = np.array(config["init_state"]["pos"], dtype=np.float32)
        
        if "rot" in config["init_state"]:
            # Convert to quaternion if needed
            rot = config["init_state"]["rot"]
            if len(rot) == 4:  # Already a quaternion
                data.qpos[3:7] = np.array(rot, dtype=np.float32)
            elif len(rot) == 3:  # Euler angles
                # Convert Euler angles to quaternion (assuming XYZ order)
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('xyz', rot, degrees=True)
                data.qpos[3:7] = r.as_quat()
    
    # Set default joint angles if specified
    joint_config = {}
    if "init_state" in config:
        # Copy all init_state settings to joint_config
        joint_config["init_state"] = config["init_state"]
        
        if "default_joint_angles" in config["init_state"]:
            joint_config["default_angles"] = config["init_state"]["default_joint_angles"]
        
        if "default_joint_limits" in config["init_state"]:
            joint_config["default_limits"] = config["init_state"]["default_joint_limits"]
            
            # Apply default joint angles
            if "default_angles" in joint_config:
                for i in range(model.nu):
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i + 7)  # Skip root joints
                    if joint_name in joint_config["default_angles"]:
                        data.qpos[i + 7] = joint_config["default_angles"][joint_name]
                    elif "default" in joint_config["default_angles"]:
                        data.qpos[i + 7] = joint_config["default_angles"]["default"]
    
    # Set PD gains if specified
    if "control" in config:
        if "stiffness" in config["control"]:
            joint_config["stiffness"] = config["control"]["stiffness"]
        
        if "damping" in config["control"]:
            joint_config["damping"] = config["control"]["damping"]
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    
    return model, data, joint_config


def initialize_robot_state(model: mujoco.MjModel, data: mujoco.MjData, 
                          joint_config: Dict[str, Any]) -> RobotState:
    """Initialize the robot state with joint limits and body IDs.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_config: Joint configuration from config file
        
    Returns:
        Initialized RobotState object
    """
    
    # Get joint limits from config
    joint_limits = joint_config["default_limits"]
    joint_indices = {}
    
    # Print all joints in the model for debugging
    print("\nJoints in the model:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"  {i}: {joint_name}")
            
            # Add all joints to the indices dictionary
            joint_indices[joint_name] = i
    
    # Find hand body IDs
    left_hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site") # nathanjzhao: custom added left hand site for geometry
    right_hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site") # nathanjzhao: custom added right hand site for geometry

    # Create the robot state
    robot_state = RobotState(
        mj_model=model,
        mj_data=data,
        joint_limits=joint_limits,
        joint_indices=joint_indices,
        left_hand_body_id=left_hand_body_id,
        right_hand_body_id=right_hand_body_id
    )
    
    # Categorize joints based on config file if available
    if "init_state" in joint_config:
        # Lower body joints
        if "lower_body_joints" in joint_config["init_state"]:
            for joint_id_str in joint_config["init_state"]["lower_body_joints"]:
                joint_id = int(joint_id_str)
                robot_state.lower_body_joints.append(joint_id)
            print(f"Loaded {len(robot_state.lower_body_joints)} lower body joints from config")
        
        # Torso joints
        if "torso_joints" in joint_config["init_state"]:
            for joint_id_str in joint_config["init_state"]["torso_joints"]:
                joint_id = int(joint_id_str)
                robot_state.torso_joints.append(joint_id)
            print(f"Loaded {len(robot_state.torso_joints)} torso joints from config")
    
    # If no joints were loaded from config, try to identify them by name
    if not robot_state.lower_body_joints and not robot_state.torso_joints:
        print("No joint categories found in config, identifying by name patterns...")
        for name, idx in joint_indices.items():
            name_lower = name.lower()
            if any(part in name_lower for part in ["hip", "knee", "ankle", "foot"]):
                robot_state.lower_body_joints.append(idx)
            elif any(part in name_lower for part in ["torso", "spine", "waist"]):
                robot_state.torso_joints.append(idx)
            elif "left" in name_lower and any(part in name_lower for part in ["shoulder", "elbow", "wrist"]):
                robot_state.left_arm_joints.append(idx)
            elif "right" in name_lower and any(part in name_lower for part in ["shoulder", "elbow", "wrist"]):
                robot_state.right_arm_joints.append(idx)
        
        print(f"Identified {len(robot_state.lower_body_joints)} lower body joints by name")
        print(f"Identified {len(robot_state.torso_joints)} torso joints by name")
    
    return robot_state


def setup_viewer(viewer, config: Dict[str, Any]) -> None:
    """Set up the viewer based on configuration.
    
    Args:
        viewer: MuJoCo viewer
        config: Configuration dictionary
    """
    if "viewer" in config:
        viewer_config = config["viewer"]
        
        # Set camera position
        if "distance" in viewer_config:
            viewer.cam.distance = viewer_config["distance"]
        else:
            viewer.cam.distance = 2.0
            
        if "azimuth" in viewer_config:
            viewer.cam.azimuth = viewer_config["azimuth"]
        else:
            viewer.cam.azimuth = 90
            
        if "elevation" in viewer_config:
            viewer.cam.elevation = viewer_config["elevation"]
        else:
            viewer.cam.elevation = -20
            
        # Set camera lookat if specified
        if "lookat" in viewer_config:
            viewer.cam.lookat[:] = viewer_config["lookat"]


def add_target_markers(model: mujoco.MjModel, data: mujoco.MjData, 
                      left_target: np.ndarray, right_target: np.ndarray) -> None:
    """Add visual markers for target positions.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        left_target: Target position for left hand
        right_target: Target position for right hand
    """
    # We'll use MuJoCo's built-in visualization tools to add markers
    # This requires modifying the scene data directly
    
    # Add a marker for the left hand target (red)
    mujoco.mjv_initGeom(
        model.vis.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.05, 0.05, 0.05]),  # Size
        left_target,  # Position
        np.array([1, 0, 0, 1]),  # Orientation (quaternion)
        np.array([1, 0, 0, 0.7])  # Color (RGBA) - red with some transparency
    )
    
    # Add a marker for the right hand target (blue)
    mujoco.mjv_initGeom(
        model.vis.geoms[1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.05, 0.05, 0.05]),  # Size
        right_target,  # Position
        np.array([1, 0, 0, 1]),  # Orientation (quaternion)
        np.array([0, 0, 1, 0.7])  # Color (RGBA) - blue with some transparency
    )


def main() -> None:
    """Main function to run the teleoperation interface."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Teleoperate a robot in MuJoCo with inverse kinematics")
    parser.add_argument("--task", type=str, help="Name of the task to run (loads config from envs/{task}.yaml)")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file (alternative to --task)")
    parser.add_argument("--model", type=str, help="Path to MuJoCo XML model file (overrides config)")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for position adjustments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode with terminal input")
    parser.add_argument("--output", type=str, default="robot_episode.webm", help="Output video file name")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for recording")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of simulation in seconds")
    args = parser.parse_args()
    
    # Ensure either task or config is provided
    if not args.task and not args.config:
        parser.error("Either --task or --config must be specified")
    
    # Load configuration
    if args.task:
        # Load from task name (similar to play_mujoco.py)
        cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
        except Exception as e:
            print(f"Error loading task config: {e}")
            sys.exit(1)
    else:
        # Load from specified config file
        config = load_config(args.config)
    
    # Override model path if specified
    if args.model:
        config["asset"]["mujoco_serial_file"] = args.model
    
    # Initialize robot from config
    model, data, joint_config = initialize_robot_from_config(config)
    
    # Initialize robot state
    robot_state = initialize_robot_state(model, data, joint_config)
    
    # Create IK solver
    ik_solver = IKSolver(robot_state=robot_state)
    
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
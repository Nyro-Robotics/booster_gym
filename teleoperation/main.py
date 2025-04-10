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

from teleoperation.utils import initialize_robot_from_config, initialize_robot_state, quat_rotate
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

    # deprecated
    def render_offscreen(self, model: mujoco.MjModel, data: mujoco.MjData, width: int, height: int,
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
            camera.azimuth = 180
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

    def headless_simulation_loop(self, robot_state: RobotState, recording_state: RecordingState, 
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
        for name in ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"]:
            if name in robot_state.joint_indices:
                left_joint_indices.append(robot_state.joint_indices[name])

        for name in ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]:
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
            
            print(f"Simulation complete.")
                
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
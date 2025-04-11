import numpy as np
import cv2
import mujoco
import attrs
from typing import List, Optional, Tuple, Callable, Dict
from teleoperation.teleop_types import RobotState
from teleoperation.record import RecordingState, capture_mujoco_frame

@attrs.define
class LevenbergMarquardtIK:
    """Inverse Kinematics solver using the Levenberg-Marquardt algorithm."""
    model: mujoco.MjModel
    data: mujoco.MjData
    step_size: float = 0.5
    tol: float = 0.01
    damping: float = 0.15
    max_iterations: int = 100
    recording_state: Optional[any] = None  # Recording state reference
    should_record: bool = False  # Simple flag to control recording
    early_stop_threshold: float = 1e-5  # Threshold for early stopping
    early_stop_patience: int = 10  # Number of iterations with minimal improvement before stopping
    
    def solve(self, goal_pos: np.ndarray, site_id: int, joint_indices: List[int]) -> np.ndarray:
        """Calculate the desired joint angles to reach the goal position.
        
        Args:
            goal_pos: Target 3D position
            site_id: ID of the site to control (end effector)
            joint_indices: Indices of joints to use for IK
            
        Returns:
            Optimized joint angles
        """
        # Create backup of current state
        backup_qpos = self.data.qpos.copy()
        backup_qvel = self.data.qvel.copy()
        
        # Get current position
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.site_xpos[site_id].copy()
        
        error = goal_pos - current_pos
        initial_error = np.linalg.norm(error)
        
        print(f"Initial position: {current_pos}")
        print(f"Goal position: {goal_pos}")
        print(f"Initial error: {initial_error}")
        print(f"Using {len(joint_indices)} joints for IK")
        
        # Initialize Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # translational jacobian
        jacr = np.zeros((3, self.model.nv))  # rotational jacobian
        
        # For physics-based IK, we'll use a PD controller approach
        kp = 10.0  # Position gain
        kd = 1.0   # Velocity damping
        
        iteration = 0
        best_error = initial_error
        best_qpos = self.data.qpos.copy()
        
        # Variables for early stopping
        stagnation_counter = 0
        previous_error = initial_error
        
        # Start recording if requested
        if self.should_record and self.recording_state is not None:
            self.recording_state.start_recording()
            print("Started recording IK convergence process")
        
        # Create a simulation copy for iterative solving
        sim_data = mujoco.MjData(self.model)
        
        # Copy data manually instead of using mj_copyData
        sim_data.qpos[:] = self.data.qpos[:]
        sim_data.qvel[:] = self.data.qvel[:]
        sim_data.act[:] = self.data.act[:]
        sim_data.ctrl[:] = self.data.ctrl[:]
        mujoco.mj_forward(self.model, sim_data)
        
        while np.linalg.norm(error) > self.tol and iteration < self.max_iterations:
            # Calculate Jacobian for the site
            mujoco.mj_jacSite(self.model, sim_data, jacp, jacr, site_id)
            
            # Extract only the columns for the joints we're using
            reduced_jacp = np.zeros((3, len(joint_indices)))
            for i, idx in enumerate(joint_indices):
                if idx < jacp.shape[1]:  # Safety check
                    reduced_jacp[:, i] = jacp[:, idx]
            
            # Calculate delta using Levenberg-Marquardt
            JTJ = reduced_jacp.T @ reduced_jacp
            lambda_I = self.damping * np.eye(JTJ.shape[0])
            JTe = reduced_jacp.T @ error
            
            # Solve (J^T J + λI) Δθ = J^T e
            try:
                delta_q_reduced = np.linalg.solve(JTJ + lambda_I, JTe)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if matrix is singular
                delta_q_reduced = np.linalg.pinv(JTJ + lambda_I) @ JTe
                print("Using pseudoinverse")
            
            # Apply forces to the joints instead of directly setting positions
            # This allows physics to be properly simulated
            for i, idx in enumerate(joint_indices):
                if i < len(delta_q_reduced) and idx < len(sim_data.ctrl):  # Safety check
                    # Calculate target position for this joint
                    target_pos = sim_data.qpos[idx] + self.step_size * delta_q_reduced[i]
                    
                    # Apply position control using a PD controller
                    current_pos = sim_data.qpos[idx]
                    current_vel = sim_data.qvel[idx]
                    
                    # Calculate force/torque using PD control
                    force = kp * (target_pos - current_pos) - kd * current_vel
                    
                    # Apply the force to the joint
                    sim_data.ctrl[idx] = force
            
            # Step the simulation forward to apply physics
            for _ in range(5):  # Take multiple small steps for stability
                mujoco.mj_step(self.model, sim_data)
            
            # Check joint limits
            self._check_joint_limits_physics(sim_data)
            
            # Forward kinematics
            mujoco.mj_forward(self.model, sim_data)
            
            # Calculate new error
            current_pos = sim_data.site_xpos[site_id].copy()
            error = goal_pos - current_pos
            current_error = np.linalg.norm(error)
            
            # Save best result
            if current_error < best_error:
                best_error = current_error
                # Copy data manually
                best_qpos = sim_data.qpos.copy()
                print(f"New best error: {current_error:.6f}")
            
            # Check for early stopping - if error isn't improving significantly
            error_improvement = previous_error - current_error
            if error_improvement < self.early_stop_threshold:
                stagnation_counter += 1
                if stagnation_counter >= self.early_stop_patience:
                    print(f"Early stopping at iteration {iteration}: Error not improving significantly")
                    print(f"Error improvement: {error_improvement:.8f}, below threshold: {self.early_stop_threshold}")
                    break
            else:
                # Reset counter if we're making good progress
                stagnation_counter = 0
            
            previous_error = current_error
            
            # Capture frame for recording if enabled
            if self.should_record and self.recording_state is not None:
                # Copy the simulation state to the main data for visualization
                # Copy data manually
                self.data.qpos[:] = sim_data.qpos[:]
                self.data.qvel[:] = sim_data.qvel[:]
                mujoco.mj_forward(self.model, self.data)
                
                try:
                    # Create markers for visualization
                    markers = [
                        # Target position marker (green sphere)
                        {
                            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                            "size": np.array([0.03, 0.03, 0.03]),
                            "pos": goal_pos,
                            "rgba": np.array([0.0, 1.0, 0.0, 0.7]),
                            "label": "Target"
                        },
                        # Current position marker (red sphere)
                        {
                            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                            "size": np.array([0.03, 0.03, 0.03]),
                            "pos": current_pos,
                            "rgba": np.array([1.0, 0.0, 0.0, 0.7]),
                            "label": "Current"
                        },
                        # Line connecting current to target
                        {
                            "type": mujoco.mjtGeom.mjGEOM_LINE,
                            "rgba": np.array([1.0, 1.0, 0.0, 0.5]),
                            "from_": current_pos,
                            "to": goal_pos
                        }
                    ]
                    
                    # Use our reusable function to capture a frame with markers
                    frame = capture_mujoco_frame(
                        model=self.model,
                        data=self.data,
                        width=self.recording_state.width,
                        height=self.recording_state.height,
                        text_overlay=[
                            (f"Iteration: {iteration}, Error: {current_error:.6f}", 
                             (10, 30), 0.7, (255, 255, 255), 2),
                            (f"Target position: {goal_pos.round(3)}", 
                             (10, 60), 0.7, (255, 255, 255), 2),
                            (f"Current position: {current_pos.round(3)}", 
                             (10, 90), 0.7, (255, 255, 255), 2)
                        ],
                        markers=markers
                    )
                    
                    # Add the frame to the recording
                    self.recording_state.frames.append(frame.copy())
                    
                    # Only log every few iterations to reduce console output
                    if iteration % 5 == 0:
                        print(f"Captured frame at iteration {iteration}")
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    import traceback
                    traceback.print_exc()
            
            iteration += 1
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}, error: {current_error:.6f}")
        
        # Use the best result found
        self.data.qpos[:] = best_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Final position and error
        final_pos = self.data.site_xpos[site_id].copy()
        final_error = np.linalg.norm(goal_pos - final_pos)
        
        print(f"Final position: {final_pos}")
        print(f"Final error: {final_error}")
        print(f"Iterations: {iteration}")
        
        # Capture one final frame with the best result
        if self.should_record and self.recording_state is not None:
            try:
                # Create markers for visualization
                markers = [
                    # Target position marker (green sphere)
                    {
                        "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                        "size": np.array([0.03, 0.03, 0.03]),
                        "pos": goal_pos,
                        "rgba": np.array([0.0, 1.0, 0.0, 0.7]),
                        "label": "Target"
                    },
                    # Final position marker (blue sphere)
                    {
                        "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                        "size": np.array([0.03, 0.03, 0.03]),
                        "pos": final_pos,
                        "rgba": np.array([0.0, 0.5, 1.0, 0.7]),
                        "label": "Final"
                    },
                    # Line connecting final to target
                    {
                        "type": mujoco.mjtGeom.mjGEOM_LINE,
                        "rgba": np.array([0.0, 1.0, 1.0, 0.5]),
                        "from_": final_pos,
                        "to": goal_pos
                    }
                ]
                
                # Use our reusable function to capture a frame
                frame = capture_mujoco_frame(
                    model=self.model,
                    data=self.data,
                    width=self.recording_state.width,
                    height=self.recording_state.height,
                    text_overlay=[
                        (f"FINAL - Error: {final_error:.6f}", 
                         (10, 30), 0.7, (0, 255, 0), 2),
                        (f"Target position: {goal_pos.round(3)}", 
                         (10, 60), 0.7, (255, 255, 255), 2),
                        (f"Final position: {final_pos.round(3)}", 
                         (10, 90), 0.7, (255, 255, 255), 2)
                    ],
                    markers=markers
                )
                
                # Add the final frame to the recording
                self.recording_state.frames.append(frame.copy())
            except Exception as e:
                print(f"Error capturing final frame: {e}")
        
        # Stop recording if we started it
        if self.should_record and self.recording_state is not None:
            self.recording_state.stop_recording()
            print("Stopped recording IK convergence process")
        
        return self.data.qpos.copy()
    
    def _check_joint_limits_physics(self, data: mujoco.MjData) -> None:
        """Ensure joint angles are within limits for physics-based simulation."""
        for i in range(self.model.nq):
            if i < len(self.model.jnt_range):
                lower = self.model.jnt_range[i][0]
                upper = self.model.jnt_range[i][1]
                
                # Apply soft limits by adding opposing forces when near limits
                if data.qpos[i] < lower + 0.1:
                    # Add force to push away from lower limit
                    force_magnitude = 10.0 * (lower + 0.1 - data.qpos[i])
                    if i < len(data.ctrl):
                        data.ctrl[i] += force_magnitude
                
                elif data.qpos[i] > upper - 0.1:
                    # Add force to push away from upper limit
                    force_magnitude = 10.0 * (upper - 0.1 - data.qpos[i])
                    if i < len(data.ctrl):
                        data.ctrl[i] += force_magnitude


@attrs.define
class IKSolver:
    """Inverse Kinematics solver with smoothing and body part freezing."""
    robot_state: RobotState
    # Control parameters for smoothing
    smoothing_factor: float = 0.1  # Smoothing factor for joint angle changes
    recording_state: Optional[any] = None  # Recording state reference
    
    def solve_ik(self, hand: str, target_pos: np.ndarray, max_iterations: int = 500, record_convergence: bool = False) -> np.ndarray:
        """Solve inverse kinematics to reach target position with smoothing.
        
        Args:
            hand: Either 'left' or 'right'
            target_pos: Target 3D position
            max_iterations: Maximum optimization iterations
            record_convergence: Whether to record the convergence process
            
        Returns:
            Optimized joint angles
        """
        # Get joint indices from robot_state
        if hand == "left":
            joint_indices = self.robot_state.left_arm_joint_indices
            valid_joint_names = self.robot_state.left_arm_joint_names
            site_name = "left_hand_site"  # Updated site name
        else:
            joint_indices = self.robot_state.right_arm_joint_indices
            valid_joint_names = self.robot_state.right_arm_joint_names
            site_name = "right_hand_site"  # Updated site name
        
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
        
        # Get the site ID for the hand
        site_id = mujoco.mj_name2id(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        # If site not found, create a warning and use body ID as fallback
        if site_id == -1:
            print(f"WARNING: Site '{site_name}' not found in the model. Using body ID as fallback.")
            print("For better IK performance, add a site to your model XML at the hand position.")
            
            # Use body ID as fallback
            site_id = self.robot_state.left_hand_body_id if hand == "left" else self.robot_state.right_hand_body_id
        else:
            print(f"Using site '{site_name}' (ID: {site_id}) for IK control")
        
        # Print joint limits for debugging
        print(f"\n--- {hand.upper()} ARM JOINT LIMITS ---")
        for i, (name, idx) in enumerate(zip(valid_joint_names, joint_indices)):
            if idx < len(self.robot_state.mj_model.jnt_range):
                lower = np.degrees(self.robot_state.mj_model.jnt_range[idx][0])
                upper = np.degrees(self.robot_state.mj_model.jnt_range[idx][1])
                print(f"{name}: [{lower:.2f}°, {upper:.2f}°]")
        print("----------------------------\n")
        
        # Create the solver with recording settings
        lm_solver = LevenbergMarquardtIK(
            model=self.robot_state.mj_model,
            data=self.robot_state.mj_data,
            step_size=0.5,
            tol=0.01,
            damping=0.15,
            max_iterations=max_iterations,
            recording_state=self.recording_state,
            should_record=record_convergence
        )
        
        if record_convergence:
            print(f"Will record {hand} hand IK convergence")
        
        # Debug visualization of joints before solving
        self.debug_visualize_joints(hand)
        
        # Solve IK with Levenberg-Marquardt
        optimized_angles = lm_solver.solve(target_pos, site_id, joint_indices)
        
        # Apply the optimized angles directly using position control
        self._apply_direct_position_control(optimized_angles, joint_indices)

        print("AFTER SOLVE")
        self.debug_visualize_joints(hand)
        
        # Verify the new hand position
        new_pos = self.robot_state.get_hand_position(hand)
        new_dist = np.linalg.norm(new_pos - target_pos)
        print(f"After IK: {hand} hand position: {new_pos}")
        print(f"After IK: Distance to target: {new_dist}")
        
        return optimized_angles
    
    def _apply_direct_position_control(self, target_angles: np.ndarray, joint_indices: List[int]) -> None:
        """Apply position control using physics simulation.
        
        Args:
            target_angles: Target joint angles
            joint_indices: Indices of joints to control
        """
        # Get current joint positions
        current_angles = self.robot_state.mj_data.qpos.copy()
        
        # PD control parameters
        kp = 50.0  # Position gain
        kd = 5.0   # Velocity damping
        
        # Number of simulation steps to take
        num_steps = 50
        
        # Create a temporary data object for simulation
        temp_data = mujoco.MjData(self.robot_state.mj_model)
        
        # Copy data manually
        temp_data.qpos[:] = self.robot_state.mj_data.qpos[:]
        temp_data.qvel[:] = self.robot_state.mj_data.qvel[:]
        temp_data.act[:] = self.robot_state.mj_data.act[:]
        temp_data.ctrl[:] = self.robot_state.mj_data.ctrl[:]
        mujoco.mj_forward(self.robot_state.mj_model, temp_data)
        
        # Run a short simulation to apply physics
        for step in range(num_steps):
            # Calculate control forces for each joint
            for idx in joint_indices:
                if idx < len(temp_data.ctrl):
                    # Calculate error and derivative terms
                    pos_error = target_angles[idx] - temp_data.qpos[idx]
                    vel_error = -temp_data.qvel[idx]  # Damping term
                    
                    # PD control law
                    control_force = kp * pos_error + kd * vel_error
                    
                    # Apply control force
                    temp_data.ctrl[idx] = control_force
            
            # Step the simulation forward
            mujoco.mj_step(self.robot_state.mj_model, temp_data)
            
            # Capture intermediate frames if recording
            if step % 5 == 0 and self.recording_state is not None and self.recording_state.recording:
                # Copy the temporary state to the main data for visualization
                self.robot_state.mj_data.qpos[:] = temp_data.qpos[:]
                self.robot_state.mj_data.qvel[:] = temp_data.qvel[:]
                mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)
                
                try:
                    # Get current hand positions
                    left_pos = self.robot_state.get_hand_position("left")
                    right_pos = self.robot_state.get_hand_position("right")
                    
                    # Create markers for visualization
                    markers = [
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
                        model=self.robot_state.mj_model,
                        data=self.robot_state.mj_data,
                        width=self.recording_state.width,
                        height=self.recording_state.height,
                        text_overlay=[
                            (f"Physics Step: {step}/{num_steps}", 
                             (10, 30), 0.7, (255, 255, 255), 2)
                        ],
                        markers=markers
                    )
                    
                    # Add the frame to the recording
                    self.recording_state.frames.append(frame.copy())
                except Exception as e:
                    print(f"Error capturing physics frame: {e}")
        
        # Copy the final state back to the main data
        self.robot_state.mj_data.qpos[:] = temp_data.qpos[:]
        self.robot_state.mj_data.qvel[:] = temp_data.qvel[:]
        
        # Forward the simulation to update positions
        mujoco.mj_forward(self.robot_state.mj_model, self.robot_state.mj_data)

    def debug_visualize_joints(self, hand: str) -> None:
        """Visualize the joints being used for IK to help debug."""
        joint_indices = self.robot_state.left_arm_joint_indices if hand == "left" else self.robot_state.right_arm_joint_indices
        joint_names = self.robot_state.left_arm_joint_names if hand == "left" else self.robot_state.right_arm_joint_names
        
        print(f"\n--- {hand.upper()} ARM JOINT DEBUG ---")
        print(f"Number of joints: {len(joint_indices)}")
        
        for i, (name, idx) in enumerate(zip(joint_names, joint_indices)):
            angle = self.robot_state.mj_data.qpos[idx]
            angle_deg = np.degrees(angle)
            
            # Get joint range if available
            if idx < len(self.robot_state.mj_model.jnt_range):
                lower = np.degrees(self.robot_state.mj_model.jnt_range[idx][0])
                upper = np.degrees(self.robot_state.mj_model.jnt_range[idx][1])
                
                # Calculate percentage of range used
                if upper > lower:
                    percentage = (angle_deg - lower) / (upper - lower) * 100
                    percentage = max(0, min(100, percentage))  # Clamp to 0-100%
                    
                    # Create a visual representation of where in the range the joint is
                    bar_width = 20
                    bar_pos = int(percentage / 100 * bar_width)
                    bar = '|' + '-' * bar_pos + 'O' + '-' * (bar_width - bar_pos - 1) + '|'
                    
                    range_info = f"Range: [{lower:.1f}°, {upper:.1f}°] {bar} {percentage:.1f}%"
                else:
                    range_info = f"Range: [{lower:.1f}°, {upper:.1f}°] (INVALID RANGE)"
            else:
                range_info = "Range: Unknown"
            
            print(f"{i+1}. {name}: {angle_deg:.2f}° ({range_info})")
        
        print("----------------------------\n")

    def verify_joint_limits(self) -> None:
        """Verify that joint limits are properly set in the model."""
        print("\n--- VERIFYING JOINT LIMITS ---")
        for i in range(self.robot_state.mj_model.njnt):
            name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if i < len(self.robot_state.mj_model.jnt_range):
                lower = np.degrees(self.robot_state.mj_model.jnt_range[i][0])
                upper = np.degrees(self.robot_state.mj_model.jnt_range[i][1])
                
                # Check if limits are reasonable
                if lower == upper:
                    print(f"WARNING: Joint {name} has equal lower and upper limits: {lower:.2f}°")
                elif lower > upper:
                    print(f"ERROR: Joint {name} has inverted limits: [{lower:.2f}°, {upper:.2f}°]")
                else:
                    range_size = upper - lower
                    if range_size < 1.0:
                        print(f"WARNING: Joint {name} has very small range: [{lower:.2f}°, {upper:.2f}°] (range: {range_size:.2f}°)")
                    else:
                        print(f"Joint {name}: [{lower:.2f}°, {upper:.2f}°] (range: {range_size:.2f}°)")
            else:
                print(f"WARNING: Joint {name} has no limits defined")
        print("----------------------------\n")

    def ensure_hand_sites_exist(self) -> None:
        """Ensure that hand sites exist in the model, creating them if needed."""
        # Check for left hand site
        left_site_id = mujoco.mj_name2id(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
        if left_site_id == -1:
            print("WARNING: 'left_hand_site' site not found in model.")
            print("For better IK performance, add a site to your model XML at the left hand position.")
            
        # Check for right hand site
        right_site_id = mujoco.mj_name2id(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        if right_site_id == -1:
            print("WARNING: 'right_hand_site' site not found in model.")
            print("For better IK performance, add a site to your model XML at the right hand position.")
        
        # Print information about all sites in the model
        print("\n--- SITES IN MODEL ---")
        for i in range(self.robot_state.mj_model.nsite):
            name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            pos = self.robot_state.mj_data.site_xpos[i]
            print(f"Site {i}: {name}, position: {pos}")
        print("---------------------\n")

    def visualize_robot_structure(self) -> None:
        """Visualize the robot's kinematic structure to help with debugging."""
        print("\n=== ROBOT STRUCTURE ===")
        
        # Print bodies
        print("\n--- BODIES ---")
        for i in range(self.robot_state.mj_model.nbody):
            name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            parent = self.robot_state.mj_model.body_parentid[i]
            parent_name = "WORLD" if parent == 0 else mujoco.mj_id2name(
                self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_BODY, parent)
            pos = self.robot_state.mj_data.xpos[i]
            print(f"Body {i}: {name}, parent: {parent_name}, position: {pos}")
        
        # Print joints
        print("\n--- JOINTS ---")
        for i in range(self.robot_state.mj_model.njnt):
            name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            type_id = self.robot_state.mj_model.jnt_type[i]
            type_name = ["FREE", "BALL", "SLIDE", "HINGE"][type_id]
            body_id = self.robot_state.mj_model.jnt_bodyid[i]
            body_name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"Joint {i}: {name}, type: {type_name}, body: {body_name}")
        
        # Print sites
        print("\n--- SITES ---")
        for i in range(self.robot_state.mj_model.nsite):
            name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            body_id = self.robot_state.mj_model.site_bodyid[i]
            body_name = mujoco.mj_id2name(self.robot_state.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            pos = self.robot_state.mj_data.site_xpos[i]
            print(f"Site {i}: {name}, body: {body_name}, position: {pos}")
        
        print("========================\n")

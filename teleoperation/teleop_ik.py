import numpy as np
import mujoco
import attrs
from typing import List
from teleoperation.teleop_types import RobotState

@attrs.define
class LevenbergMarquardtIK:
    """Inverse Kinematics solver using the Levenberg-Marquardt algorithm."""
    model: mujoco.MjModel
    data: mujoco.MjData
    step_size: float = 0.5
    tol: float = 0.01
    damping: float = 0.15
    max_iterations: int = 100
    
    def solve(self, goal_pos: np.ndarray, body_id: int, joint_indices: List[int]) -> np.ndarray:
        """Calculate the desired joint angles to reach the goal position.
        
        Args:
            goal_pos: Target 3D position
            body_id: ID of the body to control
            joint_indices: Indices of joints to use for IK
            
        Returns:
            Optimized joint angles
        """
        # Create backup of current state
        backup_qpos = self.data.qpos.copy()
        backup_qvel = self.data.qvel.copy()
        
        # Get current position
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.site_xpos[body_id].copy()
        error = goal_pos - current_pos
        initial_error = np.linalg.norm(error)
        
        print(f"Initial position: {current_pos}")
        print(f"Goal position: {goal_pos}")
        print(f"Initial error: {initial_error}")
        print(f"Using {len(joint_indices)} joints for IK")
        
        # Initialize Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # translational jacobian
        jacr = np.zeros((3, self.model.nv))  # rotational jacobian
        
        iteration = 0
        best_error = initial_error
        best_qpos = self.data.qpos.copy()
        
        while np.linalg.norm(error) > self.tol and iteration < self.max_iterations:
            # Calculate Jacobian - FIXED: Use site_id instead of passing position and body_id
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, body_id)
            
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
            
            # breakpoint()
            # Apply delta to only the joints we're controlling
            for i, idx in enumerate(joint_indices):
                if i < len(delta_q_reduced):  # Safety check
                    self.data.qpos[idx] += self.step_size * delta_q_reduced[i]
            
            # Check joint limits
            self._check_joint_limits()
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Calculate new error
            current_pos = self.data.site_xpos[body_id].copy()
            error = goal_pos - current_pos
            current_error = np.linalg.norm(error)
            
            # Save best result
            if current_error < best_error:
                best_error = current_error
                best_qpos = self.data.qpos.copy()
                print(f"New best error: {current_error:.6f}")
            
            iteration += 1
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}, error: {current_error:.6f}")
        
        # Use the best result found
        self.data.qpos[:] = best_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Final position and error
        final_pos = self.data.site_xpos[body_id].copy()
        final_error = np.linalg.norm(goal_pos - final_pos)
        
        print(f"Final position: {final_pos}")
        print(f"Final error: {final_error}")
        print(f"Iterations: {iteration}")
        
        return self.data.qpos.copy()
    
    def _check_joint_limits(self) -> None:
        """Ensure joint angles are within limits."""
        for i in range(self.model.nq):
            if i < len(self.model.jnt_range):
                lower = self.model.jnt_range[i][0]
                upper = self.model.jnt_range[i][1]
                self.data.qpos[i] = max(lower, min(self.data.qpos[i], upper))


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
        # Get joint indices from robot_state
        if hand == "left":
            joint_indices = self.robot_state.left_arm_joint_indices
            valid_joint_names = self.robot_state.left_arm_joint_names
        else:
            joint_indices = self.robot_state.right_arm_joint_indices
            valid_joint_names = self.robot_state.right_arm_joint_names
        
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
        
        # Use Levenberg-Marquardt solver
        body_id = self.robot_state.left_hand_body_id if hand == "left" else self.robot_state.right_hand_body_id
        
        lm_solver = LevenbergMarquardtIK(
            model=self.robot_state.mj_model,
            data=self.robot_state.mj_data,
            step_size=0.5,
            tol=0.01,
            damping=0.15,
            max_iterations=max_iterations
        )
        
        # Check if recording is active and capture frames during solving
        recording_active = False
        recording_state = None
        
        # Try to access recording state from global scope
        try:
            if 'recording_state' in globals():
                recording_state = globals()['recording_state']
                if hasattr(recording_state, 'recording') and recording_state.recording:
                    recording_active = True
                    print("Recording during IK solving...")
        except Exception as e:
            print(f"Warning: Could not access recording state: {e}")
            raise e
        
        # Solve IK with Levenberg-Marquardt
        optimized_angles = lm_solver.solve(target_pos, body_id, joint_indices)
        
        # Apply the optimized angles directly using position control
        self._apply_direct_position_control(optimized_angles, joint_indices)
        
        # Verify the new hand position
        new_pos = self.robot_state.get_hand_position(hand)
        new_dist = np.linalg.norm(new_pos - target_pos)
        print(f"After IK: {hand} hand position: {new_pos}")
        print(f"After IK: Distance to target: {new_dist}")
        
        return optimized_angles
    
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

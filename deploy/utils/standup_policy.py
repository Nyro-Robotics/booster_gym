import numpy as np
import torch
import logging # Add logging

logger = logging.getLogger(__name__) # Setup logger for this module

class StandupPolicy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            # Load the stand-up policy model
            standup_policy_path = self.cfg["policy"]["standup_policy_path"]
            self.policy = torch.jit.load(standup_policy_path)
            self.policy.eval()
            logger.info(f"Successfully loaded stand-up policy: {standup_policy_path}")
            
            # Load standup specific parameters from config
            self.num_observations = self.cfg["policy"]["standup_num_observations"]
            self.num_actions = self.cfg["policy"]["standup_num_actions"]
            self.num_stack = self.cfg["policy"]["num_stack"] # Expecting this in the config now
            self.standup_real_joint_indices = np.array(self.cfg["policy"]["standup_joint_indices"], dtype=int)
            
            if len(self.standup_real_joint_indices) != self.num_actions:
                 logger.warning(
                      f"Mismatch between num_actions ({self.num_actions}) and "
                      f"length of standup_joint_indices ({len(self.standup_real_joint_indices)})")
            # Check expected observation size components add up
            # 3 (grav) + 3 (ang_vel) + num_actions (rel_pos) + num_actions (vel) + num_actions (prev_actions) = 6 + 3 * num_actions
            expected_obs_size = 6 + 3 * self.num_actions 
            if self.num_observations != expected_obs_size:
                 logger.warning(
                      f"Configured standup_num_observations ({self.num_observations}) "
                      f"does not match expected size based on num_actions ({expected_obs_size}). Using configured value.")

        except KeyError as e:
             logger.error(f"Missing required key in config [policy] section: {e}")
             raise
        except Exception as e:
            logger.error(f"Failed to load stand-up policy or config: {e}")
            raise
        self._init_inference_variables()
        self.first_inference = True # Flag for initializing stacked_obs

    def get_policy_interval(self):
        return self.policy_interval # Return the pre-calculated interval

    def _init_inference_variables(self):
        # Get the full default qpos from common config
        full_default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        
        # Check if indices are valid for the full default qpos length
        if np.any(self.standup_real_joint_indices >= len(full_default_dof_pos)) or np.any(self.standup_real_joint_indices < 0):
             raise ValueError("Standup joint indices are out of bounds for common.default_qpos.")
             
        # Extract the default positions for the specific joints used by the standup policy
        self.standup_default_dof_pos_subset = full_default_dof_pos[self.standup_real_joint_indices]
        
        # This stores the target positions for ALL joints controlled by the robot (e.g., 23).
        self.dof_targets = np.copy(full_default_dof_pos)
        
        self.obs = np.zeros(self.num_observations, dtype=np.float32)
        self.actions = np.zeros(self.num_actions, dtype=np.float32) # Stores the latest computed actions
        
        # Initialize stacked_obs buffer
        # Shape will be (num_stack, num_observations) for easier handling
        self.stacked_obs = np.zeros((self.num_stack, self.num_observations), dtype=np.float32)
        
        # Calculate and store policy interval
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

        logger.info(f"StandupPolicy initialized. Observing/Actuating {self.num_actions} joints at indices: {self.standup_real_joint_indices.tolist()}. Using stack size {self.num_stack}. Policy interval: {self.policy_interval}s.")

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx=0.0, vy=0.0, vyaw=0.0):
        """
        Performs inference using the stand-up policy model, incorporating observation stacking and mirroring.
        Uses observation structure defined by standup_num_observations and standup_joint_indices.
        Observation structure (assuming 42 obs, 12 actions):
        - projected_gravity (3)
        - base_ang_vel (3)
        - relative_dof_pos for standup_joint_indices (12)
        - dof_vel for standup_joint_indices (12)
        - previous_actions (12) -> Note: Uses self.actions from the *previous* step.
        Ignores vx, vy, vyaw for observation construction.
        """
        
        # Check if real robot state vectors have expected length (e.g., 23)
        if len(dof_pos) < max(self.standup_real_joint_indices)+1 or \
           len(dof_vel) < max(self.standup_real_joint_indices)+1:
             logger.error(
                  f"StandupPolicy: Real robot dof_pos (len {len(dof_pos)}) or "
                  f"dof_vel (len {len(dof_vel)}) too short for standup_joint_indices {self.standup_real_joint_indices}."
             )
             # Cannot safely construct observation or apply actions
             return self.dof_targets # Return current targets (likely defaults)

        # --- Observation Construction ({self.num_observations} elements) ---
        current_obs_idx = 0
        norm_cfg = self.cfg["policy"]["normalization"]

        # 1. Projected Gravity (3 elements)
        self.obs[current_obs_idx : current_obs_idx+3] = projected_gravity * norm_cfg["gravity"]
        current_obs_idx += 3
        
        # 2. Base Angular Velocity (3 elements)
        self.obs[current_obs_idx : current_obs_idx+3] = base_ang_vel * norm_cfg["ang_vel"]
        current_obs_idx += 3

        # 3. Relative DoF Positions for the specified standup joints ({self.num_actions} elements)
        current_dof_pos_subset = dof_pos[self.standup_real_joint_indices]
        relative_dof_pos = (current_dof_pos_subset - self.standup_default_dof_pos_subset)
        self.obs[current_obs_idx : current_obs_idx + self.num_actions] = relative_dof_pos * norm_cfg["dof_pos"]
        current_obs_idx += self.num_actions
        
        # 4. DoF Velocities for the specified standup joints ({self.num_actions} elements)
        current_dof_vel_subset = dof_vel[self.standup_real_joint_indices]
        self.obs[current_obs_idx : current_obs_idx + self.num_actions] = current_dof_vel_subset * norm_cfg["dof_vel"]
        current_obs_idx += self.num_actions
            
        # 5. Previous Actions ({self.num_actions} elements)
        self.obs[current_obs_idx : current_obs_idx + self.num_actions] = self.actions
        current_obs_idx += self.num_actions

        # Verify final observation index matches expected size
        if current_obs_idx != self.num_observations:
            logger.warning(f"Constructed observation size ({current_obs_idx}) does not match configured ({self.num_observations})")

        # --- Update Stacked Observations ---
        if self.first_inference:
            # Initialize stack by repeating the first observation
            self.stacked_obs[:] = self.obs[np.newaxis, :]
            self.first_inference = False
        else:
            # Shift older observations back and add the new one at the front
            self.stacked_obs = np.roll(self.stacked_obs, shift=1, axis=0)
            self.stacked_obs[0, :] = self.obs

        # --- Policy Inference with Mirroring ---
        try:
            # Add batch dimension for the policy (expects [batch, stack, obs_dim])
            # Assuming the JIT model expects stacked obs as input [batch, stack, features]
            stacked_obs_tensor = torch.from_numpy(self.stacked_obs).unsqueeze(0)

            # Mirror stacked observations
            # Need to handle the stack dimension correctly in mirror_obs
            mirrored_stacked_obs_numpy = self.mirror_obs(self.stacked_obs)
            mirrored_stacked_obs_tensor = torch.from_numpy(mirrored_stacked_obs_numpy).unsqueeze(0)

            # Get actions from original and mirrored inputs
            # Assuming policy returns actions directly [batch, action_dim]
            actions_original_numpy = self.policy(stacked_obs_tensor).squeeze(0).detach().numpy()
            actions_mirrored_raw_numpy = self.policy(mirrored_stacked_obs_tensor).squeeze(0).detach().numpy()

            # Mirror the actions derived from mirrored observations
            actions_mirrored_processed_numpy = self.mirror_act(actions_mirrored_raw_numpy)

            # Average original and processed mirrored actions
            final_actions = 0.5 * (actions_original_numpy + actions_mirrored_processed_numpy)

            # Clip and store actions for the *next* observation and for target calculation
            clip_val = norm_cfg["clip_actions"]
            self.actions[:] = np.clip(final_actions, -clip_val, clip_val)

        except Exception as e:
             logger.exception(f"Error during stand-up policy inference (stacking/mirroring): {e}") # Use logger.exception for traceback
             self.actions.fill(0.0) # Default to zero actions on error

        # --- Target Calculation ---
        # Start with the full default positions for ALL joints
        # Note: self.default_dof_pos needs to be the full 23-element array here.
        # Let's ensure it is available, maybe store it as self.full_default_dof_pos in init.
        # Re-fetch from cfg for clarity, or use a stored full version.
        full_default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.dof_targets[:] = full_default_dof_pos 
        
        # Apply scaled actions from policy ONLY to the specified standup joints
        action_scale = self.cfg["policy"]["control"]["action_scale"]
        self.dof_targets[self.standup_real_joint_indices] += action_scale * self.actions
        
        # Joints not in standup_real_joint_indices remain at their default positions.
        # If specific behavior is needed for non-controlled joints during standup
        # (e.g., keep head still), it would need explicit logic here.

        return self.dof_targets 

    # --- Added: Mirroring Static Methods ---
    @staticmethod
    def mirror_obs(obs):
        """Mirrors observations. Handles stacked observations (N, 42) or single (42,)."""
        # Assuming obs has 42 features based on mujoco.py
        num_features = 42
        if obs.shape[-1] != num_features:
             raise ValueError(f"Expected observation to have {num_features} features, but got {obs.shape[-1]}")

        mat = np.zeros((num_features, num_features), dtype=np.float32)
        # This mapping assumes the standard 42 obs structure from mujoco.py
        # [grav(3), ang_vel(3), rel_dof_pos(12), dof_vel(12), prev_actions(12)]
        # Mirroring affects: y-gravity, x/z ang_vel, L/R joint pos/vel/actions
        mat[ 0: 6,  0: 6] = np.eye(6) # Grav/AngVel base structure
        # Joint Pos (12) indices 6-17
        mat[ 6: 8,  8:10] = np.eye(2) # Hip R/L
        mat[ 8:10,  6: 8] = np.eye(2)
        mat[10:14, 14:18] = np.eye(4) # Knee/Ankle R/L
        mat[14:18, 10:14] = np.eye(4)
        # Joint Vel (12) indices 18-29
        mat[18:20, 20:22] = np.eye(2) # Hip R/L vel
        mat[20:22, 18:20] = np.eye(2)
        mat[22:26, 26:30] = np.eye(4) # Knee/Ankle R/L vel
        mat[26:30, 22:26] = np.eye(4)
        # Prev Actions (12) indices 30-41
        mat[30:32, 32:34] = np.eye(2) # Hip R/L actions
        mat[32:34, 30:32] = np.eye(2)
        mat[34:38, 38:42] = np.eye(4) # Knee/Ankle R/L actions
        mat[38:42, 34:38] = np.eye(4)

        # Indices to negate
        flip_val = np.ones(num_features, dtype=np.float32)
        inverse_ids = [  1, # grav y
                         3, 5, # ang_vel x, z
                         7,  9, 11, 15, # joint pos (roll/yaw R, pitch L)
                        19, 21, 23, 27, # joint vel (roll/yaw R, pitch L)
                        31, 33, 35, 39] # prev actions (roll/yaw R, pitch L)
        flip_val[inverse_ids] = -1
        flip_mat = np.diag(flip_val)
        mirror_transform_mat = np.dot(mat, flip_mat) # Shape (42, 42)

        # Handle stacked (N, 42) or single (42,) obs
        orig_shape = obs.shape
        reshaped_obs = obs.reshape(-1, num_features) # Shape (N, 42) or (1, 42)
        mirrored_obs = np.dot(reshaped_obs, mirror_transform_mat.T) # (N, 42) @ (42, 42) -> (N, 42)
        mirrored_obs = mirrored_obs.reshape(orig_shape) # Restore original shape
        return mirrored_obs

    @staticmethod
    def mirror_act(act):
        """Mirrors actions. Handles batched (N, 12) or single (12,) actions."""
        # Assuming act has 12 features based on mujoco.py
        num_features = 12
        if act.shape[-1] != num_features:
             raise ValueError(f"Expected action to have {num_features} features, but got {act.shape[-1]}")

        mat = np.zeros((num_features, num_features), dtype=np.float32)
        # Action mirroring mapping (12 actions)
        mat[0:2, 2:4] = np.eye(2) # Hip R/L
        mat[2:4, 0:2] = np.eye(2)
        mat[4:8, 8:12] = np.eye(4) # Knee/Ankle R/L
        mat[8:12, 4:8] = np.eye(4)

        # Indices to negate
        flip_val = np.ones(num_features, dtype=np.float32)
        inverse_ids = [1, 3, 5, 9] # Actions corresponding to roll/yaw R, pitch L
        flip_val[inverse_ids] = -1
        flip_mat = np.diag(flip_val)
        mirror_transform_mat = np.dot(mat, flip_mat) # Shape (12, 12)

        # Handle batched (N, 12) or single (12,) actions
        orig_shape = act.shape
        reshaped_act = act.reshape(-1, num_features) # Shape (N, 12) or (1, 12)
        mirrored_act = np.dot(reshaped_act, mirror_transform_mat.T) # (N, 12) @ (12, 12) -> (N, 12)
        mirrored_act = mirrored_act.reshape(orig_shape) # Restore original shape
        return mirrored_act
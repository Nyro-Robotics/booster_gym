import numpy as np
import mujoco
import attrs
from typing import Dict, List, Optional, Any

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
    left_arm_joint_indices: List[int] = attrs.field(factory=list)
    left_arm_joint_names: List[str] = attrs.field(factory=list)
    right_arm_joint_indices: List[int] = attrs.field(factory=list)
    right_arm_joint_names: List[str] = attrs.field(factory=list)
    
    def __attrs_post_init__(self):
        # Initialize target positions to current hand positions
        self.target_left_pos = self.get_hand_position("left")
        self.target_right_pos = self.get_hand_position("right")
        
        # Create lists of joint indices for different body parts
        self.lower_body_joints = []
        self.torso_joints = []

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

    def load_joint_definitions(self, joint_config: Dict[str, Any]) -> None:
        """Load joint (lower body, torso, left arm, right arm) definitions from configuration.
        
        Args:
            joint_config: Joint configuration dictionary from config file
        """
        # Load lower body joints
        if "init_state" in joint_config:
            # Lower body joints
            if "lower_body_joints" in joint_config["init_state"]:
                for joint_id_str, joint_name in joint_config["init_state"]["lower_body_joints"].items():
                    joint_id = int(joint_id_str)
                    self.lower_body_joints.append(joint_id)
                    # Also add to joint_indices if not already there
                    if joint_name not in self.joint_indices:
                        self.joint_indices[joint_name] = joint_id
                print(f"Loaded {len(self.lower_body_joints)} lower body joints from config")
            
            # Torso joints
            if "torso_joints" in joint_config["init_state"]:
                for joint_id_str, joint_name in joint_config["init_state"]["torso_joints"].items():
                    joint_id = int(joint_id_str)
                    self.torso_joints.append(joint_id)
                    # Also add to joint_indices if not already there
                    if joint_name not in self.joint_indices:
                        self.joint_indices[joint_name] = joint_id
                print(f"Loaded {len(self.torso_joints)} torso joints from config")
            
            # Left arm joints
            if "left_arm_joints" in joint_config["init_state"]:
                for joint_id_str, joint_name in joint_config["init_state"]["left_arm_joints"].items():
                    joint_id = int(joint_id_str)
                    self.left_arm_joint_indices.append(joint_id)
                    self.left_arm_joint_names.append(joint_name)
                    # Also add to joint_indices if not already there
                    if joint_name not in self.joint_indices:
                        self.joint_indices[joint_name] = joint_id
                print(f"Loaded {len(self.left_arm_joint_indices)} left arm joints from config")
            
            # Right arm joints
            if "right_arm_joints" in joint_config["init_state"]:
                for joint_id_str, joint_name in joint_config["init_state"]["right_arm_joints"].items():
                    joint_id = int(joint_id_str)
                    self.right_arm_joint_indices.append(joint_id)
                    self.right_arm_joint_names.append(joint_name)
                    # Also add to joint_indices if not already there
                    if joint_name not in self.joint_indices:
                        self.joint_indices[joint_name] = joint_id
                print(f"Loaded {len(self.right_arm_joint_indices)} right arm joints from config")

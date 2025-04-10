import numpy as np
import mujoco
import sys
import yaml
from typing import Dict, Any, Tuple
from teleoperation.teleop_types import RobotState

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
    """Initialize the robot state with joint limits, joint indices, and body IDs.
    
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
    
    # Load joint definitions from config
    robot_state.load_joint_definitions(joint_config)
    
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
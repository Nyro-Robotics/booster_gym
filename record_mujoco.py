"""Robot simulation and recording tools for MuJoCo environments.

This module provides functionality for:
- Running robot simulations based on config files
- Interactive control of the robot with keyboard commands
- Recording videos in both interactive and headless modes
- Parallel rendering of frames for efficient video creation
"""

import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco
import mujoco_viewer
from utils.model import ActorCritic
import cv2
import time
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union, Deque


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.
    
    Args:
        q: Quaternion in xyzw format.
        v: Vector to rotate.
        
    Returns:
        Rotated vector.
    """
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def save_state(mj_data: mujoco.MjData, lin_vel_x: float = 0.0, lin_vel_y: float = 0.0, ang_vel_yaw: float = 0.0) -> Dict[str, Any]:
    """Save the current state of the simulation.
    
    Args:
        mj_data: MuJoCo data object
        lin_vel_x: Current linear velocity command in x direction
        lin_vel_y: Current linear velocity command in y direction
        ang_vel_yaw: Current angular velocity command around yaw axis
        
    Returns:
        Dictionary containing the state
    """
    return {
        'qpos': mj_data.qpos.copy(),
        'qvel': mj_data.qvel.copy(),
        'lin_vel_x': lin_vel_x,
        'lin_vel_y': lin_vel_y,
        'ang_vel_yaw': ang_vel_yaw
    }


def restore_state(mj_data: mujoco.MjData, mj_model: mujoco.MjModel, state: Dict[str, Any]) -> None:
    """Restore a previously saved simulation state.
    
    Args:
        mj_data: MuJoCo data object to update.
        mj_model: MuJoCo model object.
        state: Previously saved state dictionary.
    """
    mj_data.qpos[:] = state['qpos']
    mj_data.qvel[:] = state['qvel']
    mujoco.mj_forward(mj_model, mj_data)


def create_video_writer(output_path: str, fps: int, width: int, height: int) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """Create a video writer with an appropriate codec.
    
    Tries multiple codecs until a working one is found.
    
    Args:
        output_path: Desired path for the output video.
        fps: Frames per second for the video.
        width: Width of the video in pixels.
        height: Height of the video in pixels.
        
    Returns:
        Tuple of (video_writer object, actual output path) or (None, None) if no codec works.
    """
    codecs_to_try = [
        ('VP80', '.webm'),  # WebM with VP8 codec - good VSCode compatibility
        ('VP90', '.webm'),  # WebM with VP9 codec (better quality but might not be available)
        ('avc1', '.mp4'),   # This is often available and works with VSCode
        ('mp4v', '.mp4'),   # Standard MP4 codec
        ('XVID', '.avi'),   # Widely compatible
        ('MJPG', '.avi'),   # Very widely supported
        ('FMP4', '.mp4'),   # Another MP4 variant
    ]
    
    for codec, ext in codecs_to_try:
        if ext != os.path.splitext(output_path)[1]:
            # Adjust file extension to match codec
            codec_output = os.path.splitext(output_path)[0] + ext
        else:
            codec_output = output_path
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                codec_output, 
                fourcc, 
                fps, 
                (width, height)
            )
            
            if writer.isOpened():
                print(f"Successfully created video writer with codec {codec} ({ext})")
                return writer, codec_output
            else:
                print(f"Codec {codec} not available or failed to open, trying next...")
        except Exception as e:
            print(f"Error with codec {codec}: {str(e)}")
    
    print("ERROR: All codecs failed. Video recording will not work.")
    return None, None


def add_overlay(frame: np.ndarray, command_text: str) -> np.ndarray:
    """Add command text overlay to the frame.
    
    Args:
        frame: The video frame to add text to
        command_text: The text to display
        
    Returns:
        Frame with text overlay
    """
    # Create a copy to avoid modifying the original
    result = frame.copy()
    
    # Add semi-transparent background for better readability
    overlay = result.copy()
    text_width = len(command_text) * 8 + 20  # Adjust width based on text length
    text_bg_coords = (result.shape[1] - text_width, 10, result.shape[1] - 10, 40)
    cv2.rectangle(overlay, (text_bg_coords[0], text_bg_coords[1]), 
                 (text_bg_coords[2], text_bg_coords[3]), (0, 0, 0), -1)
    
    # Apply the overlay with transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Add text (smaller font)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, command_text, (result.shape[1] - text_width + 10, 30), 
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return result

def render_batch(
    states_batch: List[Dict[str, Any]], 
    batch_idx: int, 
    result_queue: Optional[mp.Queue], 
    model_path: str, 
    camera_settings: Dict[str, float], 
    width: int, 
    height: int,
    default_command_values: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Optional[List[np.ndarray]]:
    """Render a batch of frames in a separate process or directly."""
    # Set environment variable to use software rendering if in a subprocess
    if result_queue is not None:
        os.environ['MUJOCO_GL'] = 'osmesa'
    
    try:
        # Load model in this process
        local_model = mujoco.MjModel.from_xml_path(model_path)
        local_data = mujoco.MjData(local_model)
        
        # Create offscreen viewer
        local_viewer = mujoco_viewer.MujocoViewer(local_model, local_data, 'offscreen', width=width, height=height)
        
        # Apply camera settings
        local_viewer.cam.elevation = camera_settings['elevation']
        local_viewer.cam.azimuth = camera_settings['azimuth']  # Set to 0 as requested
        local_viewer.cam.distance = camera_settings['distance']
        
        # Render each state in the batch
        frames = []
        for state in states_batch:
            # Restore state
            local_data.qpos[:] = state['qpos']
            local_data.qvel[:] = state['qvel']
            mujoco.mj_forward(local_model, local_data)
            
            # Get command values from state or use defaults
            lin_vel_x = state.get('lin_vel_x', default_command_values[0])
            lin_vel_y = state.get('lin_vel_y', default_command_values[1])
            ang_vel_yaw = state.get('ang_vel_yaw', default_command_values[2])
            
            # Get robot position
            robot_pos = local_data.qpos.astype(np.float32)[0:3]
            
            # Update camera lookat to follow the robot
            local_viewer.cam.lookat[:] = robot_pos
            
            
            # Render the scene
            pixels = local_viewer.read_pixels()
            
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            
            # Add command text overlay for the video
            frame = add_overlay(frame, f"x={lin_vel_x:.2f}, y={lin_vel_y:.2f}, yaw={ang_vel_yaw:.2f}")
            
            # Ensure frame is the right type
            frame = frame.astype(np.uint8)
            
            frames.append(frame)
        
        # Clean up
        local_viewer.close()
        
        # Return the rendered frames
        if result_queue is not None:
            result_queue.put((batch_idx, frames))
            return None
        else:
            return frames
    except Exception as e:
        import traceback
        error_msg = f"Error in batch {batch_idx}: {str(e)}\n{traceback.format_exc()}"
        # Return the detailed error information
        if result_queue is not None:
            result_queue.put((batch_idx, error_msg))
            return None
        else:
            print(error_msg)
            return []


def main() -> None:
    """Main function to run the simulation.
    
    Handles command-line arguments, configuration loading, and the main simulation loop.
    """
    # This needs to be called before any other multiprocessing code
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    parser.add_argument("--headless_record", action="store_true", help="Record in headless mode (no viewer, batch render at end).")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without a viewer.")
    parser.add_argument("--output", type=str, default="simulation_video.mp4", help="Output video file name.")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for video recording. If not provided, uses 1/dt from config.")
    parser.add_argument("--max_frames", type=int, default=10000, help="Maximum number of frames to store in headless mode.")
    parser.add_argument("--render_threads", type=int, default=5, help="Number of threads to use for parallel rendering. Default is 2.")
    parser.add_argument("--render_batch_size", type=int, default=60, help="Number of frames to render in each batch.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Store every Nth frame to achieve target FPS while maintaining simulation dt.")
    args = parser.parse_args()
    
    # Load configuration
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    # Calculate simulation FPS based on dt
    sim_fps = int(1.0 / cfg["sim"]["dt"])
    
    # Determine target recording FPS and frame skip
    if args.fps is None:
        # If FPS not specified, use simulation FPS (no frame skipping)
        args.fps = sim_fps
        args.frame_skip = 1
    else:
        # Calculate frame skip to achieve target FPS while maintaining simulation dt
        # Ensure we don't skip too many frames if target FPS is very low
        calculated_skip = max(1, int(sim_fps / args.fps))
        if args.frame_skip == 1:  # Only override if not explicitly set
            args.frame_skip = calculated_skip
    
    # Recalculate actual FPS based on frame_skip
    actual_fps = sim_fps / args.frame_skip
    
    print(f"Simulation running at {sim_fps} Hz (dt={cfg['sim']['dt']})")
    print(f"Recording every {args.frame_skip} frame(s) for a video at {actual_fps:.1f} FPS")

    # Ensure the viewer section exists in the config
    if "viewer" not in cfg:
        raise ValueError(f"Config file {cfg_file} does not contain a 'viewer' section")
    
    # Load viewer settings from config
    camera_pos = cfg["viewer"]["pos"]
    camera_lookat = cfg["viewer"]["lookat"]
    record_video = cfg["viewer"]["record_video"]
    record_interval = cfg["viewer"]["record_interval"]
    record_env_idx = cfg["viewer"]["record_env_idx"]
    
    # Override record setting if specified in config
    if record_video and not args.headless_record:
        args.headless_record = True
        print("Recording enabled based on config settings")

    # Load model
    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    # Initialize MuJoCo simulation
    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    
    # Set up default joint positions and PD gains
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    
    # Initialize joint positions from config
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        # Set PD gains for position control
        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    
    # Set initial pose
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    # Initialize control variables
    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    # Video recording setup
    width, height = 640, 480  # Default resolution
    video_writer = None
    recording = False
    
    # For headless recording - store states instead of rendering
    headless_recording = False
    stored_states: Deque[Dict[str, Any]] = deque(maxlen=args.max_frames)
    
    # Frame counter for skipping
    frame_counter = 0
    
    # Create the viewer object using mujoco_viewer for interactive display
    viewer = None if args.headless or args.headless_record else mujoco_viewer.MujocoViewer(mj_model, mj_data, width=width, height=height)
    
    # Apply camera settings from config
    if viewer:
        viewer.cam.elevation = -20  # Default elevation
        if hasattr(viewer.cam, 'azimuth'):
            viewer.cam.azimuth = 0  # Default azimuth
        
        # Set camera position and lookat from config
        if hasattr(viewer.cam, 'distance'):
            # Calculate distance from position to lookat
            pos = np.array(camera_pos)
            lookat = np.array(camera_lookat)
            distance = np.linalg.norm(pos - lookat)
            viewer.cam.distance = distance
        
        # Set lookat point
        viewer.cam.lookat[:] = camera_lookat
    
    # Create an offscreen viewer for recording and video writer
    offscreen_viewer = None
    
    print("Commands:")
    print(" - Set movement command: x y yaw")
    print(" - Toggle recording: r")
    print(" - Stop recording: s")
    print(" - Quit: q")
    print("\nSet command (x, y, yaw) or action: ")
    
    recording_start_time = None
    
    # Limit render threads to reduce CPU load
    args.render_threads = min(args.render_threads, mp.cpu_count() - 1)
    print(f"Using {args.render_threads} rendering threads to avoid overloading CPU")
    
    # Main simulation loop
    while (viewer is None) or (viewer is not None and viewer.is_alive):
        # Check for user input
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                input_str = sys.stdin.readline().strip()
                if input_str == "q":
                    break
                elif input_str == "r":
                    if not headless_recording:
                        headless_recording = True
                        stored_states.clear()
                        frame_counter = 0  # Reset frame counter when starting recording
                        print("Recording started. States will be stored for batch rendering.")
                        print(f"Storing every {args.frame_skip}th frame (target FPS: {actual_fps:.1f})")
                    else:
                        print("Already recording")
                elif input_str == "s":
                    if headless_recording:
                        headless_recording = False
                        print(f"Recording stopped. Collected {len(stored_states)} frames. Preparing for rendering...")
                        
                        # Create video writer with the correct FPS
                        video_writer, output_path = create_video_writer(args.output, actual_fps, width, height)
                        
                        if video_writer is None:
                            print("Error: Could not create video writer. Rendering aborted.")
                        else:
                            # More memory efficient approach: render in smaller batches and write directly
                            start_time = time.time()
                            print(f"Starting rendering with {args.render_threads} threads...")
                            
                            # Convert deque to list for easier batch processing
                            states_list = list(stored_states)
                            total_frames = len(states_list)
                            
                            # Create smaller batches
                            batches = []
                            batch_size = min(args.render_batch_size, max(1, total_frames // (args.render_threads * 2)))
                            for i in range(0, total_frames, batch_size):
                                end = min(i + batch_size, total_frames)
                                batches.append(states_list[i:end])
                            
                            # Prepare camera settings
                            camera_settings = {
                                'elevation': -20,
                                'azimuth': 0,  # Set to 0 as requested
                                'distance': np.linalg.norm(np.array(camera_pos) - np.array(camera_lookat))
                            }
                            
                            # Special case for render_threads=1: call render_batch directly for debugging
                            if args.render_threads == 1:
                                print("Using direct rendering for debugging (render_threads=1)")
                                with tqdm(total=len(batches), desc="Rendering batches") as pbar:
                                    for batch_idx, batch in enumerate(batches):
                                        # Call render_batch directly in the main process
                                        result = render_batch(
                                            batch, batch_idx, None, cfg["asset"]["mujoco_file"], 
                                            camera_settings, width, height
                                        )
                                        
                                        # If render_batch returns frames directly when queue is None
                                        if result is not None:
                                            frames = result
                                            # Write frames to video
                                            for frame in frames:
                                                video_writer.write(frame)
                                        
                                        pbar.update(1)
                            else:
                                # Process batches with a limited number of workers
                                result_queue: mp.Queue = mp.Queue(maxsize=args.render_threads * 2)  # Limit queue size
                                active_processes = []
                                next_batch = 0
                                completed_batches = 0
                                
                                # Start initial processes
                                with tqdm(total=len(batches), desc="Rendering batches") as pbar:
                                    for i in range(min(args.render_threads, len(batches))):
                                        p = mp.Process(
                                            target=render_batch,
                                            args=(batches[next_batch], next_batch, result_queue, cfg["asset"]["mujoco_file"], 
                                                  camera_settings, width, height)
                                        )
                                        active_processes.append((next_batch, p))
                                        p.start()
                                        next_batch += 1
                                    
                                    # Process results and start new processes as needed
                                    all_frames = {}
                                    
                                    while completed_batches < len(batches):
                                        try:
                                            batch_idx, frames = result_queue.get(timeout=30)
                                            
                                            # Handle error or process frames
                                            if isinstance(frames, str):  # Error message
                                                print(f"\nError in batch {batch_idx}: {frames}")
                                            else:
                                                # Store frames
                                                all_frames[batch_idx] = frames
                                                
                                                # Write frames to video immediately to free memory
                                                for frame in frames:
                                                    video_writer.write(frame)
                                                
                                                # Free memory
                                                frames = None
                                            
                                            # Find and remove the completed process
                                            for i, (idx, p) in enumerate(active_processes):
                                                if idx == batch_idx:
                                                    p.join()
                                                    active_processes.pop(i)
                                                    break
                                            
                                            # Start a new process if more batches remain
                                            if next_batch < len(batches):
                                                p = mp.Process(
                                                    target=render_batch,
                                                    args=(batches[next_batch], next_batch, result_queue, cfg["asset"]["mujoco_file"], 
                                                          camera_settings, width, height)
                                                )
                                                active_processes.append((next_batch, p))
                                                p.start()
                                                next_batch += 1
                                            
                                            completed_batches += 1
                                            pbar.update(1)
                                            
                                        except queue.Empty:
                                            print("Waiting for rendering to complete...")
                                            # Check if processes are still alive
                                            for idx, p in active_processes:
                                                if not p.is_alive():
                                                    print(f"Process for batch {idx} crashed")
                                            
                                        except Exception as e:
                                            import traceback
                                            print(f"\nError: {str(e)}")
                                            print(traceback.format_exc())
                                    
                                # Clean up processes
                                for idx, p in active_processes:
                                    if p.is_alive():
                                        print(f"Terminating process {idx} that is still running")
                                        p.terminate()
                                    p.join(timeout=1)
                                
                            # Release the video writer
                            video_writer.release()
                            
                            print(f"Video saved to {output_path}")
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                print(f"Video file created successfully: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
                            else:
                                print(f"Warning: Video file may not have been created properly")
                            
                            # Clear stored states to free memory
                            stored_states.clear()
                    else:
                        print("Not recording")
                else:
                    parts = input_str.split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw) or action: ",
                            end="",
                        )
                    else:
                        raise ValueError
            except ValueError:
                print("Invalid input. Enter three numeric values or r/s/q.\nSet command (x, y, yaw) or action: ", end="")
        
        # Get current robot state
        dof_pos = mj_data.qpos.astype(np.float32)[7:]
        dof_vel = mj_data.qvel.astype(np.float32)[6:]
        quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
        base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        # Policy evaluation at specified decimation rate
        if it % cfg["control"]["decimation"] == 0:
            obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
            obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
            obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
            obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
            obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
            obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
            obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
            obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
            obs[35:47] = actions
            dist = model.act(torch.tensor(obs).unsqueeze(0))
            actions[:] = dist.loc.detach().numpy()
            actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
            dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            
        # Apply PD control
        mj_data.ctrl = np.clip(
            dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[:, 0],
            mj_model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(mj_model, mj_data)
        
        # Increment frame counter
        frame_counter += 1
        
        # Store state for recording only on frame_skip interval
        if headless_recording and (frame_counter % args.frame_skip == 0):
            stored_states.append(save_state(mj_data, lin_vel_x, lin_vel_y, ang_vel_yaw))
                
            # Print progress occasionally
            if len(stored_states) % 500 == 0:
                sim_time = len(stored_states) * args.frame_skip * cfg["sim"]["dt"]
                print(f"Stored {len(stored_states)} frames ({sim_time:.2f}s of simulation time)")
        
        # Update viewer if not in headless mode
        if viewer is not None:
            # Set camera lookat to follow the robot
            robot_pos = mj_data.qpos.astype(np.float32)[0:3]
            viewer.cam.lookat[:] = robot_pos
            
            # Render the scene normally
            viewer.render()
            
            # Add command text overlay using MuJoCo's native text rendering
            command_text = f"x={lin_vel_x:.2f}, y={lin_vel_y:.2f}, yaw={ang_vel_yaw:.2f}"
            
            # Add text overlay if the viewer supports it
            if hasattr(viewer, 'add_overlay'):
                # Some viewers might have a built-in method
                viewer.add_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, command_text)
            elif hasattr(viewer, 'ctx') and hasattr(viewer, 'viewport'):
                # Try using MuJoCo's overlay function directly
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    viewer.viewport,
                    command_text,
                    "",
                    viewer.ctx
                )
                
                # If the viewer uses GLFW, we might need to swap buffers manually
                if hasattr(viewer, 'window') and viewer.window is not None:
                    try:
                        import glfw
                        glfw.swap_buffers(viewer.window)
                    except (ImportError, AttributeError):
                        pass  # GLFW not available or not needed
        
        it += 1
        gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)
        
    # Close the viewers
    if viewer is not None:
        viewer.close()
    if offscreen_viewer is not None:
        offscreen_viewer.close()
    
    # Make sure to release the video writer when done
    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    main()
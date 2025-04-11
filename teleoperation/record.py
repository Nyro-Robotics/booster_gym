import cv2
import numpy as np
import mujoco
from typing import List, Optional, Tuple, Callable, Dict
import attrs
import os
import glfw

def capture_mujoco_frame(
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    width: int, 
    height: int,
    camera_name: Optional[str] = None,
    camera_id: int = -1,
    camera_matrix: Optional[np.ndarray] = None,
    text_overlay: Optional[List[Tuple[str, Tuple[int, int], float, Tuple[int, int, int], int]]] = None,
    markers: Optional[List[Dict[str, any]]] = None
) -> np.ndarray:
    """Capture a frame from MuJoCo simulation with optional markers and text overlays.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        width: Frame width
        height: Frame height
        camera_name: Name of camera to use (if provided)
        camera_id: ID of camera to use (if camera_name not provided)
        camera_matrix: Optional camera matrix to override default view
        text_overlay: List of text overlays in format [(text, position, scale, color, thickness), ...]
        markers: List of marker dictionaries with keys:
                - type: mujoco.mjtGeom enum value
                - size: [x, y, z] size array
                - pos: [x, y, z] position array
                - rgba: [r, g, b, a] color array
                - label: (optional) text label
                - mat: (optional) 3x3 rotation matrix (flattened)
                - from_: (optional) start position for lines
                - to: (optional) end position for lines
        
    Returns:
        BGR image as numpy array
    """
    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Set camera
    if camera_name is not None:
        # Use named camera
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            print(f"Warning: Camera '{camera_name}' not found, using default view")
    
    if camera_id >= 0:
        # Use specific camera
        renderer.update_scene(data, camera=camera_id)
    elif camera_matrix is not None:
        # Use custom camera matrix
        renderer.update_scene(data)
        # Set custom camera matrix (if supported by this version of MuJoCo)
        try:
            renderer.scene.camera.fixedcamid = -1
            renderer.scene.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Apply camera matrix
            # This is version-dependent and may need adjustment
            pass
        except Exception as e:
            print(f"Warning: Could not set custom camera matrix: {e}")
            renderer.update_scene(data)
    else:
        # Use default view
        renderer.update_scene(data)
    
    # Add markers to the scene if provided
    if markers:
        try:
            scene = renderer.scene
            
            for marker in markers:
                if scene.ngeom >= scene.maxgeom:
                    print("Warning: Max geoms reached, skipping marker.")
                    break
                
                g = scene.geoms[scene.ngeom]
                
                # Use mjv_initGeom for modern MuJoCo versions if available, otherwise set manually
                if hasattr(mujoco, 'mjv_initGeom'):
                    # For lines, we need special handling
                    if marker.get("type") == mujoco.mjtGeom.mjGEOM_LINE and "from_" in marker and "to" in marker:
                        mujoco.mjv_initGeom(
                            g,
                            marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE),
                            np.zeros(3),  # Size not used for lines
                            np.zeros(3),  # Position not used directly for lines
                            marker.get("mat", np.eye(3).flatten()),
                            marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0]))
                        )
                        # Set line endpoints
                        g.pos[:] = marker["from_"]
                        g.size[:] = marker["to"] - marker["from_"]
                    else:
                        # Normal geoms
                        mujoco.mjv_initGeom(
                            g,
                            marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE),
                            marker.get("size", np.array([0.01, 0.01, 0.01])),
                            marker.get("pos", np.array([0.0, 0.0, 0.0])),
                            marker.get("mat", np.eye(3).flatten()),
                            marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0]))
                        )
                    
                    # Set label separately if needed
                    g.label = marker.get("label", "")
                
                else:
                    # Fallback for older MuJoCo versions or manual setting
                    g.type = marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE)
                    g.size[:] = marker.get("size", np.array([0.01, 0.01, 0.01]))
                    g.pos[:] = marker.get("pos", np.array([0.0, 0.0, 0.0]))
                    g.mat[:] = marker.get("mat", np.eye(3).flatten())
                    g.rgba[:] = marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0]))
                    g.label = marker.get("label", "")
                    
                    # Handle line geoms
                    if marker.get("type") == mujoco.mjtGeom.mjGEOM_LINE and "from_" in marker and "to" in marker:
                        g.pos[:] = marker["from_"]
                        g.size[:] = marker["to"] - marker["from_"]
                    
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
        
        except Exception as e:
            print(f"Warning: Error adding markers: {e}")
            import traceback
            traceback.print_exc()
    
    # Render the frame
    frame = renderer.render()
    
    # Convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Add text overlays if provided
    if text_overlay:
        for text, position, scale, color, thickness in text_overlay:
            cv2.putText(
                frame_bgr,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness
            )
    
    return frame_bgr


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
        self.frames = []  # Clear any existing frames
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
            for i, frame in enumerate(self.frames):
                # Ensure frame is in the correct format
                if frame.shape[0] != self.height or frame.shape[1] != self.width:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Ensure frame is BGR (not RGB)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Already in correct format, write directly
                    video_writer.write(frame)
                else:
                    print(f"Warning: Frame {i} has unexpected shape {frame.shape}")
                    # Try to convert if possible
                    try:
                        if len(frame.shape) == 2:  # Grayscale
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        video_writer.write(frame)
                    except Exception as e:
                        print(f"Error writing frame {i}: {e}")
            
            video_writer.release()
            print(f"Video saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear frames to free memory
        self.frames = []

# deprecated, from HeadlessInterface
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
        glfw.terminate()
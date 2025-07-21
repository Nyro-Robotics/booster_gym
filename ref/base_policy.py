import sys
import threading
import time
import yaml
import argparse
import numpy as np
import onnxruntime
from loop_rate_limiters import RateLimiter
from sshkeyboard import listen_keyboard
from termcolor import colored

sys.path.append("../")
sys.path.append("./")

from sim2real.utils.comm import create_state_processor, create_command_sender

from sim2real.utils.robot import Robot
from sim2real.utils.math import quat_rotate_inverse_numpy
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

class BasePolicy:
    """
    Base policy class for FALCON deployment on Unitree humanoid robots.
    Supports both simulation and real robot deployment with keyboard/joystick controls.
    """
    
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=0.25):
        """Initialize the base policy with configuration and model."""
        self.config = config
        # Initialize robot config
        self._init_robot_config()
        # Initialize SDK components
        self._init_sdk_components()
        # Initialize observation config
        self._init_obs_config()
        # Initialize communication components
        self._init_communication_components()
        # Initialize policy components
        self._init_policy_components(model_path, policy_action_scale, rl_rate)
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()

    # ============================================================================
    # Initialization Methods
    # ============================================================================
    
    def _init_robot_config(self):
        """Initialize robot configuration and parameters."""
        self.robot = Robot(self.config)
        self.num_dofs = self.robot.NUM_JOINTS
        self.default_dof_angles = np.array(self.robot.DEFAULT_DOF_ANGLES)
        self.num_upper_dofs = self.config.get("NUM_UPPER_BODY_JOINTS", 14)
        
        # Initialize motor limits (only position limits are used)
        self.motor_pos_lower_limit_list = self.config.get("motor_pos_lower_limit_list", None)
        self.motor_pos_upper_limit_list = self.config.get("motor_pos_upper_limit_list", None)
        
        # Setup dof names and indices
        self._setup_dof_mappings()
    
    def _setup_dof_mappings(self):
        """Setup DOF names and their corresponding indices."""
        self.dof_names = self.config.get("dof_names", None)
        self.upper_dof_names = self.config.get("dof_names_upper_body", None)
        self.lower_dof_names = self.config.get("dof_names_lower_body", None)
        
        # These are used by derived classes, so keep them
        if self.upper_dof_names:
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
        else:
            self.upper_dof_indices = []
            
        if self.lower_dof_names:
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
        else:
            self.lower_dof_indices = []
    
    def _init_sdk_components(self):
        """Initialize SDK components based on robot type."""
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        
        if self.sdk_type == "unitree":
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "booster":
            from booster_robotics_sdk_python import ChannelFactory
            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"], self.config["NET"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
    
    def _init_obs_config(self):
        """Initialize observation configuration and buffers."""
        self.obs_scales = self.config["obs_scales"]
        self.obs_dims = self.config["obs_dims"]
        self.obs_dict = self.config["obs_dict"]
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.config["history_length_dict"]
        
        # Initialize observation buffers
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
    
    def _init_communication_components(self):
        """Initialize state processor and command sender using the wrapper."""
        self.state_processor = create_state_processor(self.config)
        self.command_sender = create_command_sender(self.config)
    
    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Initialize policy-related components."""
        self.setup_policy(model_path)
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        self.policy_action_scale = policy_action_scale
    
    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.get("DESIRED_BASE_HEIGHT", 0.78)
        self.gait_period = self.config.get("GAIT_PERIOD", 0.5)
        
        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])
        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))
        
        # Upper body controller
        self.upper_body_controller = None
    
    def _init_input_handlers(self):
        """Initialize input handlers (ROS, joystick, keyboard)."""
        self._init_rate_handler()
        self._init_input_device()
    
    def _init_rate_handler(self):
        """Initialize ROS handler if enabled."""
        from loguru import logger
        self.logger = logger
        self.rate = RateLimiter(self.config.get("rl_rate", 50))
    
    def _init_input_device(self):
        """Initialize input device (joystick or keyboard)."""
        if self.config.get("USE_JOYSTICK", False):
            self._init_joystick_handler()
        else:
            self._init_keyboard_handler()
    
    def _init_joystick_handler(self):
        """Initialize joystick handler."""
        if sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Windows or Mac.")
            self.logger.warning("Using keyboard instead")
            self.use_joystick = False
            self._init_keyboard_handler()
        else:
            self.logger.info("Using joystick")
            self.use_joystick = True
            self.key_states = {}
            self.last_key_states = {}
            self.wc_msg = None
            self.wc_key_map = {
                1: "R1", 2: "L1", 3: "L1+R1", 4: "start", 8: "select",
                10: "L1+select", 16: "R2", 32: "L2", 64: "F1", 128: "F2",  # F1, F2 not used in sim2sim
                256: "A", 512: "B", 768: "A+B", 1024: "X", 1280: "A+X",
                1536: "B+X", 2048: "Y", 2304: "A+Y", 2560: "B+Y", 3072: "X+Y",
                4096: "up", 4097: "R1+up", 4352: "A+up", 4608: "B+up", 5120: "X+up",
                6144: "Y+up", 4104: "select+up", 8192: "right", 8193: "R1+right", 8200: "select+right",
                8448: "A+right", 8704: "B+right", 9216: "X+right", 10240: "Y+right", 16384: "down",
                16385: "R1+down", 16392: "select+down", 16640: "A+down", 16896: "B+down", 17408: "X+down",
                18432: "Y+down", 32768: "left", 32769: "R1+left", 32776: "select+left", 33024: "A+left",
                33280: "B+left", 33792: "X+left", 34816: "Y+left",
            }
            if self.sdk_type == "unitree":
                self.wireless_controller_subscriber = ChannelSubscriber(
                    "rt/wirelesscontroller", WirelessController_
                )
                self.wireless_controller_subscriber.Init(self.wireless_controller_handler, 1)
            else:
                raise NotImplementedError(f"Joystick is not supported for {self.sdk_type} SDK.")
            self.logger.info("Wireless Controller Initialized")
    
    def _init_keyboard_handler(self):
        """Initialize keyboard handler."""
        self.logger.info("Using keyboard")
        self.use_joystick = False
        # Start keyboard listener in a daemon thread
        threading.Thread(target=self.start_key_listener, daemon=True).start()
        self.logger.info("Keyboard Listener Initialized")

    def wireless_controller_handler(self, msg: WirelessController_):
        self.wc_msg = msg

    # ============================================================================
    # Policy Methods
    # ============================================================================
    
    def setup_policy(self, model_path):
        """Setup ONNX policy model."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        
        self.onnx_input_names = input_names
        self.onnx_output_names = output_names
        
        def policy_act(obs_dict):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy = policy_act
    
    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict
    
    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        self.last_policy_action = policy_action.copy()
        self.scaled_policy_action = policy_action * self.policy_action_scale
        
        return self.scaled_policy_action

    # ============================================================================
    # Observation Processing Methods
    # ============================================================================
    
    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state."""
        current_obs_buffer_dict = {}
        
        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        # Calculate projected gravity
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse_numpy(
            current_obs_buffer_dict["base_quat"], v
        )

        return current_obs_buffer_dict
    
    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary."""
        current_obs_dict = {}
        for key in self.obs_dict:
            obs_list = sorted(self.obs_dict[key])
            current_obs_dict[key] = np.concatenate(
                [current_obs_buffer_dict[obs_name] * self.obs_scales[obs_name] for obs_name in obs_list], axis=1
            )
        return current_obs_dict
    
    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        
        # Update observation buffers
        self.obs_buf_dict = {
            key: np.concatenate(
                (
                    self.obs_buf_dict[key][:, self.obs_dim_dict[key] : (self.obs_dim_dict[key] * self.history_length_dict[key])],
                    current_obs_dict[key],
                ),
                axis=1,
            )
            for key in self.obs_buf_dict
        }
        
        return {"actor_obs": self.obs_buf_dict["actor_obs"].astype(np.float32)}

    # ============================================================================
    # Control/Command Methods
    # ============================================================================
    
    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos
    
    def policy_action(self):
        """Execute policy action and send commands to robot."""
        # Get robot state using the wrapper
        robot_state_data = self.state_processor.robot_state_data
        
        # Check if robot state data is available
        if robot_state_data is None:
            self.logger.warning("Robot state data not available yet. Waiting for first message...")
            return
        
        # Determine target joint positions
        if self.get_ready_state:
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # Apply policy action
            scaled_policy_action = self.rl_inference(robot_state_data)
            if scaled_policy_action.shape[1] != self.num_dofs:
                if not self.upper_body_controller:
                    scaled_policy_action = np.concatenate(
                        [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action], axis=1
                    )
                else:
                    raise NotImplementedError("Upper body controller not implemented")
            q_target = scaled_policy_action + self.default_dof_angles
        
        # Clip target positions to motor limits
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)
        
        # Send command
        cmd_q = q_target[0]
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])
    
    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    # ============================================================================
    # Input Handler Methods
    # ============================================================================
    
    def start_key_listener(self):
        """Start keyboard listener thread."""
        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed
        
        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()
    
    def process_joystick_input(self):
        """Process joystick input and update commands using InterfaceWrapper."""
        # Handle stick input
        # Process stick
        if self.wc_msg.keys == 0:
            self.lin_vel_command[0, 1] = (
                -(self.wc_msg.lx if abs(self.wc_msg.lx) > 0.1 else 0.0) * self.stand_command[0, 0]
            )
            self.lin_vel_command[0, 0] = (self.wc_msg.ly if abs(self.wc_msg.ly) > 0.1 else 0.0) * self.stand_command[
                0, 0
            ]
            self.ang_vel_command[0, 0] = (
                -(self.wc_msg.rx if abs(self.wc_msg.rx) > 0.1 else 0.0) * self.stand_command[0, 0]
            )
        cur_key = self.wc_key_map.get(self.wc_msg.keys, None)
        self.last_key_states = self.key_states.copy()
        if cur_key:
            self.key_states[cur_key] = True
        else:
            self.key_states = dict.fromkeys(self.wc_key_map.values(), False)

        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)

    # ============================================================================
    # Button Handler Methods
    # ============================================================================
    
    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        if keycode == "]":
            self._handle_start_policy()
        elif keycode == "o":
            self._handle_stop_policy()
        elif keycode == "i":
            self._handle_init_state()
        elif keycode in ["4", "5", "6", "7", "0"]:
            self._handle_kp_control(keycode)
    
    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "start":
            self._handle_start_policy()
        elif cur_key == "B+Y":
            self._handle_stop_policy()
        elif cur_key == "A+X":
            self._handle_init_state()
        elif cur_key in ["Y+left", "Y+right", "A+left", "A+right", "A+Y"]:
            self._handle_joystick_kp_control(cur_key)

    # ============================================================================
    # Control Action Methods
    # ============================================================================
    
    def _handle_start_policy(self):
        """Handle start policy action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self.logger.info(colored("Using policy actions", "blue"))
        self.phase = 0.0
        if hasattr(self.command_sender, 'no_action'):
            self.command_sender.no_action = 0
    
    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self.logger.info("Actions set to zero")
        if hasattr(self.command_sender, 'no_action'):
            self.command_sender.no_action = 1
    
    def _handle_init_state(self):
        """Handle initialization state."""
        self.get_ready_state = True
        self.init_count = 0
        self.logger.info("Setting to init state")
        if hasattr(self.command_sender, 'no_action'):
            self.command_sender.no_action = 0
    
    def _handle_kp_control(self, keycode):
        """Handle keyboard KP control."""
        if keycode == "5":
            self.command_sender.kp_level -= 0.01
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
        self.logger.info(f"KP level: {self.command_sender.kp_level}")
    
    def _handle_joystick_kp_control(self, keycode):
        """Handle joystick KP control."""
        if keycode == "Y+left":
            self.command_sender.kp_level -= 0.1
        elif keycode == "Y+right":
            self.command_sender.kp_level += 0.1
        elif keycode == "A+left":
            self.command_sender.kp_level -= 0.01
        elif keycode == "A+right":
            self.command_sender.kp_level += 0.01
        elif keycode == "A+Y":
            self.command_sender.kp_level = 1.0
    
    def _print_control_status(self):
        """Print current control status."""
        pass

    # ============================================================================
    # Main Run Method
    # ============================================================================
    
    def run(self):
        """Main run loop for the policy."""
        try:
            while True:
                if self.use_joystick and self.wc_msg is not None:
                    self.process_joystick_input()
                self.policy_action()
                self.rate.sleep()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = BasePolicy(config, model_path)
    policy.run()
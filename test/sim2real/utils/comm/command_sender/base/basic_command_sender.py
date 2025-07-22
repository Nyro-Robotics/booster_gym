from abc import ABC, abstractmethod

import numpy as np

from sim2real.utils.robot import Robot


class BasicCommandSender(ABC):
    """Abstract base class for command sender implementations."""
    
    def __init__(self, config):
        self.config = config
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.motor_type = self.config.get("MOTOR_TYPE", "serial")
        
        # Initialize control gains
        self.kp_level = 1.0
        self.kd_level = 1.0
        self.waist_kp_level = 1.0
            
        # Initialize weak motor joint index
        self.weak_motor_joint_index = []
        if self.robot.WeakMotorJointIndex:
            for _, value in self.robot.WeakMotorJointIndex.items():
                self.weak_motor_joint_index.append(value)
            
        self.no_action = 0
        
        # Initialize SDK-specific components
        self._init_sdk_components()

    @abstractmethod
    def _init_sdk_components(self):
        """Initialize SDK-specific components. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None):
        """Send command to robot. Must be implemented by subclasses."""
        pass

    def is_weak_motor(self, motor_index):
        """Check if a motor is a weak motor."""
        return motor_index in self.weak_motor_joint_index

    def _set_motor_command(self, motor_cmd, motor_id, joint_id, cmd_q, cmd_dq, cmd_tau):
        """Set motor command for a specific motor."""
        default_q = self.robot.DEFAULT_MOTOR_ANGLES
        
        if joint_id == -1 or self.no_action:
            motor_cmd.q = default_q[motor_id]
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0
        else:
            motor_cmd.q = cmd_q[joint_id]
            motor_cmd.dq = cmd_dq[joint_id]
            motor_cmd.tau = cmd_tau[joint_id]
            motor_cmd.kp = self.robot.MOTOR_KP[motor_id] * self.kp_level
            motor_cmd.kd = self.robot.MOTOR_KD[motor_id] * self.kd_level
    
    def _fill_motor_commands(self, motor_cmd, cmd_q, cmd_dq, cmd_tau):
        """Fill motor commands for all motors."""
        joint2motor = self.robot.JOINT2MOTOR
        motor2joint = self.robot.MOTOR2JOINT
        
        for i in range(self.robot.NUM_MOTORS):
            m_id = joint2motor[i]
            j_id = motor2joint[i]
            cmd = motor_cmd[m_id]
            self._set_motor_command(cmd, m_id, j_id, cmd_q, cmd_dq, cmd_tau) 
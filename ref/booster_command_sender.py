import numpy as np

from ..base import BasicCommandSender


class BoosterCommandSender(BasicCommandSender):
    """Booster command sender implementation."""
    
    def _init_sdk_components(self):
        """Initialize Booster SDK-specific components."""
        from booster_robotics_sdk_python import (
            B1LocoClient,
            B1LowCmdPublisher,
            LowCmdType,
            MotorCmd,
            RobotMode,
        )
        from booster_robotics_sdk_python import LowCmd as LowCmd
        
        robot_type = self.config["ROBOT_TYPE"]
        
        if robot_type in ["t1_23dof", "t1_29dof"]:
            self.LowCmd = LowCmd
            self.LowCmdType = LowCmdType
            self.MotorCmd = MotorCmd
            self.lowcmd_publisher_ = B1LowCmdPublisher()
            self.client = B1LocoClient()
            self.lowcmd_publisher_.InitChannel()
            self.client.Init()
            self.InitBoosterLowCmd()
            self.create_prepare_cmd(self.low_cmd, self.config)
            self._send_cmd(self.low_cmd)
            self.client.ChangeMode(RobotMode.kCustom)
            self.dof_names = self.config["dof_names"]
            self.dof_names_parallel_mech = self.config["dof_names_parallel_mech"]
            self.parallel_mech_indexes = [self.dof_names.index(name) for name in self.dof_names_parallel_mech]
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported yet")

    def InitBoosterLowCmd(self):
        """Initialize Booster low-level command."""
        self.low_cmd = self.LowCmd()
        if self.motor_type == "serial":
            self.low_cmd.cmd_type = self.LowCmdType.SERIAL
        elif self.motor_type == "parallel":
            self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
        self.motor_cmds = [self.MotorCmd() for _ in range(self.robot.NUM_MOTORS)]
        self.low_cmd.motor_cmd = self.motor_cmds

    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None):
        """Send command to Booster robot."""
        # In booster, we need to fill the motor_cmds first
        self.low_cmd = self.LowCmd()
        if self.motor_type == "serial":
            self.low_cmd.cmd_type = self.LowCmdType.SERIAL
        elif self.motor_type == "parallel":
            self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
        else:
            raise NotImplementedError(f"Motor type {self.motor_type} is not supported yet")
        self.low_cmd.motor_cmd = self.motor_cmds
        
        motor_cmd = self.low_cmd.motor_cmd
        self._fill_motor_commands(motor_cmd, cmd_q, cmd_dq, cmd_tau)
        
        # Send command
        self.lowcmd_publisher_.Write(self.low_cmd)

    def _send_cmd(self, cmd):
        """Send command to robot."""
        self.lowcmd_publisher_.Write(cmd)

    def init_Cmd_T1(self, low_cmd):
        """Initialize T1 command."""
        low_cmd.cmd_type = self.LowCmdType.SERIAL
        motorCmds = [self.MotorCmd() for _ in range(self.robot.NUM_MOTORS)]
        low_cmd.motor_cmd = motorCmds

        num_motors = min(len(motorCmds), self.robot.NUM_MOTORS)
        for i in range(num_motors):
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            # weight is not effective in custom mode
            low_cmd.motor_cmd[i].weight = 0.0

    def create_prepare_cmd(self, low_cmd, cfg):
        """Create prepare command for T1."""
        self.init_Cmd_T1(low_cmd)
        num_motors = min(len(low_cmd.motor_cmd), len(cfg["prepare"]["stiffness"]))
        for i in range(num_motors):
            low_cmd.motor_cmd[i].kp = cfg["prepare"]["stiffness"][i]
            low_cmd.motor_cmd[i].kd = cfg["prepare"]["damping"][i]
            low_cmd.motor_cmd[i].q = cfg["prepare"]["default_qpos"][i]
        return low_cmd

    def create_first_frame_rl_cmd(self, low_cmd, cfg):
        """Create first frame RL command for T1."""
        self.init_Cmd_T1(low_cmd)
        num_motors = min(len(low_cmd.motor_cmd), len(cfg["common"]["stiffness"]))
        for i in range(num_motors):
            low_cmd.motor_cmd[i].kp = cfg["common"]["stiffness"][i]
            low_cmd.motor_cmd[i].kd = cfg["common"]["damping"][i]
            low_cmd.motor_cmd[i].q = cfg["common"]["default_qpos"][i]
        return low_cmd 
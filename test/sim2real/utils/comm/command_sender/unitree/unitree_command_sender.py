from ..base import BasicCommandSender


class UnitreeCommandSender(BasicCommandSender):
    """Unitree command sender implementation."""
    
    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.utils.crc import CRC
        
        robot_type = self.config["ROBOT_TYPE"]
        
        if (
            "g1" in robot_type
            or "h1-2" in robot_type
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" in robot_type or "go2" in robot_type:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported yet")
            
        # Initialize low command publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.InitUnitreeLowCmd()
        self.low_state = None
        self.crc = CRC()

    def InitUnitreeLowCmd(self):
        """Initialize Unitree low-level command."""
        robot_type = self.config["ROBOT_TYPE"]
        
        # Set head for h1/go2
        if robot_type == "h1" or robot_type == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        
        for i in range(self.robot.NUM_MOTORS):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            
            # Set mode for g1/h1-2
            if (
                robot_type == "g1_29dof"
                or robot_type == "h1-2_21dof"
                or robot_type == "h1-2_27dof"
            ):
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]

    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None):
        """Send command to Unitree robot."""
        motor_cmd = self.low_cmd.motor_cmd
        self._fill_motor_commands(motor_cmd, cmd_q, cmd_dq, cmd_tau)
        
        # Add CRC and send
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd) 
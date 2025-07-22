from .base import BasicCommandSender
from .unitree import UnitreeCommandSender
from .booster import BoosterCommandSender


def create_command_sender(config):
    """
    Factory function to create the appropriate command sender based on configuration.
    
    Args:
        config: Robot configuration dictionary
    
    Returns:
        An instance of the appropriate command sender class
    """
    sdk_type = config.get("SDK_TYPE", "unitree")
    
    if sdk_type == "unitree":
        return UnitreeCommandSender(config)
    elif sdk_type == "booster":
        return BoosterCommandSender(config)
    else:
        raise ValueError(f"Unsupported SDK type: {sdk_type}")


# For backward compatibility
CommandSender = create_command_sender

__all__ = [
    "BasicCommandSender",
    "UnitreeCommandSender", 
    "BoosterCommandSender",
    "create_command_sender",
    "CommandSender",  # Backward compatibility
] 
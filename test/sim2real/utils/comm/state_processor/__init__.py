from .base import BasicStateProcessor


def create_state_processor(config):
    """
    Factory function to create the appropriate state processor based on configuration.
    
    Args:
        config: Robot configuration dictionary
    
    Returns:
        An instance of the appropriate state processor class
    """
    sdk_type = config.get("SDK_TYPE", "unitree")
    
    if sdk_type == "unitree":
        from .unitree import UnitreeStateProcessor
        return UnitreeStateProcessor(config)
    elif sdk_type == "booster":
        from .booster import BoosterStateProcessor
        return BoosterStateProcessor(config)
    else:
        raise ValueError(f"Unsupported SDK type: {sdk_type}")

__all__ = [
    "BasicStateProcessor",
    "UnitreeStateProcessor", 
    "BoosterStateProcessor",
    "create_state_processor",
] 
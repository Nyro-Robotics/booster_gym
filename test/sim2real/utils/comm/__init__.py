"""
Robot communication package.

This package provides unified interfaces for robot state processing and command sending
across different robot platforms (Unitree, Booster, etc.).
"""

from .state_processor import create_state_processor
from .command_sender import create_command_sender

__all__ = [
    "create_state_processor",
    "create_command_sender",
] 
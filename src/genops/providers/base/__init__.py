"""Base provider interfaces and utilities for GenOps AI framework integrations."""

from .provider import BaseFrameworkProvider
from .detector import FrameworkDetector, FrameworkInfo, get_framework_detector, detect_frameworks, is_framework_available

__all__ = [
    "BaseFrameworkProvider", 
    "FrameworkDetector", 
    "FrameworkInfo",
    "get_framework_detector",
    "detect_frameworks", 
    "is_framework_available"
]
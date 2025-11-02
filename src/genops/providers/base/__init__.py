"""Base provider interfaces and utilities for GenOps AI framework integrations."""

from .detector import (
    FrameworkDetector,
    FrameworkInfo,
    detect_frameworks,
    get_framework_detector,
    is_framework_available,
)
from .provider import BaseFrameworkProvider

__all__ = [
    "BaseFrameworkProvider",
    "FrameworkDetector",
    "FrameworkInfo",
    "get_framework_detector",
    "detect_frameworks",
    "is_framework_available"
]

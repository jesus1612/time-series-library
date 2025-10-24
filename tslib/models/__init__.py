"""
High-level model interfaces

Provides user-friendly APIs for time series modeling:
- ARIMAModel: Main interface for ARIMA modeling
"""

from .arima_model import ARIMAModel

__all__ = [
    "ARIMAModel",
]


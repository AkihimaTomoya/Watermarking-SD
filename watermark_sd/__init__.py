"""
Watermark-enabled Stable Diffusion package
"""

from .pipeline import WatermarkStableDiffusionPipeline
from .integration import (
    WatermarkUNetWrapper,
    integrate_watermark_into_checkpoint,
    verify_watermark_integration,
    create_watermark_config
)

__version__ = "1.0.0"
__all__ = [
    "WatermarkStableDiffusionPipeline",
    "WatermarkUNetWrapper", 
    "integrate_watermark_into_checkpoint",
    "verify_watermark_integration",
    "create_watermark_config"
]

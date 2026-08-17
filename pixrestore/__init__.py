"""PixRestore training package."""
from .flow import PixelDiffusion
from .models import LightningDiT_PixelDiffusion

# Keep the old name so existing smoke tests and scripts continue to work.
MeanFlowIR_PixelDiffusion = PixelDiffusion

__all__ = [
    "LightningDiT_PixelDiffusion",
    "PixelDiffusion",
    "MeanFlowIR_PixelDiffusion",
]

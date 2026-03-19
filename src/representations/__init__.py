"""
Representation modules for transforming trajectories.
"""

from .raw_repr import RawRepresentation
from .fourier_repr import FourierRepresentation
from .amplitude_phase_repr import AmplitudePhaseRepresentation
from .pca_repr import PCARepresentation
from .real_imag_repr import RealImagRepresentation
from .autoencoder_repr import AutoencoderRepresentation
from .amplitude_phase_autoencoder_repr import AmplitudePhaseAutoencoderRepresentation
from .enhanced_real_imag_autoencoder_repr import EnhancedRealImagAutoencoderRepresentation

__all__ = [
    "RawRepresentation",
    "FourierRepresentation",
    "AmplitudePhaseRepresentation",
    "PCARepresentation",
    "RealImagRepresentation",
    "AutoencoderRepresentation",
    "AmplitudePhaseAutoencoderRepresentation",
    "EnhancedRealImagAutoencoderRepresentation",
]
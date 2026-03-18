"""
Amplitude‑phase representation for oscillatory fields.

振幅-相位表示：将振荡场分解为振幅和相位分量。

物理意义：对于具有明确振荡结构的场（如相位族），
该表示分离了振幅包络和相位信息。振幅表示能量分布，
相位表示波前位置。希尔伯特变换用于提取解析信号。

优点：对于振荡场，可能提供更平滑、更低维的表示。
缺点：对于非振荡场可能不适用，计算复杂度较高。
"""

import numpy as np
from scipy.signal import hilbert


class AmplitudePhaseRepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.
            Possible keys:
            - use_hilbert: bool (default True) use analytic signal
            - separate_channels: bool (default True) treat amplitude and phase as separate
        """
        self.config = config or {}
        self.use_hilbert = self.config.get("use_hilbert", True)
        self.separate_channels = self.config.get("separate_channels", True)
        self.field_shape = None

    def fit(self, train_data, config=None):
        """
        Fit the representation to training data.

        Parameters
        ----------
        train_data : list of ndarray
            List of 2D fields from training trajectories.
        config : dict, optional
            Additional configuration.
        """
        if config:
            self.config.update(config)
            self.use_hilbert = self.config.get("use_hilbert", True)
            self.separate_channels = self.config.get("separate_channels", True)

        if train_data:
            self.field_shape = train_data[0].shape

    def transform(self, trajectory):
        """
        Transform a trajectory into amplitude‑phase representation.

        Parameters
        ----------
        trajectory : list of ndarray
            List of 2D fields.

        Returns
        -------
        repr_trajectory : list of ndarray
            List of representation vectors.
        """
        if self.field_shape is None:
            self.field_shape = trajectory[0].shape

        repr_traj = []
        for field in trajectory:
            if field.shape != self.field_shape:
                raise ValueError(
                    f"Field shape {field.shape} does not match stored shape {self.field_shape}"
                )

            if self.use_hilbert:
                # Compute analytic signal via Hilbert transform along one axis
                analytic = hilbert(field)
                amp = np.abs(analytic)
                phase = np.angle(analytic)
            else:
                # Simple amplitude = absolute value, phase = sign (binary)
                amp = np.abs(field)
                phase = np.sign(field)

            if self.separate_channels:
                # Concatenate flattened amplitude and phase
                vec = np.concatenate([amp.flatten(), phase.flatten()])
            else:
                # Interleave amplitude and phase as complex numbers
                vec = (amp * np.exp(1j * phase)).flatten()

            repr_traj.append(vec)
        return repr_traj

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from amplitude‑phase representation.

        Parameters
        ----------
        repr_trajectory : list of ndarray
            List of representation vectors.

        Returns
        -------
        trajectory : list of ndarray
            List of 2D fields.
        """
        if self.field_shape is None:
            raise RuntimeError("Representation not fitted: field_shape unknown")

        trajectory = []
        for vec in repr_trajectory:
            if self.separate_channels:
                # Split into amplitude and phase halves
                n = vec.size // 2
                amp_flat = vec[:n]
                phase_flat = vec[n:]
                amp = amp_flat.reshape(self.field_shape)
                phase = phase_flat.reshape(self.field_shape)
                field = amp * np.cos(phase)  # assuming phase is already unwrapped
            else:
                # Complex representation
                comp = vec.reshape(self.field_shape)
                field = np.real(comp)

            trajectory.append(field)
        return trajectory

    def metadata(self):
        """
        Return metadata about the representation.

        Returns
        -------
        metadata : dict
            Dictionary with representation info.
        """
        return {
            "name": "amplitude_phase",
            "field_shape": self.field_shape,
            "use_hilbert": self.use_hilbert,
            "separate_channels": self.separate_channels,
            "config": self.config,
        }
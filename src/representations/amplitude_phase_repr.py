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
            - phase_unwrap: bool (default False) unwrap phase along frequency dimension
        """
        self.config = config or {}
        self.use_hilbert = self.config.get("use_hilbert", True)
        self.separate_channels = self.config.get("separate_channels", True)
        self.phase_unwrap = self.config.get("phase_unwrap", False)
        self.field_shape = None
        self.complex_input = False

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
            self.phase_unwrap = self.config.get("phase_unwrap", False)

        if train_data:
            self.field_shape = train_data[0].shape
            # Detect if input fields are complex
            self.complex_input = np.iscomplexobj(train_data[0])

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
            # Detect if input fields are complex
            self.complex_input = np.iscomplexobj(trajectory[0])

        # Convert trajectory to numpy array for batch processing
        if isinstance(trajectory, list):
            traj_array = np.array(trajectory)  # shape: (num_freq, H, W)
        else:
            traj_array = trajectory

        num_freq, H, W = traj_array.shape

        # Check shape consistency
        if (H, W) != self.field_shape:
            raise ValueError(
                f"Field shape {(H, W)} does not match stored shape {self.field_shape}"
            )

        # Compute amplitude and phase for all fields
        if self.complex_input:
            # Complex field: directly compute amplitude and phase
            amplitude = np.abs(traj_array)  # shape: (num_freq, H, W)
            phase = np.angle(traj_array)    # shape: (num_freq, H, W)
        elif self.use_hilbert:
            # Real field: use Hilbert transform for each field
            amplitude = np.zeros((num_freq, H, W))
            phase = np.zeros((num_freq, H, W))
            for i in range(num_freq):
                analytic = hilbert(traj_array[i])
                amplitude[i] = np.abs(analytic)
                phase[i] = np.angle(analytic)
        else:
            # Real field: simple amplitude and sign
            amplitude = np.abs(traj_array)
            phase = np.sign(traj_array)

        # Phase unwrapping along frequency dimension if enabled
        if self.phase_unwrap:
            # Unwrap phase for each spatial point independently
            # phase shape: (num_freq, H, W), we need to unwrap along axis=0
            for i in range(H):
                for j in range(W):
                    phase[:, i, j] = np.unwrap(phase[:, i, j])

        # Build representation vectors
        repr_traj = []
        for idx in range(num_freq):
            amp = amplitude[idx]
            ph = phase[idx]

            if self.separate_channels:
                # Concatenate flattened amplitude and phase
                vec = np.concatenate([amp.flatten(), ph.flatten()])
            else:
                # Interleave amplitude and phase as complex numbers
                vec = (amp * np.exp(1j * ph)).flatten()

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
                if self.complex_input:
                    # Reconstruct complex field
                    field = amp * np.exp(1j * phase)
                else:
                    # Reconstruct real field
                    field = amp * np.cos(phase)  # assuming phase is already unwrapped
            else:
                # Complex representation (vector is complex if separate_channels=False)
                comp = vec.reshape(self.field_shape)
                if self.complex_input:
                    field = comp  # keep complex
                else:
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
            "phase_unwrap": self.phase_unwrap,
            "config": self.config,
        }
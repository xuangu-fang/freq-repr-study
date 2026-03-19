"""
Fourier representation: use FFT coefficients.

傅里叶表示：使用快速傅里叶变换（FFT）系数作为表示。

物理意义：将空间域场变换到频率域，捕获不同空间频率的成分。
对于具有周期性结构的场，傅里叶表示可能更紧凑、更平滑。

优点：对平移不变，能有效表示周期性结构，可能降低维度。
缺点：对于局部特征可能不敏感，需要处理复数系数。
"""

import numpy as np


class FourierRepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.
            Possible keys:
            - keep_all: bool (default True) keep all coefficients
            - top_k: int (if keep_all False) keep top k coefficients by magnitude
        """
        self.config = config or {}
        self.field_shape = None
        self.keep_all = self.config.get("keep_all", True)
        self.top_k = self.config.get("top_k", None)
        self.complex_input = False

    def fit(self, train_data, config=None):
        """
        Fit the representation to training data.

        Parameters
        ----------
        train_data : list of ndarray
            List of 2D fields from training trajectories.
        config : dict, optional
            Additional configuration (merged with self.config).
        """
        if config:
            self.config.update(config)
            self.keep_all = self.config.get("keep_all", True)
            self.top_k = self.config.get("top_k", None)

        if train_data:
            self.field_shape = train_data[0].shape
            # Detect if input fields are complex
            self.complex_input = np.iscomplexobj(train_data[0])

        # If top_k selection is needed, compute average magnitude spectrum
        # to decide which coefficients to keep.
        if not self.keep_all and self.top_k is not None:
            self._compute_coefficient_mask(train_data)

    def _compute_coefficient_mask(self, train_data):
        """Compute mask of top_k coefficients by average magnitude."""
        # Compute average magnitude over training data
        avg_mag = np.zeros(self.field_shape, dtype=np.float64)
        for field in train_data:
            fft = np.fft.fft2(field)
            avg_mag += np.abs(fft)
        avg_mag /= len(train_data)

        # Flatten and find indices of top_k largest magnitudes
        flat_mag = avg_mag.flatten()
        top_indices = np.argpartition(flat_mag, -self.top_k)[-self.top_k:]
        self.coeff_mask = np.zeros(flat_mag.shape, dtype=bool)
        self.coeff_mask[top_indices] = True
        self.coeff_mask = self.coeff_mask.reshape(self.field_shape)

    def transform(self, trajectory):
        """
        Transform a trajectory into Fourier representation.

        Parameters
        ----------
        trajectory : list of ndarray
            List of 2D fields.

        Returns
        -------
        repr_trajectory : list of ndarray
            List of representation vectors (complex or real).
        """
        if self.field_shape is None:
            self.field_shape = trajectory[0].shape
            # Detect if input fields are complex
            self.complex_input = np.iscomplexobj(trajectory[0])

        repr_traj = []
        for field in trajectory:
            if field.shape != self.field_shape:
                raise ValueError(
                    f"Field shape {field.shape} does not match stored shape {self.field_shape}"
                )

            fft = np.fft.fft2(field)

            if self.keep_all:
                # Flatten all coefficients
                vec = fft.flatten()
            else:
                # Keep only selected coefficients
                vec = fft[self.coeff_mask]

            repr_traj.append(vec)
        return repr_traj

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from Fourier representation.

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
            if self.keep_all:
                # Reshape to original FFT shape
                fft = vec.reshape(self.field_shape)
            else:
                # Reconstruct full FFT matrix
                fft = np.zeros(self.field_shape, dtype=complex)
                fft[self.coeff_mask] = vec

            if self.complex_input:
                field = np.fft.ifft2(fft)
            else:
                field = np.real(np.fft.ifft2(fft))
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
            "name": "fourier",
            "field_shape": self.field_shape,
            "keep_all": self.keep_all,
            "top_k": self.top_k,
            "config": self.config,
        }
"""
Real‑imaginary representation for complex fields.

实部-虚部表示：将复数场分解为实部和虚部分量。

物理意义：将复数场表示为两个实数场的组合。实部和虚部在数学上是线性的，
易于处理，但物理意义不如振幅-相位表示明确。

优点：线性表示，无相位缠绕问题，实现简单。
缺点：维度是原始场的两倍，物理可解释性较差。
"""

import numpy as np


class RealImagRepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary (currently unused).
        """
        self.config = config or {}
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

        if train_data:
            self.field_shape = train_data[0].shape
            # Detect if input fields are complex
            self.complex_input = np.iscomplexobj(train_data[0])

    def transform(self, trajectory):
        """
        Transform a trajectory into real‑imaginary representation.

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

        repr_traj = []
        for field in trajectory:
            if field.shape != self.field_shape:
                raise ValueError(
                    f"Field shape {field.shape} does not match stored shape {self.field_shape}"
                )

            if self.complex_input:
                # Complex field: split into real and imaginary parts
                real_part = field.real
                imag_part = field.imag
            else:
                # Real field: real part is the field, imaginary part is zero
                real_part = field
                imag_part = np.zeros_like(field)

            # Concatenate flattened real and imaginary parts
            vec = np.concatenate([real_part.flatten(), imag_part.flatten()])
            repr_traj.append(vec)

        return repr_traj

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from real‑imaginary representation.

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

        H, W = self.field_shape
        n_pixels = H * W

        trajectory = []
        for vec in repr_trajectory:
            # Split into real and imaginary halves
            real_flat = vec[:n_pixels]
            imag_flat = vec[n_pixels:]

            real_part = real_flat.reshape(self.field_shape)
            imag_part = imag_flat.reshape(self.field_shape)

            if self.complex_input:
                # Reconstruct complex field
                field = real_part + 1j * imag_part
            else:
                # For real fields, use only the real part
                field = real_part

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
            "name": "real_imag",
            "field_shape": self.field_shape,
            "complex_input": self.complex_input,
            "config": self.config,
        }
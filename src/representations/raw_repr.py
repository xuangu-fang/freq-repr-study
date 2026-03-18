"""
Raw representation: flatten the original field.

原始表示：将原始场展平为一维向量。

物理意义：最简单的表示，保留所有原始信息，不做任何变换。
作为基准表示，用于与其他表示比较。

优点：无信息损失，实现简单，无需训练。
缺点：维度高，可能包含冗余和噪声，对频率变化可能不敏感。
"""

import numpy as np


class RawRepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary (currently unused).
        """
        self.config = config or {}
        self.field_shape = None

    def fit(self, train_data, config=None):
        """
        Fit the representation to training data.

        Parameters
        ----------
        train_data : list of ndarray
            List of 2D fields from training trajectories.
        config : dict, optional
            Additional configuration (unused for raw representation).
        """
        # Raw representation does not require fitting,
        # but we store the field shape from the first sample.
        if train_data:
            self.field_shape = train_data[0].shape

    def transform(self, trajectory):
        """
        Transform a trajectory into raw representation.

        Parameters
        ----------
        trajectory : list of ndarray
            List of 2D fields (shape (H, W)).

        Returns
        -------
        repr_trajectory : list of ndarray
            List of 1D flattened vectors.
        """
        if self.field_shape is None:
            self.field_shape = trajectory[0].shape

        repr_traj = []
        for field in trajectory:
            if field.shape != self.field_shape:
                raise ValueError(
                    f"Field shape {field.shape} does not match stored shape {self.field_shape}"
                )
            repr_traj.append(field.flatten())
        return repr_traj

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from flattened vectors.

        Parameters
        ----------
        repr_trajectory : list of ndarray
            List of 1D vectors.

        Returns
        -------
        trajectory : list of ndarray
            List of 2D fields with original shape.
        """
        if self.field_shape is None:
            raise RuntimeError("Representation not fitted: field_shape unknown")

        trajectory = []
        for vec in repr_trajectory:
            trajectory.append(vec.reshape(self.field_shape))
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
            "name": "raw",
            "field_shape": self.field_shape,
            "config": self.config,
        }
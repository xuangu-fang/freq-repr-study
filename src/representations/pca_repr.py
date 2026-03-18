"""
PCA representation: fit PCA on training trajectories and project.

PCA表示：在训练数据上拟合主成分分析，然后投影到主成分空间。

物理意义：找到数据变化最大的正交方向，实现最优线性降维。
PCA表示捕获了数据的主要变化模式，丢弃了次要变化（通常视为噪声）。

优点：提供最优的线性降维，可能显著降低维度。
缺点：需要训练数据，对数据分布敏感，是线性变换。
"""

import numpy as np
from sklearn.decomposition import PCA


class PCARepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.
            Possible keys:
            - n_components: int or float (default 0.95) number of components
            - whiten: bool (default False)
        """
        self.config = config or {}
        self.n_components = self.config.get("n_components", 0.95)
        self.whiten = self.config.get("whiten", False)
        self.field_shape = None
        self.pca = None
        self.feature_shape = None

    def fit(self, train_data, config=None):
        """
        Fit PCA on training data.

        Parameters
        ----------
        train_data : list of ndarray
            List of 2D fields from training trajectories.
        config : dict, optional
            Additional configuration.
        """
        if config:
            self.config.update(config)
            self.n_components = self.config.get("n_components", 0.95)
            self.whiten = self.config.get("whiten", False)

        if not train_data:
            return

        self.field_shape = train_data[0].shape

        # Flatten all training fields into a matrix (n_samples, n_features)
        X = np.array([field.flatten() for field in train_data])

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(X)

        # Store the shape of the transformed vectors
        if hasattr(self.pca, "components_"):
            self.feature_shape = (self.pca.n_components_,)

    def transform(self, trajectory):
        """
        Transform a trajectory into PCA representation.

        Parameters
        ----------
        trajectory : list of ndarray
            List of 2D fields.

        Returns
        -------
        repr_trajectory : list of ndarray
            List of PCA‑projected vectors.
        """
        if self.pca is None:
            raise RuntimeError("PCA representation not fitted")

        if self.field_shape is None:
            self.field_shape = trajectory[0].shape

        # Flatten fields
        X = np.array([field.flatten() for field in trajectory])
        X_transformed = self.pca.transform(X)
        return list(X_transformed)

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from PCA representation.

        Parameters
        ----------
        repr_trajectory : list of ndarray
            List of PCA‑projected vectors.

        Returns
        -------
        trajectory : list of ndarray
            List of 2D fields.
        """
        if self.pca is None:
            raise RuntimeError("PCA representation not fitted")

        X_reconstructed = self.pca.inverse_transform(repr_trajectory)
        trajectory = [vec.reshape(self.field_shape) for vec in X_reconstructed]
        return trajectory

    def metadata(self):
        """
        Return metadata about the representation.

        Returns
        -------
        metadata : dict
            Dictionary with representation info.
        """
        info = {
            "name": "pca",
            "field_shape": self.field_shape,
            "n_components": self.n_components,
            "whiten": self.whiten,
            "explained_variance_ratio": (
                self.pca.explained_variance_ratio_.tolist()
                if self.pca is not None and hasattr(self.pca, "explained_variance_ratio_")
                else None
            ),
            "config": self.config,
        }
        return info
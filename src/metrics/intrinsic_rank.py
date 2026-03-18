"""
Intrinsic rank metric.
"""

import numpy as np


def intrinsic_rank(trajectory, frequencies=None):
    """
    计算轨迹的内在秩（intrinsic rank）。

    使用参与率（participation ratio）作为有效秩的度量：
    PR = (Σ s_i)² / Σ s_i²，其中 s_i 是奇异值。

    物理意义：表征轨迹在表示空间中占据的有效维度数。
    - 值越低（low better）：表示轨迹维度越低，更容易用低维模型描述
    - 值越高：表示轨迹需要使用更多维度才能充分描述

    对于理想的低维轨迹，参与率接近1（完美一维）或较小的整数。
    对于高维噪声轨迹，参与率接近总维度数。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.
    frequencies : ndarray, optional
        Not used.

    Returns
    -------
    rank : float
        Effective rank (participation ratio).
    """
    return participation_ratio(trajectory)


def participation_ratio(trajectory):
    """
    计算参与率（participation ratio），也称为有效秩（effective rank）。

    公式：PR = (Σ s_i)² / Σ s_i²，其中 s_i 是奇异值。

    物理意义：衡量轨迹在表示空间中占据的有效维度数。
    - 值范围：1 ≤ PR ≤ min(n_samples, n_features)
    - 接近1：轨迹近似一维
    - 接近总维度：轨迹使用所有维度

    参与率是比传统矩阵秩更鲁棒的度量，对噪声更稳定。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.

    Returns
    -------
    pr : float
        Participation ratio (effective rank).
    """
    # Stack vectors into matrix (n_samples, n_features)
    X = np.array(trajectory)
    # Compute singular values
    s = np.linalg.svd(X, compute_uv=False)
    # Participation ratio
    if np.sum(s) == 0:
        return 0.0
    pr = np.sum(s) ** 2 / np.sum(s**2)
    return pr
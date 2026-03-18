"""
Local smoothness metric.
"""

import numpy as np


def local_smoothness(trajectory, frequencies):
    """
    计算轨迹的局部平滑度（local smoothness）。

    定义：S(δ) = mean ||r(w+δ) - r(w)|| / |δ|
    对于等间距频率，计算连续点对的平均变化率。

    物理意义：评估表示向量随频率变化的平滑程度。
    - 值越低（low better）：表示空间对频率变化更平滑，更容易建模
    - 值越高：表示空间对频率变化更剧烈，更难建模

    平滑的表示空间意味着频率的微小变化只引起表示向量的微小变化，
    这对于插值和外推非常重要。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.
    frequencies : ndarray
        Array of frequency/parameter values (length N).

    Returns
    -------
    smoothness : float
        Average local smoothness.
    """
    if len(trajectory) != len(frequencies):
        raise ValueError(
            f"Trajectory length {len(trajectory)} does not match frequencies length {len(frequencies)}"
        )

    N = len(trajectory)
    if N < 2:
        return 0.0

    diffs = []
    for i in range(N - 1):
        delta = frequencies[i + 1] - frequencies[i]
        if abs(delta) < 1e-12:
            continue
        diff_norm = np.linalg.norm(trajectory[i + 1] - trajectory[i])
        diffs.append(diff_norm / abs(delta))

    if not diffs:
        return 0.0

    return np.mean(diffs)


def local_smoothness_per_pair(trajectory, frequencies):
    """
    Compute smoothness for each consecutive pair.

    Returns
    -------
    pair_smoothness : list of float
        Smoothness values for each pair.
    """
    N = len(trajectory)
    results = []
    for i in range(N - 1):
        delta = frequencies[i + 1] - frequencies[i]
        if abs(delta) < 1e-12:
            results.append(0.0)
        else:
            diff_norm = np.linalg.norm(trajectory[i + 1] - trajectory[i])
            results.append(diff_norm / abs(delta))
    return results
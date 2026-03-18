"""
Interpolation error metric.
"""

import numpy as np


def interpolation_error(trajectory, frequencies):
    """
    计算轨迹的插值误差（interpolation error）。

    隐藏中间频率点，在表示空间中进行线性插值，与真实值比较误差。

    物理意义：评估表示空间对缺失频率的插值能力。
    - 值越低（low better）：表示空间对频率变化的插值性能越好
    - 值越高：表示空间对频率变化的插值性能越差

    对于平滑的低维表示，线性插值应能准确预测中间频率的表示向量。
    对于非平滑或高维表示，插值误差会较大。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.
    frequencies : ndarray
        Array of frequency values.

    Returns
    -------
    error : float
        Mean interpolation error.
    """
    return linear_interpolation_error(trajectory, frequencies)


def linear_interpolation_error(trajectory, frequencies):
    """
    线性插值误差（linear interpolation error）。

    每隔一个频率点保留，使用线性插值预测被隐藏的点，计算预测误差。

    物理意义：评估线性插值在表示空间中的准确性。
    - 值越低（low better）：线性插值效果越好，表示空间更接近线性
    - 值越高：线性插值效果差，表示空间非线性强

    该方法模拟了在实际应用中只有稀疏频率采样时，通过插值预测中间频率的能力。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.
    frequencies : ndarray
        Array of frequency values.

    Returns
    -------
    error : float
        Mean interpolation error.
    """
    N = len(trajectory)
    if N < 3:
        return 0.0

    # Leave out indices 1, 3, 5, ...
    holdout = list(range(1, N, 2))
    keep = list(range(0, N, 2))

    if len(keep) < 2 or len(holdout) == 0:
        return 0.0

    # Interpolate linearly for each holdout frequency
    errors = []
    for idx in holdout:
        # Find surrounding kept indices
        left = max([i for i in keep if i < idx], default=None)
        right = min([i for i in keep if i > idx], default=None)

        if left is None or right is None:
            continue

        # Linear interpolation weight
        t = (frequencies[idx] - frequencies[left]) / (
            frequencies[right] - frequencies[left]
        )
        interp = (1 - t) * trajectory[left] + t * trajectory[right]
        errors.append(np.linalg.norm(interp - trajectory[idx]))

    if not errors:
        return 0.0
    return np.mean(errors)
"""
Curvature metric.
"""

import numpy as np


def curvature(trajectory, frequencies=None):
    """
    计算轨迹的曲率（curvature）。

    定义：C(w_i) = ||r(w_{i+1}) - 2 r(w_i) + r(w_{i-1})||
    计算内部点 i = 1 .. N-2 的曲率。

    物理意义：评估表示向量随频率变化的曲率（二阶变化率）。
    - 值越低（low better）：表示空间对频率变化的曲率越小，更接近线性
    - 值越高：表示空间对频率变化的曲率越大，非线性程度越高

    对于线性轨迹，曲率应接近零。高曲率表示轨迹在表示空间中弯曲严重，
    这可能使得线性插值效果不佳。

    Parameters
    ----------
    trajectory : list of ndarray
        List of representation vectors.
    frequencies : ndarray, optional
        Not used in the formula (kept for compatibility).

    Returns
    -------
    curvature_values : ndarray
        Curvature at each interior point (length N-2).
    """
    N = len(trajectory)
    if N < 3:
        return np.array([])

    curvatures = []
    for i in range(1, N - 1):
        second_diff = (
            np.array(trajectory[i + 1])
            - 2 * np.array(trajectory[i])
            + np.array(trajectory[i - 1])
        )
        curvatures.append(np.linalg.norm(second_diff))

    return np.array(curvatures)


def mean_curvature(trajectory, frequencies=None):
    """
    计算轨迹的平均曲率（mean curvature）。

    返回曲率数组的平均值，作为轨迹整体曲率的标量度量。

    物理意义：评估表示空间整体非线性程度的综合指标。
    - 值越低（low better）：表示空间整体更接近线性，更容易建模
    - 值越高：表示空间非线性程度高，建模难度大

    与局部曲率相比，平均曲率提供了轨迹整体弯曲程度的单一数值总结。

    Returns
    -------
    mean_curv : float
        Average curvature over interior points.
    """
    curv = curvature(trajectory, frequencies)
    if len(curv) == 0:
        return 0.0
    return np.mean(curv)
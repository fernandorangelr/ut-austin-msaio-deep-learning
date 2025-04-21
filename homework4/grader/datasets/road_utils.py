from functools import cached_property

import numpy as np


def homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points (np.ndarray): points with shape (n, d)

    Returns:
        np.ndarray: homogeneous (n, d+1)
    """
    return np.concatenate([points, np.ones((len(points), 1))], axis=1)


def interpolate_smooth(
    points: np.ndarray,
    fixed_distance: float | None = None,
    fixed_number: int | None = None,
):
    """
    Args:
        points (np.ndarray): points with shape (n, d).
        fixed_distance (float): fixed distance between points.
        fixed_number (int): fixed number of points.
    """
    if fixed_distance is None and fixed_number is None:
        raise ValueError("Either fixed_distance or fixed_number must be provided")

    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))

    if fixed_distance is not None:
        sample = np.arange(0, cumulative[-1], fixed_distance)
    elif fixed_number is not None:
        sample = np.linspace(0, cumulative[-1], fixed_number, endpoint=False)

    return np.array([np.interp(sample, cumulative, points[:, i]) for i in range(points.shape[1])]).T


class Track:
    def __init__(
        self,
        path_distance: np.ndarray,
        path_nodes: np.ndarray,
        path_width: np.ndarray,
        interpolate: bool = True,
        fixed_distance: float = 2.0,
    ):
        """
        Args:
            path_distance (np.ndarray): distance between nodes with shape (n, 2)
            path_nodes (np.ndarray): nodes with shape (n, 2, 3)
            path_width (np.ndarray): width of the path with shape (n, 1)
        """
        self.path_distance = np.float32(path_distance)
        self.path_nodes = np.float32(path_nodes)
        self.path_width = np.float32(path_width)

        # slightly perturb for numerically stable normals
        center = path_nodes[:, 0] + 1e-5 * np.random.randn(*path_nodes[:, 0].shape)
        width = path_width

        # compute left and right track using normal
        d = np.diff(center, axis=0, append=center[:1])
        n = np.stack([-d[:, 2], np.zeros_like(d[:, 0]), d[:, 0]], axis=1)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)

        left = center + n * (width / 2)
        right = center - n * (width / 2)

        # loop around
        center = np.concatenate([center, center])
        left = np.concatenate([left, left])
        right = np.concatenate([right, right])

        # resample points so each point is fixed_distance apart
        if interpolate:
            center = interpolate_smooth(center, fixed_distance=fixed_distance)
            left = interpolate_smooth(left, fixed_distance=fixed_distance)
            right = interpolate_smooth(right, fixed_distance=fixed_distance)

        # compute new cumulative distance (n,)
        center_delta = np.diff(center, axis=0, prepend=center[:1])
        center_delta_norm = np.linalg.norm(center_delta, axis=1)
        self.center_distance = np.cumsum(center_delta_norm)

        # (n, 3) points
        self.center = center
        self.left = left
        self.right = right
        self.width = interpolate_smooth(width, fixed_number=center.shape[0])

    def get_boundaries(
        self,
        distance: float,
        n_points: int = 10,
        interpolate: bool = True,
        fixed_distance: float = 2.5,
    ) -> np.ndarray:
        idx = np.searchsorted(self.center_distance, distance, side="left")
        center = self.center[idx : idx + n_points + 1]
        width = self.width[idx : idx + n_points]

        d = np.diff(center, axis=0)
        n = np.stack([-d[:, 2], np.zeros_like(d[:, 0]), d[:, 0]], axis=1)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-7)
        left = center[:-1] + n * (width / 2)
        right = center[:-1] - n * (width / 2)

        if interpolate:
            center = interpolate_smooth(center, fixed_distance=fixed_distance)
            left = interpolate_smooth(left, fixed_distance=fixed_distance)
            right = interpolate_smooth(right, fixed_distance=fixed_distance)

        left = homogeneous(left)
        right = homogeneous(right)

        return left, right

    @cached_property
    def track(self):
        return homogeneous(self.center)

    @cached_property
    def track_left(self):
        return homogeneous(self.left)

    @cached_property
    def track_right(self):
        return homogeneous(self.right)

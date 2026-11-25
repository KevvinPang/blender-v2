"""Small image-space geometry helpers shared by contact detection and QA."""
import numpy as np


def convex_hull(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return np.empty((0, 2), dtype=float)
    points = sorted(set(map(tuple, points.tolist())))
    if len(points) < 3:
        return np.asarray(points, dtype=float)

    def cross(origin, first, second):
        return ((first[0]-origin[0])*(second[1]-origin[1])
                -(first[1]-origin[1])*(second[0]-origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def coverage_ratio(points, center, width, radius_x=.30, radius_y=.16, samples=9):
    """Fraction of an elliptical face region enclosed by the hand landmark hull."""
    if not np.isfinite(width) or width <= 1e-8 or samples < 3:
        return 0.
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        return 0.
    hull = convex_hull(points)
    if len(hull) < 3:
        return 0.
    edges = np.roll(hull, -1, axis=0) - hull
    grid = np.asarray([
        center + np.array([x*radius_x*width, y*radius_y*width])
        for x in np.linspace(-1., 1., samples)
        for y in np.linspace(-1., 1., samples)
        if x*x+y*y <= 1.
    ])
    relative = grid[:, None, :] - hull[None, :, :]
    crosses = edges[None, :, 0]*relative[:, :, 1] - edges[None, :, 1]*relative[:, :, 0]
    inside = np.all(crosses >= -1e-9, axis=1) | np.all(crosses <= 1e-9, axis=1)
    return float(np.mean(inside))

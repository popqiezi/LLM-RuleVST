from __future__ import annotations

import numpy as np

KNOT_TO_MPS = 0.514444


def sog_cog_to_velocity(
    sog_kn: np.ndarray | float,
    cog_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(sog_kn, dtype=np.float64) * KNOT_TO_MPS
    angle = np.radians(np.asarray(cog_deg, dtype=np.float64))
    east = speed * np.sin(angle)
    north = speed * np.cos(angle)
    return east, north


def velocity_to_sog_cog(
    east_mps: np.ndarray | float,
    north_mps: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    east = np.asarray(east_mps, dtype=np.float64)
    north = np.asarray(north_mps, dtype=np.float64)
    speed_kn = np.hypot(east, north) / KNOT_TO_MPS
    cog = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    return speed_kn, cog


def dcpa_tcpa(
    relative_position_xy: np.ndarray,
    relative_velocity_xy: np.ndarray,
    epsilon: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(relative_position_xy, dtype=np.float64)
    v = np.asarray(relative_velocity_xy, dtype=np.float64)
    speed_sq = np.sum(v * v, axis=-1)
    dot = np.sum(r * v, axis=-1)

    tcpa = np.zeros_like(speed_sq)
    moving = speed_sq > epsilon
    tcpa[moving] = -dot[moving] / speed_sq[moving]

    closest = r + tcpa[..., None] * v
    dcpa = np.linalg.norm(closest, axis=-1)
    return dcpa, tcpa


def angle_between_vectors(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.zeros_like(denom)
    valid = denom > epsilon
    cosine[valid] = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)

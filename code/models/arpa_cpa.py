from __future__ import annotations

import numpy as np

from scripts.common.geo_utils import CoordinateTransformer
from scripts.common.kinematics import (
    sog_cog_to_velocity,
)


def constant_velocity_extrapolation(
    final_state_lon_lat_sog_cog: np.ndarray,
    future_time_offsets_s: np.ndarray,
    metric_crs: str = "EPSG:32650",
) -> np.ndarray:
    state = np.asarray(
        final_state_lon_lat_sog_cog,
        dtype=np.float64,
    )
    offsets = np.asarray(
        future_time_offsets_s,
        dtype=np.float64,
    )
    transformer = CoordinateTransformer(
        "EPSG:4326", metric_crs
    )
    x, y = transformer.lonlat_to_xy(
        np.asarray([state[0]]),
        np.asarray([state[1]]),
    )
    ve, vn = sog_cog_to_velocity(
        np.asarray([state[2]]),
        np.asarray([state[3]]),
    )
    future_x = x[0] + offsets * ve[0]
    future_y = y[0] + offsets * vn[0]
    lon, lat = transformer.xy_to_lonlat(
        future_x, future_y
    )
    return np.column_stack([lon, lat])

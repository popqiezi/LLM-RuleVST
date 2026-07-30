from __future__ import annotations

import numpy as np
from pyproj import Transformer


class CoordinateTransformer:
    def __init__(
        self,
        geographic_crs: str = "EPSG:4326",
        metric_crs: str = "EPSG:32650",
    ) -> None:
        self.to_metric = Transformer.from_crs(
            geographic_crs, metric_crs, always_xy=True
        )
        self.to_geographic = Transformer.from_crs(
            metric_crs, geographic_crs, always_xy=True
        )

    def lonlat_to_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.to_metric.transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def xy_to_lonlat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self.to_geographic.transform(x, y)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def angular_difference_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return np.abs((a_arr - b_arr + 180.0) % 360.0 - 180.0)


def bearing_deg(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    return np.mod(np.degrees(np.arctan2(east, north)), 360.0)


def relative_bearing_deg(
    own_x: np.ndarray,
    own_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    own_cog_deg: np.ndarray,
) -> np.ndarray:
    absolute = bearing_deg(target_x - own_x, target_y - own_y)
    return (absolute - own_cog_deg + 180.0) % 360.0 - 180.0

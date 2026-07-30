import numpy as np

from scripts.common.kinematics import (
    dcpa_tcpa,
    sog_cog_to_velocity,
)


def test_velocity_direction():
    east, north = sog_cog_to_velocity(
        np.asarray([10.0]), np.asarray([90.0])
    )
    assert east[0] > 5.0
    assert abs(north[0]) < 1.0e-8


def test_dcpa_tcpa_head_on():
    relative_position = np.asarray([[1000.0, 0.0]])
    relative_velocity = np.asarray([[-10.0, 0.0]])
    dcpa, tcpa = dcpa_tcpa(
        relative_position, relative_velocity
    )
    assert abs(dcpa[0]) < 1.0e-9
    assert abs(tcpa[0] - 100.0) < 1.0e-9

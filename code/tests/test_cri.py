import numpy as np

from scripts.compute_cri_targets import calculate_cri


def config():
    return {
        "d1_m": 100.0,
        "d2_m": 900.0,
        "w1_temporal": 0.1,
        "w2_spatial": 0.9,
        "reference_time_s": 1.0,
        "minimum_relative_speed_mps": 1.0e-6,
        "minimum_tangent_magnitude": 1.0e-6,
        "exponent_minimum": -60.0,
        "exponent_maximum": 0.0,
        "clip_risk_factors_to_unit_interval": True,
    }


def test_cri_bounds():
    cri, temporal, spatial = calculate_cri(
        np.asarray([50.0, 500.0, 1000.0]),
        np.asarray([50.0, 500.0, 1000.0]),
        np.asarray([10.0, 30.0, 60.0]),
        np.asarray([5.0, 5.0, 5.0]),
        np.asarray([1.0, 1.0, 1.0]),
        config(),
    )
    assert np.all(cri >= 0.0)
    assert np.all(cri <= 1.0)
    assert cri[0] == 1.0
    assert cri[-1] == 0.0

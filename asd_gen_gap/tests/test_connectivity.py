import logging

import numpy as np
import pytest

from asd_gen_gap.data.connectivity import compute_connectivity, flatten_upper_triangle


def test_compute_connectivity_matches_hand_computable_correlation() -> None:
    timeseries = np.array(
        [
            [-1.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    connectivity = compute_connectivity(timeseries)

    assert np.diag(connectivity) == pytest.approx([0.0, 0.0, 0.0])
    assert connectivity[0, 1] == pytest.approx(np.arctanh(0.5))
    assert connectivity[1, 0] == pytest.approx(connectivity[0, 1])
    assert connectivity[0, 2] == pytest.approx(np.arctanh(-1.0 / 2.0))


def test_constant_or_nan_roi_edges_are_zero_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    timeseries = np.array(
        [
            [1.0, 7.0, 1.0],
            [2.0, 7.0, np.nan],
            [3.0, 7.0, 3.0],
        ]
    )

    with caplog.at_level(logging.WARNING):
        connectivity = compute_connectivity(timeseries)

    assert connectivity[0, 1] == 0.0
    assert connectivity[1, 2] == 0.0
    assert connectivity[0, 2] == 0.0
    assert "Setting connectivity edges to zero" in caplog.text


def test_compute_then_flatten_uses_stable_upper_triangle_names() -> None:
    timeseries = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
            [3.0, 4.0, 2.0],
            [4.0, 3.0, 1.0],
        ]
    )

    connectivity = compute_connectivity(timeseries)
    flattened = flatten_upper_triangle(connectivity)

    assert list(flattened) == ["conn_0_1", "conn_0_2", "conn_1_2"]
    assert flattened["conn_0_2"] == pytest.approx(connectivity[0, 2])

import numpy as np
import pytest
from sightline.data.frames import history_indices, stack_frames


def test_history_indices_clamp_at_the_start_of_the_drive():
    h = history_indices(np.array([0, 1, 2, 10]), (2,))
    assert h.tolist() == [[0, 0], [1, 0], [2, 0], [10, 8]]


def test_history_never_looks_into_the_future():
    idx = np.arange(100)
    h = history_indices(idx, (2, 6))
    assert np.all(h[:, 1:] <= h[:, :1])


def test_stack_frames_orders_channels_current_first():
    inputs = np.zeros((20, 66, 200, 3), np.uint8)
    for i in range(20):
        inputs[i] = i
    out = stack_frames(inputs, np.array([5, 12]), (2,))
    assert out.shape == (2, 66, 200, 6) and out.dtype == np.uint8
    assert out[0, 0, 0, 0] == 5 and out[0, 0, 0, 3] == 3
    assert out[1, 0, 0, 0] == 12 and out[1, 0, 0, 3] == 10


def test_no_lags_is_the_single_frame_input():
    inputs = np.arange(5 * 66 * 200 * 3, dtype=np.uint64).astype(np.uint8).reshape(5, 66, 200, 3)
    np.testing.assert_array_equal(stack_frames(inputs, np.array([3]), ()), inputs[[3]])


@pytest.mark.parametrize("lags", [(0,), (-2,)])
def test_lags_must_point_into_the_past(lags):
    with pytest.raises(ValueError, match="positive"):
        history_indices(np.arange(5), lags)

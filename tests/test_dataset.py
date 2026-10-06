import numpy as np
import pytest
import torch
from sightline.data.dataset import SteeringDataset


class FakeCache:
    def __init__(self, n=26):  # 10 * 25 = 250 still fits in uint8
        self.inputs = np.zeros((n, 66, 200, 3), np.uint8)
        for i in range(n):
            self.inputs[i, :, : 100] = 10 * i      # left half carries the frame number
        self.angles_deg = np.linspace(-30, 30, n).astype(np.float32)

    def __len__(self):
        return len(self.angles_deg)


def test_flip_negates_the_angle_and_flips_every_stacked_frame():
    ds = SteeringDataset(FakeCache(), np.arange(5, 25), augment="flip-only", seed=0, frame_lags=(2,))
    seen_flip = seen_plain = False
    for i in range(len(ds)):
        x, y = ds[i]
        idx = ds.indices[i]
        plain = float(FakeCache().angles_deg[idx])
        if y.item() == -plain and plain != 0:
            seen_flip = True
            assert x[0, -1, 0] == 10 * idx and x[0, -1, 3] == 10 * (idx - 2)
        else:
            seen_plain = True
            assert x[0, 0, 0] == 10 * idx and x[0, 0, 3] == 10 * (idx - 2)
    assert seen_flip and seen_plain


def test_eval_mode_returns_exact_cache_bytes():
    c = FakeCache()
    ds = SteeringDataset(c, np.arange(3, 8), augment="none", frame_lags=(2,))
    x, y = ds[0]
    assert torch.equal(x[..., :3], torch.from_numpy(c.inputs[3]))
    assert y.item() == float(c.angles_deg[3])


def test_photometric_jitter_is_identical_across_stacked_frames():
    c = FakeCache()
    c.inputs[:] = 100
    ds = SteeringDataset(c, np.arange(5, 25), augment="strong", seed=3, frame_lags=(2,))
    for i in range(len(ds)):
        x, _ = ds[i]
        a, b = x[..., :3].float(), x[..., 3:].float()
        # identical inputs + shared jitter/geometry => identical outputs except per-pixel noise
        assert (a - b).abs().mean() < 12.0


@pytest.mark.parametrize("augment", [True, False, "heavy"])
def test_augment_takes_a_recipe_name(augment):
    with pytest.raises(ValueError, match="recipe"):
        SteeringDataset(FakeCache(), np.arange(3, 8), augment=augment)

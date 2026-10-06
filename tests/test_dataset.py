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
    assert torch.equal(x[..., 3:], torch.from_numpy(c.inputs[1]))  # lag 2 behind frame 3
    assert y.item() == float(c.angles_deg[3])


def test_strong_recipe_transforms_every_stacked_frame_identically():
    c = FakeCache()
    c.inputs[:] = np.random.default_rng(99).integers(0, 256, (66, 200, 3), dtype=np.uint8)
    ds = SteeringDataset(c, np.arange(5, 25), augment="strong", seed=3, frame_lags=(2,))
    for i in range(len(ds)):
        x = ds[i][0].numpy().astype(np.float32)
        a, b = x[..., :3], x[..., 3:]
        assert (np.abs(a - b) > 40).mean() < 1e-3          # geometry identical; noise only
        assert np.abs((a - b).mean(axis=(0, 1))).max() < 1.0  # same gain/cast per channel


def _order_breaks(src: np.ndarray, out: np.ndarray) -> int:
    """Pixel pairs in a column whose output order contradicts their input order."""
    width = src.shape[1]
    keys = (src.transpose(1, 0, 2).reshape(width, -1).astype(np.int64) * 256
            + out.transpose(1, 0, 2).reshape(width, -1))
    keys.sort(axis=1)  # per column: by input value, then output value
    return int((np.diff(keys % 256, axis=1) < 0).sum())


def test_standard_recipe_never_applies_the_strong_only_operations():
    # standard = an optional flip, then increasing maps shared by every pixel of a
    # column (gain/bias, shadow band), so each output column keeps its input's
    # order. Translation, colour cast, noise and cutout all break that order.
    texture = np.random.default_rng(7).integers(0, 256, (66, 200, 3), dtype=np.uint8)
    c = FakeCache()
    c.inputs[:] = texture
    for seed in range(5):
        ds = SteeringDataset(c, np.arange(26), augment="standard", seed=seed)
        for i in range(len(ds)):
            out = ds[i][0].numpy()
            assert min(_order_breaks(texture, out), _order_breaks(texture[:, ::-1], out)) == 0


@pytest.mark.parametrize("augment", [True, False, "heavy"])
def test_augment_takes_a_recipe_name(augment):
    with pytest.raises(ValueError, match="recipe"):
        SteeringDataset(FakeCache(), np.arange(3, 8), augment=augment)

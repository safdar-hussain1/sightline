import json

import numpy as np
import pytest
import torch
from sightline.models import MODEL_SPECS, SHIPPED_LAGS, build_model


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_maps_uint8_frames_to_one_angle(name):
    m = build_model(name).eval()
    c = 3 * (1 + len(m.frame_lags))
    out = m(torch.zeros(2, 66, 200, c, dtype=torch.uint8))
    assert out.shape == (2, 1) and torch.isfinite(out).all()


def test_parameter_counts():
    assert sum(p.numel() for p in build_model("pilotnet").parameters()) == 1_595_511
    assert sum(p.numel() for p in build_model("sightlinenet-1f").parameters()) == 287_489
    extra = 24 * 3 * 5 * 5 * len(SHIPPED_LAGS)
    assert sum(p.numel() for p in build_model("sightlinenet").parameters()) == 287_489 + extra


def test_pilotnet_cannot_steer_past_half_a_turn():
    m = build_model("pilotnet").eval()
    with torch.no_grad():
        m.head.bias.fill_(1e6)
        out = m(torch.zeros(1, 66, 200, 3, dtype=torch.uint8))
    assert 179.0 < out.item() < 180.0


def test_sightlinenet_output_is_unbounded():
    m = build_model("sightlinenet-1f").eval()
    with torch.no_grad():
        m.regressor[-1].weight.zero_()
        m.regressor[-1].bias.fill_(20.0)
        out = m(torch.zeros(1, 66, 200, 3, dtype=torch.uint8))
    assert out.item() == pytest.approx(600.0)


def test_pilotnet_refuses_history_frames():
    with pytest.raises(ValueError, match="single-frame"):
        build_model("pilotnet", frame_lags=(2,))


def test_adapter_rejects_channel_counts_that_are_not_rgb_stacks():
    m = build_model("sightlinenet-1f")
    with pytest.raises(ValueError, match="NHWC"):
        m(torch.zeros(1, 66, 200, 4, dtype=torch.uint8))


def test_saliency_wrapper_matches_the_plain_model_for_every_architecture():
    from sightline.saliency import SteeringWithSaliency
    for name in MODEL_SPECS:
        torch.manual_seed(0)
        m = build_model(name).eval()
        c = 3 * (1 + len(m.frame_lags))
        x = torch.randint(0, 256, (3, 66, 200, c), dtype=torch.uint8)
        with torch.no_grad():
            a, sal = SteeringWithSaliency(m)(x)
            b = m(x)
        torch.testing.assert_close(a, b)
        assert sal.shape == (3, 66, 200) and float(sal.min()) >= 0 and float(sal.max()) <= 1


# --- runs: what training records is what loading a run restores -----------


class TinyCache:
    """30 frames whose pixels carry the frame number; angles follow it."""

    def __init__(self, n=30):
        self.inputs = np.zeros((n, 66, 200, 3), np.uint8)
        for i in range(n):
            self.inputs[i, :, : 100] = 8 * i
        self.angles_deg = np.linspace(-20, 20, n).astype(np.float32)

    def __len__(self):
        return len(self.angles_deg)


def _write_run(run, config, model):
    run.mkdir(parents=True, exist_ok=True)
    (run / "history.json").write_text(json.dumps({"config": config}))
    torch.save(model.state_dict(), run / "best.pt")


def test_a_run_reloads_with_its_recorded_frame_lags_and_dropout(tmp_path):
    from sightline.reporting import load_run_model

    config = {"model": "sightlinenet", "frame_lags": [], "dropout": 0.5, "notes": {}}
    _write_run(tmp_path / "r", config, build_model("sightlinenet", frame_lags=()))
    model, _ = load_run_model(tmp_path / "r")
    assert model.frame_lags == ()
    assert model.regressor[2].p == 0.5


def test_a_run_without_frame_lags_is_refused(tmp_path):
    from sightline.reporting import load_run_model

    config = {"model": "sightlinenet", "augment": True, "notes": {}}
    _write_run(tmp_path / "old", config, build_model("sightlinenet-1f"))
    with pytest.raises(ValueError, match="^run 'old' records no frame_lags"):
        load_run_model(tmp_path / "old")


def test_training_records_the_resolved_recipe_and_predicts_on_stacked_frames(tmp_path, monkeypatch):
    from sightline import train
    from sightline.data.frames import stack_frames
    from sightline.data.splits import SplitIndices
    from sightline.reporting import load_run_model

    monkeypatch.setattr(train, "pick_device", lambda: torch.device("cpu"))
    cache = TinyCache()
    splits = SplitIndices(train=np.arange(0, 16), val=np.arange(18, 22), test=np.arange(24, 30))
    train.train_model(cache, splits, train.TrainConfig(model="sightlinenet", epochs=1, batch_size=4), tmp_path)

    config = json.loads((tmp_path / "history.json").read_text())["config"]
    spec = MODEL_SPECS["sightlinenet"]
    assert (config["augment"], config["dropout"]) == (spec.augment, spec.dropout)
    assert config["frame_lags"] == list(SHIPPED_LAGS)

    model, _ = load_run_model(tmp_path)
    assert model.frame_lags == SHIPPED_LAGS
    pred = train.predict(model, cache, splits.test, torch.device("cpu"))
    with torch.no_grad():
        direct = model(torch.from_numpy(stack_frames(cache.inputs, splits.test, SHIPPED_LAGS)))[:, 0]
    np.testing.assert_allclose(pred, direct.numpy(), rtol=1e-5, atol=1e-5)

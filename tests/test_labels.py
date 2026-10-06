import numpy as np
import pytest
from sightline.data.labels import load_labels, repair_dropouts, _parse_timestamp


def _write(tmp_path, text):
    (tmp_path / "data.txt").write_text(text)
    return tmp_path


def test_unpadded_milliseconds_are_milliseconds():
    a = _parse_timestamp("2018-07-01 17:09:45:11")
    b = _parse_timestamp("2018-07-01 17:09:45:110")
    assert round((b - a) * 1000) == 99


def test_both_label_formats_parse(tmp_path):
    d = _write(tmp_path, "0.jpg 1.5\n1.jpg -2.25\n")
    lab = load_labels(d)
    assert lab.filenames == ["0.jpg", "1.jpg"] and not lab.has_timestamps
    np.testing.assert_allclose(lab.angles_deg, [1.5, -2.25])


def test_malformed_line_names_file_and_line(tmp_path):
    d = _write(tmp_path, "0.jpg 1.0\n1.jpg notanumber\n")
    with pytest.raises(ValueError, match=r"data.txt:2"):
        load_labels(d)


def test_mixed_timestamp_presence_is_rejected(tmp_path):
    d = _write(tmp_path, "0.jpg 1.0,2018-07-01 17:09:44:912\n1.jpg 2.0\n")
    with pytest.raises(ValueError, match="timestamps present on some lines"):
        load_labels(d)


def _labels(tmp_path, angles):
    return load_labels(_write(tmp_path, "".join(f"{i}.jpg {a}\n" for i, a in enumerate(angles))))


def test_single_frame_dropout_mid_turn_is_interpolated(tmp_path):
    lab = repair_dropouts(_labels(tmp_path, [490.0, 501.78, 0.0, 498.55, 495.0]))
    assert lab.repaired_indices.tolist() == [2]
    assert lab.angles_deg[2] == pytest.approx((501.78 + 498.55) / 2)


@pytest.mark.parametrize("angles", [
    [0.0, 0.0, 0.0],                 # genuine straight driving
    [20.0, 0.0, -20.0],              # sign change through zero is a real crossing
    [10.0, 0.0, 10.0],               # neighbours below the 15 deg threshold
    [0.0, 30.0, 30.0],               # first frame has no left neighbour
])
def test_genuine_zeros_are_left_alone(tmp_path, angles):
    lab = repair_dropouts(_labels(tmp_path, angles))
    assert lab.repaired_indices.size == 0
    np.testing.assert_array_equal(lab.angles_deg, angles)


def test_repair_does_not_mutate_input(tmp_path):
    raw = _labels(tmp_path, [100.0, 0.0, 100.0])
    repair_dropouts(raw)
    assert raw.angles_deg[1] == 0.0

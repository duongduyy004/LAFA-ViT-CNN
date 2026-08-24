import csv
import random
import re

import pytest
import torch
from PIL import Image

from favit_lsda import data
from favit_lsda.data import (
    FaceTransform,
    FrameFaceDataset,
    GroupedForgeryDataset,
    build_branch_inputs,
)


@pytest.mark.parametrize(
    ("srm", "fft", "keys"),
    [
        (False, False, ("rgb",)),
        (True, False, ("rgb", "srm")),
        (False, True, ("rgb", "fft")),
        (True, True, ("rgb", "srm", "fft")),
    ],
)
def test_build_branch_inputs_returns_exact_keys(srm, fft, keys):
    rgb = torch.linspace(-1, 1, 3 * 19 * 23).reshape(3, 19, 23)
    inputs = build_branch_inputs(rgb, srm, fft)
    assert tuple(inputs) == keys
    assert all(value.shape == (3, 19, 23) for value in inputs.values())
    assert torch.equal(inputs["rgb"], rgb)
    assert all(torch.isfinite(value).all() for value in inputs.values())


@pytest.mark.parametrize(
    ("srm", "fft"), [(True, False), (False, True), (True, True)]
)
def test_constant_artifacts_normalize_to_finite_zero_tensors(srm, fft):
    rgb = torch.full((3, 16, 16), 0.25)
    inputs = build_branch_inputs(rgb, srm, fft)
    for name, value in inputs.items():
        if name != "rgb":
            assert torch.equal(value, torch.zeros_like(value))


@pytest.mark.parametrize(
    ("srm", "fft"), [(True, False), (False, True), (True, True)]
)
def test_near_constant_artifacts_are_zeroed_instead_of_amplified(srm, fft):
    """Catches min-max normalization amplifying float noise to full range.

    Unlike the exactly-constant case above, this input varies -- just by an
    amount far below any meaningful image signal. Dividing by that span would
    stretch pure noise across [-1, 1].
    """
    torch.manual_seed(0)
    rgb = torch.full((3, 16, 16), 0.25) + torch.randn(3, 16, 16) * 1e-6
    assert not torch.equal(rgb, torch.full_like(rgb, 0.25))
    inputs = build_branch_inputs(rgb, srm, fft)
    assert all(torch.isfinite(value).all() for value in inputs.values())
    for name, value in inputs.items():
        if name != "rgb":
            assert torch.equal(value, torch.zeros_like(value))


@pytest.mark.parametrize(
    ("srm", "fft"), [(True, False), (False, True), (True, True)]
)
def test_solid_non_gray_frame_is_treated_as_constant(srm, fft):
    """Catches a per-channel-flat frame slipping past a global-range check.

    Each channel is individually constant, so there is no signal for the
    derived artifacts to pick up -- even though the channels differ from
    each other and the global min/max range is therefore large.
    """
    rgb = torch.stack(
        [
            torch.full((16, 16), 0.9),
            torch.full((16, 16), -0.9),
            torch.full((16, 16), 0.0),
        ]
    )
    inputs = build_branch_inputs(rgb, srm, fft)
    for name, value in inputs.items():
        if name != "rgb":
            assert torch.equal(value, torch.zeros_like(value))


def test_invalid_branch_input_reports_sample_path():
    with pytest.raises(ValueError, match=r"branch inputs for bad.png.*RGB tensor"):
        build_branch_inputs(torch.ones(1, 8, 8), sample_path="bad.png")
    with pytest.raises(ValueError, match=r"branch inputs for bad.png.*finite"):
        build_branch_inputs(
            torch.full((3, 8, 8), float("nan")), sample_path="bad.png"
        )


def test_disabled_builders_are_not_called(monkeypatch):
    def fail(_):
        raise AssertionError("disabled artifact builder called")

    monkeypatch.setattr(data, "_srm_artifact", fail)
    monkeypatch.setattr(data, "_fft_artifact", fail)
    assert tuple(build_branch_inputs(torch.zeros(3, 8, 8))) == ("rgb",)


def test_transform_returns_mapping_after_augmentation():
    transform = FaceTransform(32, enable_srm=True, enable_fft=True)
    assert transform.expected_branches == ("rgb", "srm", "fft")
    assert transform.expected_spatial_size == (32, 32)
    inputs = transform(
        Image.new("RGB", (40, 36), "red")
    )
    assert {name: value.shape for name, value in inputs.items()} == {
        "rgb": (3, 32, 32),
        "srm": (3, 32, 32),
        "fft": (3, 32, 32),
    }


def test_domain_shift_transform_preserves_shape_and_range():
    transform = FaceTransform(
        64,
        crop_scale_min=0.85,
        color_jitter=0.1,
        grayscale_probability=1.0,
        blur_probability=1.0,
        degradation_probability=1.0,
        jpeg_probability=1.0,
        jpeg_quality_min=30,
    )
    inputs = transform(Image.new("RGB", (80, 72), "red"))
    assert tuple(inputs) == ("rgb",)
    tensor = inputs["rgb"]
    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


class _ExplicitTransform:
    expected_branches = ("rgb", "srm")
    expected_spatial_size = (32, 32)

    def __init__(self, output):
        self.output = output

    def __call__(self, _image, *args, **kwargs):
        return self.output

    def sample_flip(self):
        return False

    def sample_crop(self):
        return (1.0, 0.0, 0.0)


def _write_frame_manifest(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "face.jpg")
    manifest = tmp_path / "frames.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerow({"path": "face.jpg", "label": "1", "video_id": "video"})
    return manifest


def test_dataset_rejects_custom_transform_without_explicit_contract(tmp_path):
    class UndeclaredTransform:
        def __call__(self, _image, *args, **kwargs):
            return {"rgb": torch.zeros(3, 32, 32)}

    manifest = _write_frame_manifest(tmp_path)
    with pytest.raises(ValueError, match=r"transform.*expected_branches"):
        FrameFaceDataset(manifest, tmp_path, UndeclaredTransform())


@pytest.mark.parametrize(
    "branches",
    [
        ("srm",),
        ("fft",),
        ("srm", "fft"),
    ],
)
def test_dataset_transform_contract_requires_rgb_first(tmp_path, branches):
    class MissingRgbTransform(_ExplicitTransform):
        expected_branches = branches

    manifest = _write_frame_manifest(tmp_path)
    with pytest.raises(
        ValueError,
        match=rf"expected_branches.*begin with 'rgb'.*got {re.escape(repr(branches))}",
    ):
        FrameFaceDataset(
            manifest,
            tmp_path,
            MissingRgbTransform(
                {name: torch.zeros(3, 32, 32) for name in branches}
            ),
        )


def test_frame_dataset_accepts_custom_transform_with_explicit_contract(tmp_path):
    manifest = _write_frame_manifest(tmp_path)
    expected = {
        "rgb": torch.zeros(3, 32, 32),
        "srm": torch.ones(3, 32, 32),
    }
    inputs, label, video_id = FrameFaceDataset(
        manifest, tmp_path, _ExplicitTransform(expected)
    )[0]
    assert tuple(inputs) == ("rgb", "srm")
    assert torch.equal(inputs["rgb"], expected["rgb"])
    assert torch.equal(inputs["srm"], expected["srm"])
    assert (label, video_id) == (1, "video")


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            {"rgb": torch.zeros(3, 32, 32), "fft": torch.zeros(3, 32, 32)},
            r"face\.jpg.*expected branches.*rgb.*srm.*rgb.*fft",
        ),
        (
            {"rgb": torch.zeros(1, 32, 32), "srm": torch.zeros(3, 32, 32)},
            r"face\.jpg.*branch 'rgb'.*observed shape \(1, 32, 32\)",
        ),
        (
            {
                "rgb": torch.zeros(3, 32, 32, dtype=torch.uint8),
                "srm": torch.zeros(3, 32, 32),
            },
            r"face\.jpg.*branch 'rgb'.*dtype torch\.uint8",
        ),
        (
            {
                "rgb": torch.zeros(3, 32, 32),
                "srm": torch.full((3, 32, 32), float("nan")),
            },
            r"face\.jpg.*branch 'srm'.*finite",
        ),
        (
            {"rgb": torch.zeros(3, 31, 32), "srm": torch.zeros(3, 31, 32)},
            r"face\.jpg.*branch 'rgb'.*observed shape \(3, 31, 32\).*3, 32, 32",
        ),
    ],
)
def test_frame_dataset_rejects_malformed_transform_output(tmp_path, output, message):
    manifest = _write_frame_manifest(tmp_path)
    dataset = FrameFaceDataset(manifest, tmp_path, _ExplicitTransform(output))
    with pytest.raises(ValueError, match=message):
        dataset[0]


def _write_group_manifest(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real.jpg")
    methods = ("Deepfakes", "Face2Face")
    rows = []
    for index, method in enumerate(methods):
        fake_name = f"fake_{index}.jpg"
        Image.new("RGB", (20, 20), "red").save(tmp_path / fake_name)
        rows.append(
            {"fake_path": fake_name, "real_path": "real.jpg", "method": method}
        )
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fake_path", "real_path", "method"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest, methods


def test_grouped_dataset_builds_canonical_real_fake_domains(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    dataset = GroupedForgeryDataset(
        manifest, tmp_path, FaceTransform(224), forgery_methods=methods
    )
    inputs, labels = dataset[0]
    assert tuple(inputs) == ("rgb",)
    assert inputs["rgb"].shape == (3, 3, 224, 224)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_grouped_dataset_returns_branch_inputs_and_shared_group_geometry(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    dataset = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(32, crop_scale_min=0.75, enable_fft=True),
        methods,
    )
    inputs, labels = dataset[0]
    assert tuple(inputs) == ("rgb", "fft")
    assert inputs["rgb"].shape == (3, 3, 32, 32)
    assert inputs["fft"].shape == (3, 3, 32, 32)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_grouped_dataset_rejects_branch_geometry_outside_transform_contract(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    transform = _ExplicitTransform(
        {
            "rgb": torch.zeros(3, 32, 32),
            "srm": torch.zeros(3, 16, 32),
        }
    )
    dataset = GroupedForgeryDataset(manifest, tmp_path, transform, methods)
    with pytest.raises(
        ValueError,
        match=r"real\.jpg.*branch 'srm'.*observed shape \(3, 16, 32\).*3, 32, 32",
    ):
        dataset[0]


def test_artifact_flags_do_not_change_augmented_rgb_rng_sequence(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    random.seed(11)
    torch.manual_seed(11)
    inputs_a, _ = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(32, color_jitter=0.2, jpeg_probability=1.0),
        methods,
    )[0]
    random.seed(11)
    torch.manual_seed(11)
    inputs_b, _ = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(
            32,
            color_jitter=0.2,
            jpeg_probability=1.0,
            enable_srm=True,
            enable_fft=True,
        ),
        methods,
    )[0]
    assert torch.equal(inputs_a["rgb"], inputs_b["rgb"])
    assert tuple(inputs_a) == ("rgb",)
    assert tuple(inputs_b) == ("rgb", "srm", "fft")


def test_frame_dataset_returns_branch_inputs_label_and_video_id(tmp_path):
    manifest = _write_frame_manifest(tmp_path)
    inputs, label, video_id = FrameFaceDataset(
        manifest, tmp_path, FaceTransform(32, enable_srm=True)
    )[0]
    assert tuple(inputs) == ("rgb", "srm")
    assert inputs["rgb"].shape == (3, 32, 32)
    assert inputs["srm"].shape == (3, 32, 32)
    assert (label, video_id) == (1, "video")


def test_incomplete_domain_groups_are_reported_and_dropped(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real_a.jpg")
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real_b.jpg")
    for name in ("a_df.jpg", "a_f2f.jpg", "b_df.jpg"):
        Image.new("RGB", (20, 20), "red").save(tmp_path / name)
    rows = [
        {"fake_path": "a_df.jpg", "real_path": "real_a.jpg", "method": "DF"},
        {"fake_path": "a_f2f.jpg", "real_path": "real_a.jpg", "method": "F2F"},
        {"fake_path": "b_df.jpg", "real_path": "real_b.jpg", "method": "DF"},
    ]
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    dataset = GroupedForgeryDataset(
        manifest, tmp_path, FaceTransform(32), forgery_methods=("DF", "F2F")
    )
    assert len(dataset) == 1
    assert dataset.dropped_incomplete_groups == 1

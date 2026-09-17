from __future__ import annotations

import csv
import io
import random
from collections.abc import Mapping
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


#: Absolute dynamic-range floor below which a signal counts as constant.
#:
#: Artifacts are min-max normalized, which divides by the observed range. An
#: input whose range is numerically meaningless -- a flat frame carrying only
#: float noise -- would otherwise have that noise stretched across the full
#: ``[-1, 1]`` output range. A machine-epsilon test is far too tight for this:
#: it only catches spans that are *exactly* degenerate, so noise of, say,
#: ``1e-6`` on a flat image still gets amplified by six orders of magnitude.
#: Anything at or below this floor normalizes to zero instead. The floor sits
#: well under any real image signal, where 8-bit quantization alone puts the
#: smallest representable step near ``8e-3`` after ``[-1, 1]`` normalization.
_CONSTANT_RANGE_TOLERANCE = 1e-4


def _is_constant(value: Tensor) -> bool:
    """Report whether ``value`` carries no meaningfully distinguishable signal.

    Checked per channel: a frame whose channels are each individually flat but
    differ from one another (e.g. a solid non-gray color) still has no signal
    for the derived artifacts to pick up on, even though its global range is
    large.
    """
    per_channel_range = value.amax(dim=(-2, -1)) - value.amin(dim=(-2, -1))
    return bool(per_channel_range.amax() <= _CONSTANT_RANGE_TOLERANCE)


def _normalize_artifact(value: Tensor) -> Tensor:
    minimum = value.amin(dim=(-2, -1), keepdim=True)
    maximum = value.amax(dim=(-2, -1), keepdim=True)
    span = maximum - minimum
    return torch.where(
        span > _CONSTANT_RANGE_TOLERANCE,
        2 * (value - minimum) / span - 1,
        torch.zeros_like(value),
    )


def _srm_artifact(rgb: Tensor) -> Tensor:
    kernel = rgb.new_tensor(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]
    ).expand(3, 1, 5, 5)
    return F.conv2d(rgb.unsqueeze(0), kernel, padding=2, groups=3).squeeze(0)


def _fft_artifact(rgb: Tensor) -> Tensor:
    return torch.fft.fftshift(
        torch.log1p(torch.abs(torch.fft.fft2(rgb))), dim=(-2, -1)
    )


def build_branch_inputs(
    rgb: Tensor,
    enable_srm: bool = False,
    enable_fft: bool = False,
    sample_path: str | Path | None = None,
) -> dict[str, Tensor]:
    description = f"branch inputs for {sample_path or '<unknown path>'}"
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"{description}: expected RGB tensor [3, H, W]")
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all():
        raise ValueError(f"{description}: RGB must be finite floating point")
    result = {"rgb": rgb}
    constant = _is_constant(rgb)
    for name, enabled, builder in (
        ("srm", enable_srm, _srm_artifact),
        ("fft", enable_fft, _fft_artifact),
    ):
        if enabled:
            artifact = torch.zeros_like(rgb) if constant else builder(rgb)
            result[name] = _normalize_artifact(artifact)
    return result


class FaceTransform:
    """Face transform with train-only perturbations aimed at codec/domain shift.

    Geometric crop parameters can be shared by every image in an LSDA group,
    while photometric and codec perturbations remain independent per image.
    Defaults intentionally reproduce the old deterministic evaluation transform.
    """

    def __init__(
        self,
        image_size: int = 224,
        horizontal_flip: float = 0.0,
        crop_scale_min: float = 1.0,
        color_jitter: float = 0.0,
        grayscale_probability: float = 0.0,
        blur_probability: float = 0.0,
        degradation_probability: float = 0.0,
        jpeg_probability: float = 0.0,
        jpeg_quality_min: int = 40,
        enable_srm: bool = False,
        enable_fft: bool = False,
    ) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.crop_scale_min = float(crop_scale_min)
        self.grayscale_probability = float(grayscale_probability)
        self.blur_probability = float(blur_probability)
        self.degradation_probability = float(degradation_probability)
        self.jpeg_probability = float(jpeg_probability)
        self.jpeg_quality_min = int(jpeg_quality_min)
        self.enable_srm = bool(enable_srm)
        self.enable_fft = bool(enable_fft)
        self.expected_branches = (
            "rgb",
            *(("srm",) if self.enable_srm else ()),
            *(("fft",) if self.enable_fft else ()),
        )
        self.expected_spatial_size = (self.image_size, self.image_size)
        probabilities = (
            horizontal_flip,
            grayscale_probability,
            blur_probability,
            degradation_probability,
            jpeg_probability,
        )
        if not all(0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("crop_scale_min must be in (0, 1]")
        if not 1 <= self.jpeg_quality_min <= 100:
            raise ValueError("jpeg_quality_min must be in [1, 100]")
        jitter = float(color_jitter)
        self.color_jitter = (
            ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(jitter / 4.0, 0.1),
            )
            if jitter > 0.0
            else None
        )

    def sample_flip(self) -> bool:
        return random.random() < self.horizontal_flip

    def sample_crop(self) -> tuple[float, float, float]:
        scale = random.uniform(self.crop_scale_min, 1.0)
        return scale, random.random(), random.random()

    def _resize_crop(
        self, image: Image.Image, crop: tuple[float, float, float]
    ) -> Image.Image:
        scale, vertical_position, horizontal_position = crop
        width, height = image.size
        side = max(1, round(min(width, height) * scale))
        top = round((height - side) * vertical_position)
        left = round((width - side) * horizontal_position)
        return TF.resized_crop(
            image,
            top,
            left,
            side,
            side,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _apply_degradation(self, image: Image.Image) -> Image.Image:
        if random.random() < self.blur_probability:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.1, 2.0))
        if random.random() < self.degradation_probability:
            ratio = random.uniform(0.45, 0.9)
            reduced_size = max(16, round(self.image_size * ratio))
            image = TF.resize(
                image,
                [reduced_size, reduced_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=random.choice(
                    (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)
                ),
                antialias=True,
            )
        if random.random() < self.jpeg_probability:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=random.randint(self.jpeg_quality_min, 100),
                subsampling=random.choice((0, 1, 2)),
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                image = encoded.convert("RGB").copy()
        return image

    def __call__(
        self,
        image: Image.Image,
        flip: bool = False,
        crop: tuple[float, float, float] | None = None,
        sample_path: str | Path | None = None,
    ) -> dict[str, Tensor]:
        if image.mode != "RGB":
            raise ValueError(f"expected RGB image: {sample_path or '<unknown path>'}")
        image = self._resize_crop(image, crop or self.sample_crop())
        if flip:
            image = TF.hflip(image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if random.random() < self.grayscale_probability:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = self._apply_degradation(image)
        rgb = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
        return build_branch_inputs(
            rgb, self.enable_srm, self.enable_fft, sample_path
        )


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _transform_contract(transform) -> tuple[tuple[str, ...], tuple[int, int]]:
    branches = getattr(transform, "expected_branches", None)
    spatial_size = getattr(transform, "expected_spatial_size", None)
    if not isinstance(branches, (tuple, list)):
        raise ValueError(
            "transform must expose expected_branches as explicit branch metadata"
        )
    branches = tuple(branches)
    canonical_layouts = (
        ("rgb",),
        ("rgb", "srm"),
        ("rgb", "fft"),
        ("rgb", "srm", "fft"),
    )
    if branches not in canonical_layouts:
        raise ValueError(
            "transform expected_branches must begin with 'rgb' and exactly "
            f"follow canonical branch order, got {branches!r}"
        )
    if (
        not isinstance(spatial_size, (tuple, list))
        or len(spatial_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in spatial_size
        )
    ):
        raise ValueError(
            "transform must expose expected_spatial_size as two positive integers"
        )
    return branches, (spatial_size[0], spatial_size[1])


def _validate_transform_output(
    output,
    expected_branches: tuple[str, ...],
    expected_spatial_size: tuple[int, int],
    sample_path: Path,
) -> dict[str, Tensor]:
    if not isinstance(output, Mapping):
        raise ValueError(
            f"transform output for {sample_path}: expected a branch mapping, "
            f"got {type(output).__name__}"
        )
    observed_branches = tuple(output)
    if observed_branches != expected_branches:
        raise ValueError(
            f"transform output for {sample_path}: expected branches "
            f"{expected_branches}, got {observed_branches}"
        )
    expected_shape = (3, *expected_spatial_size)
    for name in expected_branches:
        value = output[name]
        if not isinstance(value, Tensor):
            raise ValueError(
                f"transform output for {sample_path}: branch {name!r} observed "
                f"shape <not a tensor> and dtype {type(value).__name__}; expected "
                f"finite floating tensor {list(expected_shape)}"
            )
        shape = tuple(value.shape)
        if shape != expected_shape:
            raise ValueError(
                f"transform output for {sample_path}: branch {name!r} observed "
                f"shape {shape} and dtype {value.dtype}; expected finite floating "
                f"tensor {list(expected_shape)}"
            )
        if not value.is_floating_point():
            raise ValueError(
                f"transform output for {sample_path}: branch {name!r} observed "
                f"shape {shape} and dtype {value.dtype}; expected finite floating "
                f"tensor {list(expected_shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"transform output for {sample_path}: branch {name!r} must be "
                f"finite; observed shape {shape} and dtype {value.dtype}"
            )
    return dict(output)


class GroupedForgeryDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    """Build LSDA groups in canonical order: real, then each forgery method."""

    REQUIRED_COLUMNS = {"fake_path", "real_path", "method"}

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        transform: FaceTransform,
        forgery_methods: tuple[str, ...] = (
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ),
    ) -> None:
        rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - rows[0].keys()
        if missing:
            raise ValueError(f"LSDA manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform
        self.expected_branches, self.expected_spatial_size = _transform_contract(
            transform
        )
        self.forgery_methods = tuple(forgery_methods)
        canonical = {method.casefold(): method for method in self.forgery_methods}
        grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            method = canonical.get(row["method"].casefold())
            if method is not None:
                grouped[row["real_path"]][method].append(row)
        self.groups = [
            (real_path, method_rows)
            for real_path, method_rows in grouped.items()
            if all(method_rows[method] for method in self.forgery_methods)
        ]
        self.dropped_incomplete_groups = len(grouped) - len(self.groups)
        if not self.groups:
            raise ValueError(
                "manifest has no complete real + forgery-method groups for LSDA"
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        real_path, method_rows = self.groups[index]
        real_source = _resolve(self.data_root, real_path)
        with Image.open(real_source) as image:
            real = image.copy()
        selected_fakes = []
        for method in self.forgery_methods:
            row = random.choice(method_rows[method])
            fake_source = _resolve(self.data_root, row["fake_path"])
            with Image.open(fake_source) as image:
                selected_fakes.append((image.copy(), fake_source))
        flip = self.transform.sample_flip()
        crop = self.transform.sample_crop()
        samples = [self.transform(real, flip, crop, real_source)] + [
            self.transform(image, flip, crop, fake_source)
            for image, fake_source in selected_fakes
        ]
        sample_paths = [real_source, *(source for _, source in selected_fakes)]
        samples = [
            _validate_transform_output(
                sample,
                self.expected_branches,
                self.expected_spatial_size,
                sample_path,
            )
            for sample, sample_path in zip(samples, sample_paths)
        ]
        inputs = {
            name: torch.stack([sample[name] for sample in samples])
            for name in self.expected_branches
        }
        domain_labels = torch.arange(len(self.forgery_methods) + 1, dtype=torch.long)
        return inputs, domain_labels


class FrameFaceDataset(Dataset[tuple[dict[str, Tensor], int, str]]):
    REQUIRED_COLUMNS = {"path", "label", "video_id"}

    def __init__(
        self, manifest: str | Path, data_root: str | Path, transform: FaceTransform
    ) -> None:
        self.rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - self.rows[0].keys()
        if missing:
            raise ValueError(f"frame manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform
        self.expected_branches, self.expected_spatial_size = _transform_contract(
            transform
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], int, str]:
        row = self.rows[index]
        source = _resolve(self.data_root, row["path"])
        with Image.open(source) as image:
            branch_inputs = self.transform(image.copy(), sample_path=source)
        branch_inputs = _validate_transform_output(
            branch_inputs,
            self.expected_branches,
            self.expected_spatial_size,
            source,
        )
        return branch_inputs, int(row["label"]), row["video_id"]


class BinaryFrameDataset(Dataset[tuple[dict[str, Tensor], int, str]]):
    """Load independent binary frames for conventional (non-LSDA) training.

    The input manifest may either be a regular frame manifest with
    ``path,label,video_id`` columns, or the existing FF++ pair manifest with
    ``fake_path,real_path,method`` columns. Pair manifests are flattened into
    independent samples and duplicate real frames are removed.
    """

    FRAME_COLUMNS = {"path", "label", "video_id"}
    PAIR_COLUMNS = {"fake_path", "real_path", "method"}

    def __init__(
        self, manifest: str | Path, data_root: str | Path, transform: FaceTransform
    ) -> None:
        rows = _read_manifest(manifest)
        columns = rows[0].keys()
        if self.FRAME_COLUMNS <= columns:
            records = [
                (row["path"], int(row["label"]), row["video_id"])
                for row in rows
            ]
        elif self.PAIR_COLUMNS <= columns:
            labels_by_path: dict[str, int] = {}
            for row in rows:
                for path, label in (
                    (row["real_path"], 0),
                    (row["fake_path"], 1),
                ):
                    previous = labels_by_path.setdefault(path, label)
                    if previous != label:
                        raise ValueError(
                            f"training path has conflicting labels: {path}"
                        )
            records = [
                (path, label, path) for path, label in labels_by_path.items()
            ]
        else:
            raise ValueError(
                "binary training manifest must contain either "
                "path,label,video_id or fake_path,real_path,method columns"
            )
        labels = [label for _, label, _ in records]
        if not labels or set(labels) != {0, 1}:
            raise ValueError(
                "binary training manifest must contain both real (0) and fake (1)"
            )
        if any(label not in (0, 1) for label in labels):
            raise ValueError("binary training labels must be 0 or 1")
        self.records = records
        self.labels = tuple(labels)
        self.data_root = Path(data_root)
        self.transform = transform
        self.expected_branches, self.expected_spatial_size = _transform_contract(
            transform
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], int, str]:
        relative_path, label, video_id = self.records[index]
        source = _resolve(self.data_root, relative_path)
        with Image.open(source) as image:
            branch_inputs = self.transform(
                image.copy(),
                flip=self.transform.sample_flip(),
                crop=self.transform.sample_crop(),
                sample_path=source,
            )
        branch_inputs = _validate_transform_output(
            branch_inputs,
            self.expected_branches,
            self.expected_spatial_size,
            source,
        )
        return branch_inputs, label, video_id

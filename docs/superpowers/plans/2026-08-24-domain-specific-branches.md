# Domain-Specific RGB, SRM, and FFT Branches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace ArtifactCNN with mandatory RGB/FA-ViT plus optional SRM/Xception and FFT/MobileNetV3-Small encoders, joined by fixed-slot late fusion.

**Architecture:** FaceTransform emits a mapping containing RGB and only enabled forensic tensors. FA-ViT remains the LSDA source; ImageNet-pretrained Xception and MobileNetV3-Small project SRM and FFT features to the ViT width; fixed [RGB, SRM, FFT] concatenation feeds the detector and FAL.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, torchvision 0.16+, timm 1.0+, pytest, PyYAML

**Spec:** docs/superpowers/specs/2026-08-24-domain-specific-branches-design.md

## Global Constraints

- RGB/FA-ViT is mandatory; only SRM and FFT may be disabled.
- SRM uses timm xception; FFT uses mobilenetv3_small_100.
- Both forensic encoders default to ImageNet pretrained and are fully fine-tuned.
- Xception receives only three-channel SRM; MobileNet receives only three-channel FFT log-magnitude.
- Fusion slots always have [RGB, SRM, FFT] order and total width 3 * embed_dim.
- Disabled branches are not constructed or executed and contribute zero slots.
- Remove Wavelet, ArtifactCNN, artifact_mode, and cnn_in_channels.
- LSDA teachers, distillation, and domain invariance remain FA-ViT-only.
- Tests must not download weights.
- Preserve the user's untracked wiki/ directory.

---

### Task 1: Define the branch configuration contract

**Files:**
- Modify: favit_lsda/config.py:1-85
- Modify: tests/test_config.py:1-40

**Interfaces:**
- Produces: BranchConfig
- Produces: resolve_branch_config(model_config: dict[str, Any]) -> BranchConfig
- Produces: BranchConfig.enabled_branches -> tuple[str, ...]

- [ ] **Step 1: Write failing config tests**

~~~python
from favit_lsda.config import resolve_branch_config, validate_model_config

def test_branch_config_defaults_to_rgb_only():
    value = resolve_branch_config({})
    assert value.enabled_branches == ("rgb",)
    assert value.srm_backbone == "xception"
    assert value.fft_backbone == "mobilenetv3_small_100"
    assert value.forensic_pretrained is True

def test_branch_config_enables_both_forensic_paths():
    value = resolve_branch_config(
        {"enable_srm_branch": True, "enable_fft_branch": True}
    )
    assert value.enabled_branches == ("rgb", "srm", "fft")

@pytest.mark.parametrize("key", ["artifact_mode", "cnn_in_channels"])
def test_legacy_artifact_fields_are_rejected(key):
    with pytest.raises(ValueError, match=rf"obsolete.*{key}"):
        validate_model_config({key: "legacy"})

@pytest.mark.parametrize(
    ("field", "value"),
    [("srm_backbone", "resnet50"), ("fft_backbone", "freqnet")],
)
def test_unsupported_backbones_are_rejected(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*{value}"):
        validate_model_config({field: value})
~~~

- [ ] **Step 2: Run the tests to prove the API is missing**

Run: rtk pytest tests/test_config.py -q

Expected: FAIL because BranchConfig and resolve_branch_config do not exist.

- [ ] **Step 3: Implement immutable resolution and validation**

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class BranchConfig:
    enable_srm: bool
    enable_fft: bool
    srm_backbone: str
    fft_backbone: str
    forensic_pretrained: bool

    @property
    def enabled_branches(self) -> tuple[str, ...]:
        return (
            "rgb",
            *(("srm",) if self.enable_srm else ()),
            *(("fft",) if self.enable_fft else ()),
        )

def resolve_branch_config(model_config: dict[str, Any]) -> BranchConfig:
    legacy = sorted({"artifact_mode", "cnn_in_channels"} & model_config.keys())
    if legacy:
        raise ValueError(
            f"obsolete model field(s) {legacy}; use enable_srm_branch and "
            "enable_fft_branch"
        )
    value = BranchConfig(
        enable_srm=bool(model_config.get("enable_srm_branch", False)),
        enable_fft=bool(model_config.get("enable_fft_branch", False)),
        srm_backbone=str(model_config.get("srm_backbone", "xception")),
        fft_backbone=str(
            model_config.get("fft_backbone", "mobilenetv3_small_100")
        ),
        forensic_pretrained=bool(model_config.get("forensic_pretrained", True)),
    )
    if value.srm_backbone != "xception":
        raise ValueError(f"unsupported srm_backbone: {value.srm_backbone!r}")
    if value.fft_backbone != "mobilenetv3_small_100":
        raise ValueError(f"unsupported fft_backbone: {value.fft_backbone!r}")
    return value
~~~

Make validate_model_config call this function. In build_model_from_config,
resolve once and pass enable_srm_branch, enable_fft_branch, srm_backbone,
fft_backbone, and forensic_pretrained to create_favit_lsda. When its explicit
pretrained override is false, override all three encoder-pretraining flags so
resume, evaluation, and tests never download weights.

- [ ] **Step 4: Run tests and commit**

Run: rtk pytest tests/test_config.py -q

Expected: PASS.

~~~bash
git add favit_lsda/config.py tests/test_config.py
git commit -m "refactor: define forensic branch configuration"
~~~

---

### Task 2: Emit branch-input mappings from the data pipeline

**Files:**
- Modify: favit_lsda/data.py:20-378
- Modify: tests/test_data.py:1-226

**Interfaces:**
- Produces: build_branch_inputs(rgb: Tensor, enable_srm: bool = False,
  enable_fft: bool = False, sample_path: str | Path | None = None)
  -> dict[str, Tensor]
- Changes: FaceTransform.__call__ -> dict[str, Tensor]
- Changes: GroupedForgeryDataset.__getitem__ -> tuple[dict[str, Tensor], Tensor]
- Changes: FrameFaceDataset.__getitem__ -> tuple[dict[str, Tensor], int, str]

- [ ] **Step 1: Replace mode/width tests with failing mapping tests**

~~~python
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

def test_disabled_builders_are_not_called(monkeypatch):
    def fail(_):
        raise AssertionError("disabled artifact builder called")
    monkeypatch.setattr(data, "_srm_artifact", fail)
    monkeypatch.setattr(data, "_fft_artifact", fail)
    assert tuple(build_branch_inputs(torch.zeros(3, 8, 8))) == ("rgb",)

def test_transform_returns_mapping_after_augmentation():
    inputs = FaceTransform(32, enable_srm=True, enable_fft=True)(
        Image.new("RGB", (40, 36), "red")
    )
    assert {name: value.shape for name, value in inputs.items()} == {
        "rgb": (3, 32, 32),
        "srm": (3, 32, 32),
        "fft": (3, 32, 32),
    }
~~~

Update grouped dataset assertions to [domains, 3, H, W] per key and frame
dataset assertions to the same exact key order.

- [ ] **Step 2: Run tests and verify failure**

Run: rtk pytest tests/test_data.py -q

Expected: FAIL because build_branch_inputs and new FaceTransform flags are absent.

- [ ] **Step 3: Remove Wavelet/layout code and build only enabled artifacts**

Delete _ARTIFACT_LAYOUTS, artifact_channels, resolve_artifact_config,
_wavelet_artifact, and build_cnn_input. Retain SRM, FFT, constant detection,
and normalization. Add:

~~~python
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
~~~

- [ ] **Step 4: Return and stack mappings**

FaceTransform stores enable_srm and enable_fft and returns build_branch_inputs
after existing augmentation and RGB normalization. GroupedForgeryDataset uses:

~~~python
samples = [self.transform(real, flip, crop, real_source)] + [
    self.transform(image, flip, crop, source)
    for image, source in selected_fakes
]
keys = tuple(samples[0])
if any(tuple(sample) != keys for sample in samples[1:]):
    raise ValueError("inconsistent branch keys within LSDA group")
inputs = {
    name: torch.stack([sample[name] for sample in samples])
    for name in keys
}
return inputs, domain_labels
~~~

FrameFaceDataset returns branch_inputs, label, video_id.

- [ ] **Step 5: Run tests and commit**

Run: rtk pytest tests/test_data.py -q

Expected: PASS with no wavelet tests.

~~~bash
git add favit_lsda/data.py tests/test_data.py
git commit -m "refactor: emit RGB SRM and FFT branch mappings"
~~~

---

### Task 3: Add projected forensic encoders

**Files:**
- Create: favit_lsda/forensic.py
- Create: tests/test_forensic.py

**Interfaces:**
- Produces: ProjectedForensicEncoder(model_name: str, embed_dim: int,
  pretrained: bool, dropout: float)
- Produces: forward(images: Tensor) -> Tensor[B, embed_dim]

- [ ] **Step 1: Write tests with a fake timm backbone**

~~~python
class TinyBackbone(torch.nn.Module):
    num_features = 5
    def forward(self, images):
        return images.mean((-2, -1)).mean(1, keepdim=True).expand(-1, 5)

@pytest.mark.parametrize("name", ["xception", "mobilenetv3_small_100"])
def test_encoder_builds_requested_pretrained_model(monkeypatch, name):
    calls = []
    def fake_create(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return TinyBackbone()
    monkeypatch.setattr("favit_lsda.forensic.timm.create_model", fake_create)
    model = ProjectedForensicEncoder(name, 7, True, 0.0)
    output = model(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 7)
    assert calls == [
        (name, {"pretrained": True, "num_classes": 0, "global_pool": "avg"})
    ]
    output.square().mean().backward()
    assert model.project[0].weight.grad is not None
~~~

Also test rank, three-channel, finite floating-point, and rank-2 pooled-output
validation.

- [ ] **Step 2: Run tests to prove the module is missing**

Run: rtk pytest tests/test_forensic.py -q

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement the wrapper**

~~~python
class ProjectedForensicEncoder(nn.Module):
    def __init__(self, model_name, embed_dim, pretrained, dropout):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.project = nn.Sequential(
            nn.Linear(int(self.backbone.num_features), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"{self.model_name} expects [B, 3, H, W], got {tuple(images.shape)}"
            )
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError(f"{self.model_name} input must be finite floating point")
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError(
                f"{self.model_name} pooled output must be rank 2, got {features.shape}"
            )
        return self.project(features)
~~~

- [ ] **Step 4: Run and commit**

Run: rtk pytest tests/test_forensic.py -q

Expected: PASS.

~~~bash
git add favit_lsda/forensic.py tests/test_forensic.py
git commit -m "feat: add projected forensic encoders"
~~~

---

### Task 4: Implement fixed-slot multibranch fusion

**Files:**
- Modify: favit_lsda/model.py:212-606
- Modify: tests/test_model.py:1-166

**Interfaces:**
- Changes: forward(inputs: Mapping[str, Tensor], return_features=False)
- Changes: forward_group(grouped_inputs: Mapping[str, Tensor])
- Produces properties: enabled_branches, srm_backbone_name,
  fft_backbone_name, fusion_name

- [ ] **Step 1: Write failing tests for four ablations and zero slots**

Monkeypatch ProjectedForensicEncoder with a tiny differentiable encoder. Test
(False,False), (True,False), (False,True), and (True,True). For every case,
forward_group must return [G, domains, embed_dim] features and backpropagate to
each enabled projection. Capture late_fusion input for RGB-only:

~~~python
assert captured.shape[-1] == model.embed_dim * 3
assert torch.count_nonzero(captured[:, model.embed_dim:]) == 0
~~~

Retain teacher-free inference and FA-ViT-only invariance tests, changing calls
to mappings. Add tests for missing/unexpected keys and geometry mismatch.

- [ ] **Step 2: Run focused tests**

Run: rtk pytest tests/test_model.py -q

Expected: FAIL because ArtifactCNN and positional RGB/CNN arguments remain.

- [ ] **Step 3: Construct optional encoders and fixed fusion**

Remove ArtifactCNN and its constructor fields. Add the five branch constructor
arguments and:

~~~python
self.enabled_branches = (
    "rgb",
    *(("srm",) if enable_srm_branch else ()),
    *(("fft",) if enable_fft_branch else ()),
)
self.srm_backbone_name = srm_backbone if enable_srm_branch else None
self.fft_backbone_name = fft_backbone if enable_fft_branch else None
self.fusion_name = "fixed_slot_concat"
self.srm_encoder = (
    ProjectedForensicEncoder(
        srm_backbone, self.embed_dim, forensic_pretrained, feature_dropout
    ) if enable_srm_branch else None
)
self.fft_encoder = (
    ProjectedForensicEncoder(
        fft_backbone, self.embed_dim, forensic_pretrained, feature_dropout
    ) if enable_fft_branch else None
)
self.late_fusion = nn.Sequential(
    nn.Linear(self.embed_dim * 3, self.embed_dim),
    nn.LayerNorm(self.embed_dim),
    nn.GELU(),
    nn.Dropout(feature_dropout),
)
~~~

Include only non-None encoders in the always-trainable module list.

- [ ] **Step 4: Validate input mappings before any encoder runs**

Add _validate_inputs(inputs, grouped) that requires exactly enabled_branches,
rank 4 or 5, three channels, floating finite tensors, and matching
batch/group/domain/spatial geometry.

- [ ] **Step 5: Fuse three fixed slots**

~~~python
rgb_features, student_maps = self._student_features(cls_features, patch_maps)
zero = torch.zeros_like(rgb_features)
srm_features = (
    self.srm_encoder(flat_inputs["srm"]) if self.srm_encoder else zero
)
fft_features = (
    self.fft_encoder(flat_inputs["fft"]) if self.fft_encoder else zero
)
features = self.late_fusion(
    torch.cat((rgb_features, srm_features, fft_features), dim=1)
)
return features, rgb_features, student_maps
~~~

forward encodes inputs["rgb"]. forward_group flattens every enabled rank-5
tensor before fusion and leaves existing teacher/LSDA code unchanged. Feed only
rgb_features to gradient reversal. Thread new arguments through
create_favit_lsda.

- [ ] **Step 6: Run and commit**

Run: rtk pytest tests/test_model.py tests/test_lsda.py -q

Expected: PASS.

~~~bash
git add favit_lsda/model.py tests/test_model.py
git commit -m "feat: fuse RGB SRM and FFT encoder features"
~~~

---

### Task 5: Wire mappings through engine, train, and evaluation

**Files:**
- Modify: favit_lsda/engine.py:14-147
- Modify: train.py:83-455
- Modify: favit_lsda/evaluation.py:43-169
- Modify: tests/test_engine.py:1-112

**Interfaces:**
- Produces: move_branch_inputs(inputs: Mapping[str, Tensor], device) -> dict
- Consumes mapping datasets and mapping model methods.

- [ ] **Step 1: Rewrite engine integration test**

Use FaceTransform(224, enable_srm=True, enable_fft=True), both model branch
flags, forensic_pretrained=False, and fake forensic encoders. Keep one real
DataLoader training step and frame-level evaluation assertions.

- [ ] **Step 2: Run to verify tuple unpacking fails**

Run: rtk pytest tests/test_engine.py -q

Expected: FAIL where engine still unpacks RGB and CNN separately.

- [ ] **Step 3: Move and forward mappings**

~~~python
def move_branch_inputs(inputs, device):
    return {
        name: value.to(device, non_blocking=True)
        for name, value in inputs.items()
    }

for grouped_inputs, domain_labels in loader:
    grouped_inputs = move_branch_inputs(grouped_inputs, device)
    output = model.forward_group(grouped_inputs)

for inputs, labels, video_ids in loader:
    inputs = move_branch_inputs(inputs, device)
    logits = model(inputs)
~~~

- [ ] **Step 4: Build branch-aware transforms**

In train.py resolve BranchConfig once and pass enable_srm/enable_fft to train
and clean transforms. In evaluation.py derive flags from model.enabled_branches.
Remove all resolve_artifact_config and model.artifact_mode usage.

- [ ] **Step 5: Give every pretrained backbone the reduced LR**

~~~python
backbone_prefixes = (
    "backbone.",
    "srm_encoder.backbone.",
    "fft_encoder.backbone.",
)
is_backbone = name.startswith(backbone_prefixes)
~~~

Add a test proving forensic backbone parameters get
learning_rate * backbone_lr_multiplier while forensic projections and fusion
get the base rate.

- [ ] **Step 6: Run and commit**

Run: rtk pytest tests/test_engine.py -q

Expected: PASS after fixtures use mappings.

~~~bash
git add favit_lsda/engine.py favit_lsda/evaluation.py train.py tests/test_engine.py
git commit -m "refactor: route branch mappings through train and evaluation"
~~~

---

### Task 6: Version multibranch checkpoints

**Files:**
- Modify: favit_lsda/checkpoints.py:1-66
- Modify: favit_lsda/evaluation.py:43-83
- Modify: train.py:42-74,310-426
- Modify: tests/test_checkpoint.py:14-319

**Interfaces:**
- Produces: EXPECTED_ARCHITECTURE = "favit_lsda_multibranch"
- Produces: SUPPORTED_FORMAT_VERSION = 4
- Produces: validate_checkpoint_branches(checkpoint: dict[str, Any],
  model_config: dict[str, Any], checkpoint_path: Path) -> None
- Produces: model_branch_metadata(model) -> dict[str, Any]

- [ ] **Step 1: Write failing version-4 tests**

~~~python
def _tiny_checkpoint(model_config, state=None):
    branches = resolve_branch_config(model_config)
    return {
        "format_version": 4,
        "architecture": "favit_lsda_multibranch",
        "enabled_branches": list(branches.enabled_branches),
        "srm_backbone": branches.srm_backbone if branches.enable_srm else None,
        "fft_backbone": branches.fft_backbone if branches.enable_fft else None,
        "fusion": "fixed_slot_concat",
        "config": {"model": model_config},
        "model": state or {},
    }
~~~

Test SRM-vs-FFT mismatch, legacy favit_lsda_cnn rejection, unsupported
format versions, validation before model construction, and saved metadata.

- [ ] **Step 2: Run tests and verify old metadata fails**

Run: rtk pytest tests/test_checkpoint.py -q

Expected: FAIL because version 3 artifact metadata remains.

- [ ] **Step 3: Implement metadata and comparison**

~~~python
EXPECTED_ARCHITECTURE = "favit_lsda_multibranch"
SUPPORTED_FORMAT_VERSION = 4

def model_branch_metadata(model):
    return {
        "format_version": SUPPORTED_FORMAT_VERSION,
        "architecture": EXPECTED_ARCHITECTURE,
        "enabled_branches": list(model.enabled_branches),
        "srm_backbone": model.srm_backbone_name,
        "fft_backbone": model.fft_backbone_name,
        "fusion": model.fusion_name,
    }
~~~

validate_checkpoint_branches first checks architecture/version, then compares
enabled_branches, optional backbone names, and fusion against
resolve_branch_config(model_config). Its mismatch error includes checkpoint,
config, and path.

- [ ] **Step 4: Save metadata from the model and load strictly**

Build the saved state explicitly:

~~~python
state = {
    **model_branch_metadata(model),
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "best_auc": best_auc,
    "best_selection_auc": best_auc,
    "selection_name": selection_name,
    "selection_metrics": selection_metrics,
    "celebdf_test_metrics": None,
    "epochs_without_improvement": epochs_without_improvement,
    "random_state": capture_random_state(),
    "config": config,
}
~~~

Call the new validator before resume state loading and twice in evaluation as
today. Since old checkpoints are rejected, evaluation uses
model.load_state_dict(checkpoint["model"], strict=True) and removes legacy
missing-key allowances.

- [ ] **Step 5: Exclude all detector-specific modules from --init-favit**

~~~python
excluded_prefixes = (
    "head.",
    "srm_encoder.",
    "fft_encoder.",
    "late_fusion.",
)
~~~

Update the test with fake weights under every prefix and assert they remain
freshly initialized.

- [ ] **Step 6: Run and commit**

Run: rtk pytest tests/test_checkpoint.py tests/test_engine.py -q

Expected: PASS.

~~~bash
git add favit_lsda/checkpoints.py favit_lsda/evaluation.py train.py tests/test_checkpoint.py
git commit -m "feat: version multibranch checkpoint metadata"
~~~

---

### Task 7: Replace configs, docs, and verify

**Files:**
- Create: configs/favit_lsda_rgb.yaml
- Create: configs/favit_lsda_rgb_srm.yaml
- Create: configs/favit_lsda_rgb_fft.yaml
- Create: configs/favit_lsda_rgb_srm_fft.yaml
- Delete: all six configs/favit_lsda_cnn_*.yaml files
- Modify: configs/favit_lsda_ffpp_c23_celebdf.yaml:28-46
- Modify: run_ffpp_tests.py:9-21
- Modify: tests/test_config.py:8-40
- Modify: README.md:1-440

**Interfaces:**
- Produces four runnable, otherwise identical ablation configs.

- [ ] **Step 1: Add failing exact-config tests**

~~~python
@pytest.mark.parametrize(
    ("name", "srm", "fft"),
    [
        ("rgb", False, False),
        ("rgb_srm", True, False),
        ("rgb_fft", False, True),
        ("rgb_srm_fft", True, True),
    ],
)
def test_experiment_config_has_exact_branches(name, srm, fft):
    config = load_config(Path("configs") / f"favit_lsda_{name}.yaml")
    validate_model_config(config["model"])
    assert config["model"]["enable_srm_branch"] is srm
    assert config["model"]["enable_fft_branch"] is fft
    assert config["model"]["srm_backbone"] == "xception"
    assert config["model"]["fft_backbone"] == "mobilenetv3_small_100"
    assert config["model"]["forensic_pretrained"] is True
    assert config["output_dir"] == f"outputs/favit_lsda_{name}"
~~~

- [ ] **Step 2: Run to verify new files are absent**

Run: rtk pytest tests/test_config.py -q

Expected: FAIL with missing new config files.

- [ ] **Step 3: Create four configs and remove six old configs**

Each new model section contains:

~~~yaml
enable_srm_branch: false
enable_fft_branch: false
srm_backbone: xception
fft_backbone: mobilenetv3_small_100
forensic_pretrained: true
~~~

Change only the two booleans and output_dir between ablations; preserve seed,
data, augmentation, LSDA, loss, and optimizer settings. Add explicit RGB-only
fields to favit_lsda_ffpp_c23_celebdf.yaml. Remove both Wavelet configs.

- [ ] **Step 4: Update runner and README**

CASES contains exactly favit_lsda_rgb.yaml, favit_lsda_rgb_srm.yaml,
favit_lsda_rgb_fft.yaml, and favit_lsda_rgb_srm_fft.yaml. README documents the
three encoders, fixed-slot fusion, pretrained/full-finetune policy, RGB-only
LSDA, four commands, Wavelet removal, and version-3 migration via --init-favit.

- [ ] **Step 5: Prove obsolete contracts are absent**

Run:

~~~bash
rtk rg -n "artifact_mode|cnn_in_channels|rgb_wavelet|srm_wavelet|FreqNet" \
  favit_lsda train.py configs tests README.md run_ffpp_tests.py
~~~

Expected: matches occur only in config/checkpoint tests that deliberately
reject legacy fields and in the intentional README legacy-checkpoint migration
note; no runtime input or model path consumes them.

- [ ] **Step 6: Run focused and full verification**

~~~bash
rtk pytest tests/test_data.py tests/test_forensic.py tests/test_config.py \
  tests/test_model.py tests/test_engine.py tests/test_checkpoint.py -q
rtk pytest -q
rtk git diff --check
~~~

Expected: all tests PASS; diff check emits no output.

- [ ] **Step 7: Run real offline timm smoke forwards**

~~~bash
rtk python3 -c "import torch; from favit_lsda.forensic import ProjectedForensicEncoder; print(tuple(ProjectedForensicEncoder('xception',768,False,0.0)(torch.randn(1,3,224,224)).shape)); print(tuple(ProjectedForensicEncoder('mobilenetv3_small_100',768,False,0.0)(torch.randn(1,3,224,224)).shape))"
~~~

Expected: two lines containing (1, 768). If dependencies are absent, install
the project environment and rerun; do not weaken this verification.

- [ ] **Step 8: Commit and inspect final state**

~~~bash
git add configs run_ffpp_tests.py tests/test_config.py README.md
git commit -m "docs: publish multibranch ablation workflow"
rtk git log --oneline -7
rtk git status --short
~~~

Expected: task commits are present and only pre-existing user-owned changes,
such as untracked wiki/, remain outside the implementation.

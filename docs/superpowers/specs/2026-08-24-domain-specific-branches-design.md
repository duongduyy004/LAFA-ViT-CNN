# Domain-Specific RGB, SRM, and FFT Branches Design

**Status:** Approved  
**Date:** 2026-08-24

## Goal

Replace the shared RGB-plus-artifact CNN with three domain-specific paths:

1. the existing FA-ViT path for RGB;
2. an ImageNet-pretrained Xception path for the fixed SRM residual; and
3. an ImageNet-pretrained MobileNetV3-Small path for FFT log-magnitude.

RGB/FA-ViT is mandatory. The SRM and FFT paths are independently optional so
the same architecture family supports RGB, RGB+SRM, RGB+FFT, and
RGB+SRM+FFT ablations.

## Non-Goals

- Do not preserve the old `ArtifactCNN` or RGB-plus-artifact channel layout.
- Do not retain wavelet preprocessing, wavelet configs, or wavelet checkpoints.
- Do not reproduce FreqNet; MobileNetV3-Small is the FFT encoder.
- Do not support disabling FA-ViT, because LSDA, teachers, distillation, GAM/LAM,
  and domain invariance require its patch representation.
- Do not silently load old `favit_lsda_cnn` checkpoints into the new model.

## Configuration Contract

The model configuration gains these fields:

```yaml
enable_srm_branch: true
enable_fft_branch: true
srm_backbone: xception
fft_backbone: mobilenetv3_small_100
forensic_pretrained: true
```

`enable_srm_branch` and `enable_fft_branch` default to `false`; RGB is always
enabled. `srm_backbone` and `fft_backbone` are validated against the exact
supported values above rather than treated as arbitrary timm model names.
`forensic_pretrained` defaults to `true`, but tests and offline construction may
override it with `false`.

The obsolete `artifact_mode` and `cnn_in_channels` fields are rejected with an
error that directs the caller to the new branch flags. This prevents a config
from appearing valid while changing its input semantics.

The six old ArtifactCNN ablation configs are replaced by four explicit configs:

- `configs/favit_lsda_rgb.yaml`
- `configs/favit_lsda_rgb_srm.yaml`
- `configs/favit_lsda_rgb_fft.yaml`
- `configs/favit_lsda_rgb_srm_fft.yaml`

The two wavelet configs are removed. Scripts and README commands must use the
new names.

## Input Construction

`FaceTransform` first applies the existing shared crop, flip, photometric,
degradation, and JPEG operations. It then normalizes RGB to approximately
`[-1, 1]` and returns a branch-input mapping:

```python
dict[str, Tensor]  # always "rgb"; optionally "srm" and "fft"
```

The exact tensors are:

- `rgb`: normalized augmented RGB, shape `[3, H, W]`;
- `srm`: the existing per-channel fixed 5x5 SRM convolution followed by
  per-image, per-channel min-max normalization to `[-1, 1]`;
- `fft`: the existing per-channel
  `fftshift(log1p(abs(fft2(rgb))))` representation followed by the same
  normalization.

Near-constant artifact channels remain zero. Disabled artifacts are not
computed and are absent from the mapping. There is no second ImageNet
normalization pass: the forensic representations retain their signed,
zero-centered convention, while full-network fine-tuning and BatchNorm adapt
the pretrained encoders to that distribution.

`GroupedForgeryDataset` stacks each mapping value into `[domains, C, H, W]`.
`FrameFaceDataset` returns a single-image mapping. PyTorch's default collation
then produces the batch mapping consumed by the engine and model. Both dataset
types validate that their mapping keys match the configured branch set.

## Encoder Components

### RGB/FA-ViT

The existing FA-ViT path is unchanged through GAM, the spatial CNN/LAM
injections, the student adapter, and `vit_feature_fusion`. It produces:

- `rgb_feature`: `[B, D]` for detection and fusion;
- `student_map`: `[B, D, 14, 14]` for LSDA distillation;
- the original patch maps used by the real and fake teacher adapters.

For ViT-Base, `D = 768`.

### SRM/Xception

When enabled, construct the encoder with:

```python
timm.create_model("xception", pretrained=forensic_pretrained,
                  num_classes=0, global_pool="avg")
```

Xception consumes only the three-channel SRM tensor. Its pooled output passes
through a branch projection:

```text
Linear(xception.num_features -> D) -> LayerNorm -> GELU -> Dropout
```

The complete Xception backbone and projection are trainable.

### FFT/MobileNetV3-Small

When enabled, construct the encoder with:

```python
timm.create_model("mobilenetv3_small_100", pretrained=forensic_pretrained,
                  num_classes=0, global_pool="avg")
```

MobileNetV3-Small consumes only the three-channel FFT log-magnitude tensor. Its
pooled output uses the same projection contract:

```text
Linear(mobilenet.num_features -> D) -> LayerNorm -> GELU -> Dropout
```

The complete MobileNetV3-Small backbone and projection are trainable.

The reusable timm wrapper and projection live in a focused
`favit_lsda/forensic.py` module rather than further expanding `model.py`.

## Fixed-Slot Late Fusion

The fusion layout is always `[RGB, SRM, FFT]`. Disabled optional paths contribute
`zeros_like(rgb_feature)` and are not instantiated or executed:

```text
rgb_feature ───────────────────────────────┐
srm_feature or zeros_like(rgb_feature) ────┼─ concat [B, 3D]
fft_feature or zeros_like(rgb_feature) ────┘
                                              |
                                              v
Linear(3D -> D) -> LayerNorm -> GELU -> Dropout
                                              |
                                              v
                                  binary head and FAL
```

This fixed layout keeps the detection head shape stable across ablations and
does not force heterogeneous encoder features into the same coordinate system
through direct addition. Each ablation is trained and evaluated as its own
run; disabling a branch at inference for a checkpoint trained with that branch
is rejected by checkpoint/config validation.

## Training Behavior

Grouped training continues to use one real plus one sample from each configured
fake domain. The FA-ViT output alone feeds:

- real and per-method fake teacher adapters;
- latent-space augmentation;
- real/fake MSE distillation;
- teacher domain classification; and
- the gradient-reversed student invariance classifier.

Only the late-fused representation feeds binary classification and FAL. This
prevents SRM/FFT features from changing the meaning of the LSDA teacher/student
objectives.

Optimizer grouping treats all pretrained encoder weights as backbones:

- `backbone.*` (FA-ViT);
- `srm_encoder.backbone.*` when present; and
- `fft_encoder.backbone.*` when present.

Those parameters use `backbone_lr_multiplier`. New branch projections, fusion,
adapters, and heads use the base learning rate. All enabled forensic parameters
remain trainable.

## Inference Behavior

Inference accepts the same branch-input mapping as training. It requires `rgb`
and requires exactly the optional keys declared by the model configuration.
Unexpected or missing keys raise a descriptive `ValueError` before any encoder
runs. Teachers, LSDA augmentation, and domain classifiers remain excluded from
the inference path.

## Checkpoint Contract

Increase the checkpoint format version and set architecture metadata to
`favit_lsda_multibranch`. Persist and validate:

```yaml
enabled_branches: [rgb, srm, fft]  # actual enabled subset, canonical order
srm_backbone: xception             # only when SRM is enabled
fft_backbone: mobilenetv3_small_100 # only when FFT is enabled
fusion: fixed_slot_concat
```

Resume and evaluation reject any mismatch between checkpoint metadata and the
caller config. Legacy ArtifactCNN checkpoints fail with a migration message;
automatic partial conversion is outside scope. `--init-favit` may still load
matching FA-ViT tensors, but must exclude the new forensic encoders, projections,
fusion module, and binary head.

## Error Handling

- Reject configs that contain removed artifact fields or unsupported backbone
  names.
- Reject missing, unexpected, non-floating, non-finite, or non-three-channel
  branch tensors with the branch name and observed shape in the message.
- Reject grouped mappings whose branch tensors disagree on group/domain/spatial
  geometry.
- Reject old or branch-incompatible checkpoints before model construction.
- Let pretrained-weight download errors surface with the timm model name; users
  can set `forensic_pretrained: false` for offline construction.

## Testing Strategy

Tests must not download weights. They construct the production model with
`forensic_pretrained=false` or monkeypatch `timm.create_model` with small
feature encoders.

Required coverage:

1. Data tests verify exact keys and `[3, H, W]` shapes for all four branch
   combinations, confirm disabled builders are not called, and remove wavelet
   expectations.
2. Encoder tests verify Xception/MobileNet model names, pretrained flags,
   projection shapes, and end-to-end gradients.
3. Model tests cover all four ablations, fixed `[RGB, SRM, FFT]` slot order,
   zero slots for disabled branches, fused feature shape, full backward, and
   teacher-free inference.
4. LSDA tests verify teacher/domain/invariance inputs remain FA-ViT-only.
5. Config tests reject legacy fields and unsupported branch backbones.
6. Checkpoint tests verify new metadata and reject legacy/mismatched branch
   layouts.
7. Engine/evaluation tests exercise mapping batches in grouped training and
   single-frame inference.
8. The complete test suite runs after focused tests, followed by a real
   `pretrained=false` smoke forward when dependencies are available.

## Documentation and Migration

Update README architecture diagrams, configuration tables, commands, input
descriptions, and checkpoint compatibility notes. Remove claims that artifacts
are concatenated with RGB or consumed by `ArtifactCNN`. State explicitly that
Xception and MobileNetV3-Small are ImageNet initializations fully fine-tuned on
SRM and FFT respectively.

No dataset manifest format changes are required. Existing training/evaluation
manifests remain valid because derived forensic inputs are still created from
the loaded RGB frames.

from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from favit_lsda import model as model_module
from favit_lsda.losses import FineGrainedAdaptiveLoss, balanced_binary_cross_entropy
from favit_lsda.model import create_favit_lsda, gradient_reverse


class TinyProjectedForensicEncoder(nn.Module):
    """Offline, differentiable stand-in for the timm forensic encoders."""

    def __init__(self, model_name, embed_dim, pretrained, dropout):
        super().__init__()
        self.model_name = model_name
        self.project = nn.Linear(3, embed_dim)

    def forward(self, images):
        return self.project(images.mean(dim=(-2, -1)))


def _tiny_model(
    monkeypatch,
    *,
    enable_srm=False,
    enable_fft=False,
    forgery_methods=("DF", "F2F"),
):
    monkeypatch.setattr(
        model_module, "ProjectedForensicEncoder", TinyProjectedForensicEncoder
    )
    return create_favit_lsda(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        forgery_methods=forgery_methods,
        train_backbone_norms=False,
        train_cls_token=False,
        enable_srm_branch=enable_srm,
        enable_fft_branch=enable_fft,
        forensic_pretrained=False,
    )


def _inputs(*, enable_srm=False, enable_fft=False, groups=1):
    values = OrderedDict(
        rgb=torch.randn(groups, 3, 3, 224, 224),
    )
    if enable_srm:
        values["srm"] = torch.randn(groups, 3, 3, 224, 224)
    if enable_fft:
        values["fft"] = torch.randn(groups, 3, 3, 224, 224)
    return values


@pytest.mark.parametrize(
    ("enable_srm", "enable_fft"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_forward_group_supports_all_fixed_slot_ablations(
    monkeypatch, enable_srm, enable_fft
):
    model = _tiny_model(
        monkeypatch, enable_srm=enable_srm, enable_fft=enable_fft
    )
    inputs = _inputs(enable_srm=enable_srm, enable_fft=enable_fft)

    output = model.forward_group(inputs)
    loss = output["logits"].square().mean() + output["distill_real"] + output["distill_fake"]
    loss.backward()

    assert tuple(inputs) == model.enabled_branches
    assert output["features"].shape == (1, 3, model.embed_dim)
    assert model.head.weight.grad is not None
    if enable_srm:
        assert model.srm_encoder.project.weight.grad is not None
    else:
        assert model.srm_encoder is None
    if enable_fft:
        assert model.fft_encoder.project.weight.grad is not None
    else:
        assert model.fft_encoder is None


def test_forward_mapping_returns_logits_and_features_without_teachers(monkeypatch):
    model = _tiny_model(monkeypatch).eval()
    monkeypatch.setattr(
        model.real_teacher,
        "forward",
        lambda _: (_ for _ in ()).throw(AssertionError("teacher called")),
    )
    logits, features = model(
        {"rgb": torch.randn(1, 3, 224, 224)}, return_features=True
    )
    assert logits.shape == (1, 2)
    assert features.shape == (1, model.embed_dim)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {},
        {"rgb": torch.randn(1, 3, 224, 224), "unexpected": torch.randn(1, 3, 224, 224)},
    ],
)
def test_forward_rejects_missing_or_unexpected_branch_keys(monkeypatch, bad_inputs):
    model = _tiny_model(monkeypatch)
    with pytest.raises(
        ValueError,
        match=r"branch '(rgb|unexpected)'.*observed shape.*exactly enabled branches",
    ):
        model(bad_inputs)


def test_forward_group_rejects_branch_geometry_mismatch_before_encoding(monkeypatch):
    model = _tiny_model(monkeypatch, enable_srm=True)
    inputs = _inputs(enable_srm=True)
    inputs["srm"] = torch.randn(1, 3, 3, 192, 224)
    with pytest.raises(ValueError, match=r"branch 'srm'.*observed shape.*rgb"):
        model.forward_group(inputs)


def test_group_validation_accepts_five_domains(monkeypatch):
    model = _tiny_model(
        monkeypatch, forgery_methods=("DF", "F2F", "FS", "NT")
    )
    model._validate_inputs(
        {"rgb": torch.randn(1, 5, 3, 224, 224)}, grouped=True
    )


def test_rgb_validation_diagnostics_include_branch_and_shape(monkeypatch):
    model = _tiny_model(monkeypatch)
    cases = [
        torch.randn(1, 3, 224),
        torch.randn(1, 1, 224, 224),
        torch.full((1, 3, 224, 224), float("nan")),
    ]
    for value in cases:
        with pytest.raises(ValueError, match=r"branch 'rgb'.*shape"):
            model._validate_inputs({"rgb": value}, grouped=False)


@pytest.mark.parametrize(
    "bad_value",
    [
        torch.randn(1, 3, 224),
        torch.randn(1, 1, 224, 224),
        torch.full((1, 3, 224, 224), float("inf")),
    ],
)
def test_optional_validation_diagnostics_include_branch_and_shape(
    monkeypatch, bad_value
):
    model = _tiny_model(monkeypatch, enable_srm=True)
    inputs = OrderedDict(
        rgb=torch.randn(1, 3, 224, 224),
        srm=bad_value,
    )
    with pytest.raises(ValueError, match=r"branch 'srm'.*shape"):
        model._validate_inputs(inputs, grouped=False)


def test_geometry_diagnostic_names_offending_branch_and_shapes(monkeypatch):
    model = _tiny_model(monkeypatch, enable_srm=True)
    inputs = _inputs(enable_srm=True)
    inputs["srm"] = torch.randn(1, 3, 3, 192, 224)
    with pytest.raises(
        ValueError,
        match=r"branch 'srm'.*shape.*rgb.*shape",
    ):
        model._validate_inputs(inputs, grouped=True)


def test_group_invariance_classifier_uses_rgb_features_only(monkeypatch):
    model = _tiny_model(monkeypatch, enable_fft=True).eval()
    inputs = _inputs(enable_fft=True)
    with torch.no_grad():
        cls, patch_maps = model.encode_latents(inputs["rgb"].flatten(0, 1))
        vit_features, _ = model._student_features(cls, patch_maps)
    captured = []
    hook = model.student_domain_classifier.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach().clone())
    )
    try:
        model.forward_group(inputs)
    finally:
        hook.remove()
    assert torch.equal(captured[0], vit_features.reshape(1, 3, -1)[:, 1:])


def test_group_training_forward_and_full_backward(monkeypatch):
    model = _tiny_model(monkeypatch, enable_srm=True, enable_fft=True)
    inputs = _inputs(enable_srm=True, enable_fft=True, groups=2)
    output = model.forward_group(inputs)
    domain_labels = torch.arange(3).expand(2, -1)
    binary = F.cross_entropy(
        output["logits"].flatten(0, 1), (domain_labels > 0).long().flatten()
    )
    domain = F.cross_entropy(
        output["domain_logits"].flatten(0, 1), domain_labels.flatten()
    )
    real = output["features"][:, :1].expand(-1, 2, -1).reshape(-1, model.embed_dim)
    fake = output["features"][:, 1:].reshape(-1, model.embed_dim)
    fal = FineGrainedAdaptiveLoss()(model.head.weight[0], real, fake)
    loss = binary + domain + output["distill_real"] + output["distill_fake"] + fal
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.srm_encoder.project.weight.grad is not None
    assert model.fft_encoder.project.weight.grad is not None
    assert model.latent_augmenter.comprehensive_fusion[0].weight.grad is not None


def test_gradient_reversal_only_negates_feature_gradient():
    features = torch.ones(2, 3, requires_grad=True)
    gradient_reverse(features, strength=0.25).sum().backward()
    assert torch.allclose(features.grad, torch.full_like(features, -0.25))


def test_binary_loss_balances_one_real_against_many_fakes():
    logits = torch.tensor([[3.0, -1.0]] + [[-0.5, 0.5]] * 4)
    labels = torch.tensor([0, 1, 1, 1, 1])
    loss = balanced_binary_cross_entropy(logits, labels)
    real = F.cross_entropy(logits[:1], labels[:1])
    fake = F.cross_entropy(logits[1:], labels[1:])
    assert torch.allclose(loss, (real + fake) / 2)


def test_favit_adapters_still_start_as_noops_and_backbone_is_frozen(monkeypatch):
    model = _tiny_model(monkeypatch)
    tokens = torch.randn(1, 197, model.embed_dim)
    assert torch.count_nonzero(model.backbone.blocks[0].attn.gam(tokens)) == 0
    assert torch.count_nonzero(model.injectors[0].scale) == 0
    assert not model.backbone.patch_embed.proj.weight.requires_grad
    assert model.student_adapter.scale.requires_grad


def test_rgb_cnn_branch_matches_favit_cnn_reference_layer_stack():
    """Pins the ported FA-ViT-CNN CNN_feature_extractor_branch layer-for-layer.

    The conv stack is the contribution being adopted from the reference
    implementation; a silent substitution (different widths, strides, or a
    dropped MaxPool) would change what the branch can represent while every
    shape assertion still passed.
    """
    branch = model_module.CNNFeatureBranch(3, 768)
    observed = [
        (type(layer).__name__, getattr(layer, "out_channels", None))
        for layer in branch.features
    ]
    assert observed == [
        ("Conv2d", 32), ("BatchNorm2d", None), ("GELU", None), ("MaxPool2d", None),
        ("Conv2d", 64), ("BatchNorm2d", None), ("GELU", None), ("MaxPool2d", None),
        ("Conv2d", 128), ("BatchNorm2d", None), ("GELU", None), ("MaxPool2d", None),
        ("Conv2d", 128), ("BatchNorm2d", None), ("GELU", None),
        ("AdaptiveAvgPool2d", None), ("Flatten", None),
    ]
    assert [type(layer).__name__ for layer in branch.project] == [
        "Linear", "LayerNorm"
    ]
    assert branch.project[0].in_features == 128


def test_rgb_cnn_branch_pools_any_spatial_size_to_embed_dim():
    branch = model_module.CNNFeatureBranch(3, 64)
    assert branch(torch.randn(2, 3, 224, 224)).shape == (2, 64)


def test_rgb_cnn_branch_occupies_second_fixed_slot(monkeypatch):
    """Pins slot order [vit, rgb_cnn, srm, fft] against a silent reordering."""
    model = _tiny_model(monkeypatch, enable_fft=True).eval()
    rgb = torch.randn(1, 3, 224, 224)
    captured = []
    hook = model.late_fusion.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach())
    )
    try:
        with torch.no_grad():
            model({"rgb": rgb, "fft": torch.randn(1, 3, 224, 224)})
            expected = model.rgb_cnn_encoder(rgb)
    finally:
        hook.remove()
    width = model.embed_dim
    assert captured[0].shape == (1, width * 4)
    assert torch.allclose(captured[0][:, width : width * 2], expected, atol=1e-6)
    assert torch.count_nonzero(captured[0][:, width * 2 : width * 3]) == 0


def test_rgb_only_still_populates_the_rgb_cnn_slot(monkeypatch):
    """The RGB CNN branch is mandatory; only SRM/FFT slots may be zero."""
    model = _tiny_model(monkeypatch)
    captured = []
    hook = model.late_fusion.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach())
    )
    try:
        model({"rgb": torch.randn(1, 3, 224, 224)})
    finally:
        hook.remove()
    width = model.embed_dim
    assert captured[0].shape == (1, width * 4)
    assert torch.count_nonzero(captured[0][:, width : width * 2]) > 0
    assert torch.count_nonzero(captured[0][:, width * 2 :]) == 0


def test_rgb_cnn_branch_cannot_be_disabled(monkeypatch):
    """Catches the branch regressing to an optional slot."""
    import inspect

    model = _tiny_model(monkeypatch)
    assert model.rgb_cnn_encoder is not None
    signature = inspect.signature(create_favit_lsda)
    assert "enable_rgb_cnn_branch" not in signature.parameters


def test_rgb_cnn_branch_reuses_rgb_input_without_new_branch_key(monkeypatch):
    """The branch is fed inputs['rgb']; it must not demand its own mapping key."""
    model = _tiny_model(monkeypatch)
    assert model.enabled_branches == ("rgb",)
    model.forward_group({"rgb": torch.randn(1, 3, 3, 224, 224)})


def test_rgb_cnn_branch_receives_gradient_in_group_training(monkeypatch):
    model = _tiny_model(monkeypatch)
    output = model.forward_group({"rgb": torch.randn(1, 3, 3, 224, 224)})
    output["logits"].square().mean().backward()
    assert model.rgb_cnn_encoder.features[0].weight.grad is not None
    assert model.rgb_cnn_encoder.project[0].weight.grad is not None

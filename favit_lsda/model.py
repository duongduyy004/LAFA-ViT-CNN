from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import timm
import torch
from torch import Tensor, nn

from .forensic import ProjectedForensicEncoder
from .lsda import LatentSpaceAugmenter, ResidualLatentAdapter


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(features: Tensor, strength: float = 1.0) -> Tensor:
    """Keep the forward representation but reverse its domain-loss gradient."""

    return _GradientReverse.apply(features, strength)


class SeparableConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, padding: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class GlobalAdaptiveModule(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.down = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.spatial = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.up = nn.Linear(hidden_dim, embed_dim * 3, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        patches = tokens - 1
        grid = math.isqrt(patches)
        if grid * grid != patches:
            raise ValueError(f"GAM requires a square patch grid, got {patches} patches")
        reduced = self.down(x)
        patch = reduced[:, 1:].reshape(batch, grid, grid, self.hidden_dim)
        patch = self.spatial(patch.permute(0, 3, 1, 2))
        patch = patch.permute(0, 2, 3, 1).reshape(batch, patches, self.hidden_dim)
        cls = reduced[:, :1].reshape(batch, 1, 1, self.hidden_dim).permute(0, 3, 1, 2)
        cls = self.spatial(cls).permute(0, 2, 3, 1).reshape(batch, 1, self.hidden_dim)
        delta = self.up(torch.cat((cls, patch), dim=1))
        return delta.reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)


class ForgeryAwareAttention(nn.Module):
    def __init__(self, attention: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.num_heads = attention.num_heads
        self.scale = attention.scale
        self.qkv = attention.qkv
        self.q_norm = getattr(attention, "q_norm", nn.Identity())
        self.k_norm = getattr(attention, "k_norm", nn.Identity())
        self.attn_drop = attention.attn_drop
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop
        self.gam = GlobalAdaptiveModule(
            self.qkv.in_features, self.num_heads, hidden_dim
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if attn_mask is not None or is_causal:
            raise NotImplementedError("FA-ViT uses neither masked nor causal attention")
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dq, dk, dv = self.gam(x).unbind(0)
        q, k, v = self.q_norm(q + dq), self.k_norm(k + dk), v + dv
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class SpatialCNN(nn.Module):
    def __init__(
        self, in_channels: int, embed_dim: int, projection_kernel: int, stride: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels * 2,
                in_channels * 4,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
        )
        self.project = nn.Conv2d(
            in_channels * 2,
            embed_dim,
            kernel_size=projection_kernel,
            stride=projection_kernel,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.block(x)
        return x, self.project(x).flatten(2).transpose(1, 2)


class CNNFeatureBranch(nn.Module):
    """Parallel CNN branch adopted from FA-ViT-CNN's ``CNN_feature_extractor_branch``.

    Unlike ``spatial_stem``/``SpatialCNN``, whose features exist to be injected
    into the ViT through LAM, this branch never touches the transformer: it
    pools to a single vector that enters late fusion beside the FA-ViT feature.
    It is mandatory: RGB always reaches late fusion through both the FA-ViT
    path and this one, so the slot is never zero-filled.

    The convolutional stack is kept layer-for-layer identical to the reference
    implementation. Only the projection width differs, and only because the
    reference exposes it as a configurable ``freq_dim``; ``embed_dim`` keeps
    every fusion slot the same width.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.project = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.project(self.features(images))


class LocalAdaptiveAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by number of heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.position_projection = nn.Linear(3, num_heads, bias=False)
        self.gating = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer("relative_indices", torch.empty(0), persistent=False)
        directions = [(v, h) for v in (-1, 0, 1) for h in (-1, 0, 1)]
        with torch.no_grad():
            for head in range(num_heads):
                vertical, horizontal = directions[head % len(directions)]
                self.position_projection.weight[head] = torch.tensor(
                    [2.0 * horizontal, 2.0 * vertical, -1.0]
                )

    def _relative(self, patches: int, device: torch.device) -> Tensor:
        if self.relative_indices.numel() and self.relative_indices.shape[1] == patches:
            return self.relative_indices.to(device)
        grid = math.isqrt(patches)
        if grid * grid != patches:
            raise ValueError(f"LAM requires a square grid, got {patches} tokens")
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        delta = coordinates[None] - coordinates[:, None]
        distance = delta.square().sum(dim=-1, keepdim=True)
        relative = torch.cat((delta[..., 1:], delta[..., :1], distance), dim=-1).float()
        self.relative_indices = relative.unsqueeze(0).to(device)
        return self.relative_indices

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        batch, query_tokens, channels = query.shape
        patches = context.shape[1]
        head_dim = channels // self.num_heads
        q = self.q(query).reshape(batch, query_tokens, self.num_heads, head_dim).transpose(1, 2)
        k = self.k(context).reshape(batch, patches, self.num_heads, head_dim).transpose(1, 2)
        v = self.v(context).reshape(batch, patches, self.num_heads, head_dim).transpose(1, 2)
        global_attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        local_attention = self.position_projection(self._relative(patches, query.device))
        local_attention = local_attention.permute(0, 3, 1, 2).softmax(dim=-1)
        gate = torch.sigmoid(self.gating).view(1, -1, 1, 1)
        patch_attention = (1 - gate) * global_attention[:, :, 1:] + gate * local_attention
        attention = torch.cat((global_attention[:, :, :1], patch_attention), dim=2)
        output = (attention @ v).transpose(1, 2).reshape(batch, query_tokens, channels)
        return self.proj(output)


class LocalInjector(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, eps=1e-6)
        self.context_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = LocalAdaptiveAttention(dim, num_heads)
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        adapted = self.attention(self.query_norm(query), self.context_norm(context))
        return query + self.scale * self.output_norm(adapted)


class ForgeryAwareLSDAViT(nn.Module):
    """FA-ViT student with LSDA domain teachers and latent augmentation."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        forgery_methods: Sequence[str] = (
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ),
        gam_reduction: int = 2,
        inject_layers: Sequence[int] = (0, 3, 6),
        train_backbone_norms: bool = True,
        train_cls_token: bool = True,
        latent_transforms: Sequence[str] = (
            "hard_interpolation",
            "centrifugal",
            "gaussian",
            "rotation",
            "difference",
        ),
        max_rotation_degrees: float = 30.0,
        latent_noise_std: float = 1.0,
        mixup_concentration_min: float = 0.5,
        mixup_concentration_max: float = 2.0,
        feature_dropout: float = 0.0,
        unfreeze_last_blocks: int = 0,
        domain_adversarial_strength: float = 1.0,
        enable_srm_branch: bool = False,
        enable_fft_branch: bool = False,
        srm_backbone: str = "xception",
        fft_backbone: str = "mobilenetv3_small_100",
        forensic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "blocks") or not hasattr(backbone, "patch_embed"):
            raise TypeError("backbone must be a timm VisionTransformer")
        if len(forgery_methods) < 2 or len(set(forgery_methods)) != len(forgery_methods):
            raise ValueError("forgery_methods must contain at least two unique domains")
        self.backbone = backbone
        self.embed_dim = int(backbone.embed_dim)
        self.enabled_branches = (
            "rgb",
            *(("srm",) if enable_srm_branch else ()),
            *(("fft",) if enable_fft_branch else ()),
        )
        self.srm_backbone_name = srm_backbone if enable_srm_branch else None
        self.fft_backbone_name = fft_backbone if enable_fft_branch else None
        self.fusion_name = "fixed_slot_concat"
        self.forgery_methods = tuple(forgery_methods)
        self.num_domains = len(self.forgery_methods) + 1
        self.inject_layers = tuple(int(index) for index in inject_layers)
        self.domain_adversarial_strength = float(domain_adversarial_strength)
        if not 0.0 <= feature_dropout < 1.0:
            raise ValueError("feature_dropout must be in [0, 1)")
        if not 0 <= unfreeze_last_blocks <= len(backbone.blocks):
            raise ValueError("unfreeze_last_blocks is outside the backbone depth")
        num_heads = int(backbone.blocks[0].attn.num_heads)
        hidden_dim = self.embed_dim // gam_reduction
        for block in self.backbone.blocks:
            block.attn = ForgeryAwareAttention(block.attn, hidden_dim)

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SeparableConv2d(32),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.spatial_blocks = nn.ModuleList(
            (
                SpatialCNN(32, self.embed_dim, 4, 1),
                SpatialCNN(64, self.embed_dim, 2, 2),
                SpatialCNN(128, self.embed_dim, 1, 2),
            )
        )
        if len(self.inject_layers) != len(self.spatial_blocks):
            raise ValueError("FA-ViT requires exactly three injection layers")
        self.injectors = nn.ModuleList(
            LocalInjector(self.embed_dim, num_heads) for _ in self.inject_layers
        )

        self.student_adapter = ResidualLatentAdapter(self.embed_dim)
        self.real_teacher = ResidualLatentAdapter(self.embed_dim)
        self.fake_teachers = nn.ModuleList(
            ResidualLatentAdapter(self.embed_dim) for _ in self.forgery_methods
        )
        self.latent_augmenter = LatentSpaceAugmenter(
            self.embed_dim,
            transforms=latent_transforms,
            max_rotation_degrees=max_rotation_degrees,
            noise_std=latent_noise_std,
            mixup_concentration_min=mixup_concentration_min,
            mixup_concentration_max=mixup_concentration_max,
        )
        self.vit_feature_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(feature_dropout),
        )
        self.rgb_cnn_encoder = CNNFeatureBranch(3, self.embed_dim)
        self.srm_encoder = (
            ProjectedForensicEncoder(
                srm_backbone, self.embed_dim, forensic_pretrained, feature_dropout
            )
            if enable_srm_branch
            else None
        )
        self.fft_encoder = (
            ProjectedForensicEncoder(
                fft_backbone, self.embed_dim, forensic_pretrained, feature_dropout
            )
            if enable_fft_branch
            else None
        )
        self.late_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(feature_dropout),
        )
        self.head = nn.Linear(self.embed_dim, num_classes, bias=False)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.num_domains),
        )
        self.student_domain_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(feature_dropout),
            nn.Linear(self.embed_dim // 2, len(self.forgery_methods)),
        )
        self._set_trainable_parameters(
            train_backbone_norms, train_cls_token, unfreeze_last_blocks
        )

    def _set_trainable_parameters(
        self,
        train_backbone_norms: bool,
        train_cls_token: bool,
        unfreeze_last_blocks: int,
    ) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in self.backbone.blocks:
            for parameter in block.attn.gam.parameters():
                parameter.requires_grad = True
        if train_backbone_norms:
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = train_cls_token
        if unfreeze_last_blocks:
            for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        modules = (
            self.spatial_stem,
            self.spatial_blocks,
            self.injectors,
            self.student_adapter,
            self.real_teacher,
            self.fake_teachers,
            self.latent_augmenter,
            self.vit_feature_fusion,
            self.rgb_cnn_encoder,
            *(module for module in (self.srm_encoder, self.fft_encoder) if module is not None),
            self.late_fusion,
            self.head,
            self.domain_classifier,
            self.student_domain_classifier,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _embed(self, images: Tensor) -> Tensor:
        tokens = self.backbone.patch_embed(images)
        tokens = self.backbone._pos_embed(tokens)
        if hasattr(self.backbone, "patch_drop"):
            tokens = self.backbone.patch_drop(tokens)
        if hasattr(self.backbone, "norm_pre"):
            tokens = self.backbone.norm_pre(tokens)
        return tokens

    def encode_latents(self, images: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.spatial_stem(images)
        tokens = self._embed(images)
        injection_index = 0
        for block_index, block in enumerate(self.backbone.blocks):
            if block_index in self.inject_layers:
                spatial, spatial_tokens = self.spatial_blocks[injection_index](spatial)
                if spatial_tokens.shape[1] != tokens.shape[1] - 1:
                    raise ValueError("FA-ViT expects matching square 224x224 token grids")
                tokens = self.injectors[injection_index](tokens, spatial_tokens)
                injection_index += 1
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        patches = tokens.shape[1] - 1
        grid = math.isqrt(patches)
        if grid * grid != patches:
            raise ValueError(f"LSDA requires a square patch grid, got {patches}")
        patch_map = tokens[:, 1:].transpose(1, 2).reshape(
            images.shape[0], self.embed_dim, grid, grid
        )
        return tokens[:, 0], patch_map

    def _student_features(self, cls_features: Tensor, patch_maps: Tensor) -> tuple[Tensor, Tensor]:
        student_maps = self.student_adapter(patch_maps)
        pooled = student_maps.mean(dim=(-2, -1))
        vit_features = self.vit_feature_fusion(torch.cat((cls_features, pooled), dim=1))
        return vit_features, student_maps

    def _validate_inputs(
        self, inputs: Mapping[str, Tensor], grouped: bool
    ) -> None:
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping of branch names to tensors")
        expected_branches = set(self.enabled_branches)
        observed_branches = set(inputs)
        if observed_branches != expected_branches:
            missing = sorted(expected_branches - observed_branches)
            unexpected = sorted(observed_branches - expected_branches)
            branch = missing[0] if missing else unexpected[0]
            value = inputs.get(branch)
            observed_shape = (
                tuple(value.shape)
                if hasattr(value, "shape")
                else "<missing>"
            )
            raise ValueError(
                f"branch {branch!r} has invalid mapping entry: "
                f"observed shape {observed_shape}; inputs must contain exactly "
                f"enabled branches {self.enabled_branches}, got {tuple(inputs)}"
            )
        expected_rank = 5 if grouped else 4
        channel_index = 2 if grouped else 1
        reference_geometry: tuple[int, ...] | None = None
        for name in self.enabled_branches:
            value = inputs[name]
            shape = tuple(value.shape)
            if value.ndim != expected_rank:
                raise ValueError(
                    f"branch {name!r} expected rank {expected_rank}, "
                    f"observed shape {shape}"
                )
            if value.shape[channel_index] != 3:
                raise ValueError(
                    f"branch {name!r} expected 3 channels, observed shape {shape}"
                )
            if not value.is_floating_point():
                raise ValueError(
                    f"branch {name!r} expected floating-point values, "
                    f"observed shape {shape} and dtype {value.dtype}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"branch {name!r} expected finite values, observed shape {shape}"
                )
            geometry = value.shape[:2] + value.shape[-2:]
            if name == "rgb":
                if grouped and value.shape[1] != self.num_domains:
                    raise ValueError(
                        f"branch 'rgb' expected {self.num_domains} domains, "
                        f"observed shape {shape}"
                    )
                reference_geometry = geometry
            elif geometry != reference_geometry:
                raise ValueError(
                    f"branch {name!r} geometry mismatch: observed shape {shape}; "
                    f"branch 'rgb' observed shape {tuple(inputs['rgb'].shape)}"
                )

    def _fused_features(
        self,
        cls_features: Tensor,
        patch_maps: Tensor,
        flat_inputs: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        vit_features, student_maps = self._student_features(cls_features, patch_maps)
        zero = torch.zeros_like(vit_features)
        rgb_cnn_features = self.rgb_cnn_encoder(flat_inputs["rgb"])
        srm_features = self.srm_encoder(flat_inputs["srm"]) if self.srm_encoder else zero
        fft_features = self.fft_encoder(flat_inputs["fft"]) if self.fft_encoder else zero
        return (
            self.late_fusion(
                torch.cat(
                    (vit_features, rgb_cnn_features, srm_features, fft_features), dim=1
                )
            ),
            vit_features,
            student_maps,
        )

    def forward_features(self, inputs: Mapping[str, Tensor]) -> Tensor:
        self._validate_inputs(inputs, grouped=False)
        images = inputs["rgb"]
        cls_features, patch_maps = self.encode_latents(images)
        features, _, _ = self._fused_features(cls_features, patch_maps, inputs)
        return features

    def forward(
        self, inputs: Mapping[str, Tensor], return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.forward_features(inputs)
        logits = self.head(features)
        return (logits, features) if return_features else logits

    def forward_group(
        self, grouped_inputs: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        self._validate_inputs(grouped_inputs, grouped=True)
        grouped_images = grouped_inputs["rgb"]
        groups, domains, channels, height, width = grouped_images.shape
        flat_inputs = {
            name: value.reshape(groups * domains, channels, height, width)
            for name, value in grouped_inputs.items()
        }
        cls, patch_maps = self.encode_latents(
            grouped_images.reshape(groups * domains, channels, height, width)
        )
        features, vit_features, student_maps = self._fused_features(
            cls, patch_maps, flat_inputs
        )
        grid_h, grid_w = student_maps.shape[-2:]
        features = features.reshape(groups, domains, self.embed_dim)
        vit_features = vit_features.reshape(groups, domains, self.embed_dim)
        student_maps = student_maps.reshape(
            groups, domains, self.embed_dim, grid_h, grid_w
        )
        source_maps = patch_maps.reshape(
            groups, domains, self.embed_dim, grid_h, grid_w
        )

        real_teacher = self.real_teacher(source_maps[:, 0])
        fake_teacher = torch.stack(
            [
                adapter(source_maps[:, index + 1])
                for index, adapter in enumerate(self.fake_teachers)
            ],
            dim=1,
        )
        comprehensive_fake = self.latent_augmenter(fake_teacher)
        distill_real = nn.functional.mse_loss(student_maps[:, 0], real_teacher)
        distill_fake = nn.functional.mse_loss(student_maps[:, 1:], comprehensive_fake)
        teacher_maps = torch.cat((real_teacher.unsqueeze(1), fake_teacher), dim=1)
        domain_logits = self.domain_classifier(teacher_maps.mean(dim=(-2, -1)))
        invariance_logits = self.student_domain_classifier(
            gradient_reverse(vit_features[:, 1:], self.domain_adversarial_strength)
        )
        return {
            "logits": self.head(features),
            "features": features,
            "domain_logits": domain_logits,
            "invariance_logits": invariance_logits,
            "distill_real": distill_real,
            "distill_fake": distill_fake,
        }

    def trainable_parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_favit_lsda(
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    pretrained: bool = True,
    num_classes: int = 2,
    forgery_methods: Sequence[str] = (
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ),
    gam_reduction: int = 2,
    inject_layers: Sequence[int] = (0, 3, 6),
    train_backbone_norms: bool = True,
    train_cls_token: bool = True,
    latent_transforms: Sequence[str] = (
        "hard_interpolation",
        "centrifugal",
        "gaussian",
        "rotation",
        "difference",
    ),
    max_rotation_degrees: float = 30.0,
    latent_noise_std: float = 1.0,
    mixup_concentration_min: float = 0.5,
    mixup_concentration_max: float = 2.0,
    feature_dropout: float = 0.0,
    unfreeze_last_blocks: int = 0,
    domain_adversarial_strength: float = 1.0,
    enable_srm_branch: bool = False,
    enable_fft_branch: bool = False,
    srm_backbone: str = "xception",
    fft_backbone: str = "mobilenetv3_small_100",
    forensic_pretrained: bool = True,
) -> ForgeryAwareLSDAViT:
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return ForgeryAwareLSDAViT(
        backbone=backbone,
        num_classes=num_classes,
        forgery_methods=forgery_methods,
        gam_reduction=gam_reduction,
        inject_layers=inject_layers,
        train_backbone_norms=train_backbone_norms,
        train_cls_token=train_cls_token,
        latent_transforms=latent_transforms,
        max_rotation_degrees=max_rotation_degrees,
        latent_noise_std=latent_noise_std,
        mixup_concentration_min=mixup_concentration_min,
        mixup_concentration_max=mixup_concentration_max,
        feature_dropout=feature_dropout,
        unfreeze_last_blocks=unfreeze_last_blocks,
        domain_adversarial_strength=domain_adversarial_strength,
        enable_srm_branch=enable_srm_branch,
        enable_fft_branch=enable_fft_branch,
        srm_backbone=srm_backbone,
        fft_backbone=fft_backbone,
        forensic_pretrained=forensic_pretrained,
    )

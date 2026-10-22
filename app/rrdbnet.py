"""Vendored RRDBNet architecture used by Real-ESRGAN.

This module is a faithful, dependency-light copy of BasicsR's
``basicsr/archs/rrdbnet_arch.py`` (v1.4.2) together with the three small
helpers it needs from ``arch_util.py`` (``default_init_weights``,
``make_layer`` and ``pixel_unshuffle``).

Only ``torch`` is required -- ``basicsr`` / ``realesrgan`` are deliberately
**not** imported because they are not installed in the project venv.  Keeping
the layer names identical means the official Real-ESRGAN ``.pth`` checkpoints
load with ``strict=True``.

Checkpoint helpers
------------------
:func:`load_rrdbnet` loads an official Real-ESRGAN checkpoint:

* ``torch.load(path, map_location="cpu")``
* use ``params_ema`` when present (official inference weights), else ``params``
* ``load_state_dict(..., strict=True)``
* ``eval()`` + ``requires_grad_(False)`` and move to the requested device
"""
from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import init as init
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    "ResidualDenseBlock",
    "RRDB",
    "RRDBNet",
    "default_init_weights",
    "make_layer",
    "pixel_unshuffle",
    "load_rrdbnet",
    "extract_state_dict",
]


# ---------------------------------------------------------------------------
# arch_util helpers (inlined, no basicsr dependency)
# ---------------------------------------------------------------------------
@torch.no_grad()
def default_init_weights(module_list, scale: float = 1.0, bias_fill: float = 0.0, **kwargs):
    """Initialize network weights (BasicsR ``arch_util.default_init_weights``)."""
    if not isinstance(module_list, (list, tuple)):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block: int, **kwarg) -> nn.Sequential:
    """Make a ``nn.Sequential`` stacking ``num_basic_block`` copies of ``basic_block``."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel unshuffle: ``(b, c, h*s, w*s) -> (b, c*s*s, h, w)``."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0, "pixel_unshuffle 输入尺寸必须能被 scale 整除"
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB / ESRGAN."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (used by RRDBNet in ESRGAN)."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block (ESRGAN).

    Model layout matches BasicsR 1.4.2 exactly, so the official Real-ESRGAN
    checkpoints (``RealESRGAN_x4plus.pth`` / ``RealESRGAN_x2plus.pth``) load
    strictly.

    Args:
        num_in_ch: Channel number of inputs.
        num_out_ch: Channel number of outputs.
        scale: ``4`` for the native x4 model, ``2`` for the x2 model (which
            first applies ``pixel_unshuffle(x, 2)`` then two 2x upsamples).
        num_feat: Channel number of intermediate features (default ``64``).
        num_block: Number of RRDB blocks (default ``23``).
        num_grow_ch: Channels for each growth (default ``32``).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _read_checkpoint(path: os.PathLike | str) -> object:
    """Load a checkpoint to CPU, preferring the safer ``weights_only`` mode."""
    path = os.fspath(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch too old to know weights_only
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Return the weight mapping from an official Real-ESRGAN checkpoint.

    Official checkpoints are ``{"params_ema": OrderedDict, "params": OrderedDict}``;
    plain state dicts are accepted too.  A leading ``module.`` prefix (DataParallel
    artefacts) is stripped when present.
    """
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping) and candidate:
                state_dict = candidate
                break
        if state_dict is None and checkpoint and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
    if state_dict is None:
        raise ValueError("无法识别的模型权重格式：既没有 params_ema，也不是 state_dict")

    state_dict = dict(state_dict)
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        state_dict = {str(k)[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_rrdbnet(
    path: os.PathLike | str,
    scale: int,
    device: str | torch.device = "cpu",
    *,
    num_feat: int = 64,
    num_block: int = 23,
    num_grow_ch: int = 32,
) -> RRDBNet:
    """Build an :class:`RRDBNet` and load official Real-ESRGAN weights into it.

    ``torch.load(..., map_location="cpu")`` is used, ``params_ema`` (falling back
    to ``params``) is loaded with ``strict=True``; the returned module is in
    ``eval()`` mode, has gradients disabled and is moved to ``device``.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ValueError: when the checkpoint payload is not a state dict.
        RuntimeError: when the state dict does not match the architecture.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型权重不存在：{path}")
    if scale not in (1, 2, 4):
        raise ValueError(f"不支持的 scale：{scale!r}（仅支持 1/2/4）")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=int(scale),
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
    )
    state_dict = extract_state_dict(_read_checkpoint(path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if device is not None:
        model.to(device)
    return model


def count_parameters(module: nn.Module) -> int:
    """Total number of parameters (handy for diagnostics/tests)."""
    return sum(int(p.numel()) for p in module.parameters())


def named_layer_names(module: nn.Module) -> Iterable[str]:
    """Yield every named module (diagnostics/tests)."""
    for name, _ in module.named_modules():
        yield name


def state_dict_keys(module: nn.Module) -> Sequence[str]:
    """Return the model's state-dict keys (diagnostics/tests)."""
    return list(module.state_dict().keys())

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PRESETS = {'720p': ('short', 720), '1080p': ('short', 1080),
           '2k': ('long', 2560), '4k': ('long', 3840)}


class Target(BaseModel):
    model_config = ConfigDict(extra='forbid')
    preset: Literal['720p', '1080p', '2k', '4k', 'custom'] = '1080p'
    edge: Literal['short', 'long'] = 'short'
    pixels: int = Field(default=1080, ge=16, le=7680, strict=True)


class InterpolationOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    model: Literal['rife-4.25', 'rife-4.25-lite'] = 'rife-4.25'
    target_fps: str = '2x'
    backend: Literal['pytorch', 'tensorrt'] = 'pytorch'
    precision: Literal['fp16', 'fp32'] = 'fp16'
    scale: Literal[1.0, 0.5, 0.25] = 1.0
    scene_cut_detection: bool = True
    order: Literal['sr_then_vfi'] = 'sr_then_vfi'

    @field_validator('target_fps', mode='before')
    @classmethod
    def valid_fps(cls, value):
        from .framerate import parse_rate
        if isinstance(value, bool):
            raise ValueError('目标帧率无效')
        text = str(value).strip().lower()
        if text not in {'2x', '4x'}:
            parse_rate(text)
        return text


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target: Target = Field(default_factory=Target)
    model: Literal['auto', 'quality', 'anime', 'general-fast', 'general-clean', 'anime-fast', 'lanczos'] = 'auto'
    strength: float = Field(default=0.75, ge=0.1, le=1)
    temporal: float = Field(default=0.4, ge=0, le=0.65)
    crf: int = Field(default=18, ge=14, le=28, strict=True)
    tile: int = Field(default=256, ge=64, le=2048, strict=True)
    tensorrt: bool = False
    tensorrt_precision: Literal['fp16', 'fp32'] = 'fp16'
    pipeline: bool = True
    workflow: Literal['upscale', 'interpolate', 'combined'] = 'upscale'
    interpolation: InterpolationOptions = Field(default_factory=InterpolationOptions)

    @model_validator(mode='after')
    def select_workflow(self):
        if self.workflow != 'upscale':
            self.interpolation.enabled = True
        elif self.interpolation.enabled:
            self.workflow = 'combined'
        return self


def plan(width: int, height: int, target: Target, sar: float = 1.0) -> dict:
    """Preserve display aspect; choose nearest even dimensions for H.264 4:2:0."""
    if width < 1 or height < 1 or not 0.01 <= sar <= 100:
        raise ValueError('视频尺寸或像素宽高比无效')
    edge, pixels = PRESETS.get(target.preset, (target.edge, target.pixels))
    display_w, display_h = width * sar, float(height)
    basis = min(display_w, display_h) if edge == 'short' else max(display_w, display_h)
    scale = pixels / basis
    # Round half up, rather than banker's rounding. Selected edge may change 1px.
    even = lambda n: max(2, 2 * int(n / 2 + 0.5))
    out_w, out_h = even(display_w * scale), even(display_h * scale)
    if max(out_w, out_h) > 7680 or out_w * out_h > 34_000_000:
        raise ValueError('输出超过 7680 边长或 3400 万像素上限，请降低目标尺寸')
    return {'width': out_w, 'height': out_h, 'edge': edge, 'pixels': pixels,
            'scale': round(scale, 6), 'ratio_error_percent': round(abs(out_w / out_h / (display_w / display_h) - 1) * 100, 4),
            'mode': 'upscale' if scale > 1.0001 else ('downscale' if scale < .9999 else 'same'),
            'note': '按显示宽高比计算，编码尺寸取最接近的偶数；不裁切、不加黑边。'}


def spatial_plan(metadata, options):
    if options.workflow != 'interpolate':
        return plan(metadata['width'], metadata['height'], options.target, metadata['sar'])
    width, height = metadata['width'] * metadata['sar'], metadata['height']
    pixels = max(16, int(max(width, height) + .5))
    result = plan(metadata['width'], metadata['height'], Target(preset='custom', edge='long', pixels=pixels), metadata['sar'])
    result.update(mode='same', note='仅插帧：保留显示尺寸，必要时归一化像素宽高比并取偶数。')
    return result

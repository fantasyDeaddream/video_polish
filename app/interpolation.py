"""Bounded-pair RIFE inference. No disk frame cache or changes to SR inputs."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ROOT

RIFE_MODELS = {
    'rife-4.25': {'version': '4.25', 'label': 'RIFE 4.25 · 标准', 'lite': False,
                  'sha256': '6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2'},
    'rife-4.25-lite': {'version': '4.25.lite', 'label': 'RIFE 4.25.lite · 快速', 'lite': True,
                       'sha256': '81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1'},
}


def weight_path(key):
    return ROOT / 'models' / 'rife' / RIFE_MODELS[key]['version'] / 'flownet.pkl'


def interpolation_status():
    return {'default_enabled': False, 'models': [
        {'id': key, 'label': meta['label'], 'available': weight_path(key).is_file()}
        for key, meta in RIFE_MODELS.items()], 'backends': ['pytorch', 'tensorrt'],
        'order': 'sr_then_vfi', 'fps_limit': 240, 'precisions': ['fp16', 'fp32']}


def is_scene_cut(first, second):
    # Motion alone should not trigger a cut; combine spatial and histogram tests.
    a = cv2.resize(first, (160, 90), interpolation=cv2.INTER_AREA)
    b = cv2.resize(second, (160, 90), interpolation=cv2.INTER_AREA)
    difference = float(np.mean(np.abs(a.astype(np.float32) - b))) / 255
    hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    distance = cv2.compareHist(cv2.normalize(hist_a, None), cv2.normalize(hist_b, None), cv2.HISTCMP_BHATTACHARYYA)
    return difference > .40 or (difference > .18 and distance > .65)


@dataclass
class Prepared:
    tensor: object
    features: object
    height: int
    width: int


class RifeInterpolator:
    def __init__(self, options, check=lambda: None, stage=lambda message: None):
        import torch
        from .rife_arch import RifeNetwork
        self.options, self.check, self.stage = options, check, stage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.precision = options.precision if self.device == 'cuda' else 'fp32'
        if options.backend == 'tensorrt' and self.device != 'cuda':
            raise RuntimeError('插帧 TensorRT 需要 NVIDIA CUDA 显卡')
        self.dtype = torch.float16 if self.precision == 'fp16' else torch.float32
        path = weight_path(options.model)
        if not path.is_file():
            raise RuntimeError('缺少 RIFE 权重，请先运行 scripts/fetch_interpolation.py')
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.digest != RIFE_MODELS[options.model]['sha256']:
            raise RuntimeError('RIFE 权重 SHA256 不匹配，请重新准备官方模型')
        stage('加载 RIFE 插帧模型')
        self.network = RifeNetwork(RIFE_MODELS[options.model]['lite'], options.scale)
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        inference_keys = set(self.network.state_dict())
        extra = set(state) - inference_keys
        if any(not k.startswith(('teacher.', 'caltime.')) for k in extra):
            raise RuntimeError('RIFE checkpoint contains unexpected inference parameters')
        # The official archives retain training-only heads commented out in IFNet.
        state = {k: v for k, v in state.items() if k in inference_keys}
        self.network.load_state_dict(state, strict=True)
        self.network.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.shape, self.accelerator = None, None
        self.generated, self.uploads = 0, 0
        self.quality = []
        check()

    def prepare(self, bgr):
        import torch
        from torch.nn import functional as F
        self.check()
        h, w = bgr.shape[:2]
        multiple = max(128, int(128 / self.options.scale))
        ph, pw = math.ceil(h / multiple) * multiple, math.ceil(w / multiple) * multiple
        if self.shape is not None and self.shape != (ph, pw):
            raise ValueError('插帧过程中分辨率改变')
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=False):
            if self.shape is None:
                self.shape = ph, pw
                self.network.configure_grid(ph, pw, self.device)
            rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
            image = torch.from_numpy(rgb).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.
            image = F.pad(image, (0, pw - w, 0, ph - h)).to(self.dtype)
            features = self.network.encode(image)
        self.uploads += self.device == 'cuda'
        return Prepared(image, features, h, w)

    def interpolate(self, first, second, timestep):
        import torch
        self.check()
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=False):
            t = torch.full((1, 1, 1, 1), float(timestep), dtype=self.dtype, device=self.device)
            inputs = first.tensor, second.tensor, t, first.features, second.features
            if self.options.backend == 'tensorrt':
                from .rife_tensorrt import RifeTensorRT
                if self.accelerator is None:
                    torch.cuda.empty_cache()
                    self.accelerator = RifeTensorRT(self.network, self.digest, inputs, self.options.scale,
                                                   self.precision, self.stage, self.check)
                elif self.generated and self.generated % 180 == 0:
                    # A black intro must not be the only numerical check of a
                    # long movie. Recheck actual later content without rebuilding
                    # the identical engine or changing any generated pixels.
                    self.quality.extend(dict(m, checked_before_generated_frame=self.generated)
                                        for m in self.accelerator.validations.values())
                    self.accelerator.validations.clear()
                output = self.accelerator(*inputs)
            else:
                output = self.network(*inputs)
            if not torch.isfinite(output).all().item():
                raise RuntimeError('插帧产生非有限像素，已停止；可尝试 FP32')
            rgb = output[0, :, :first.height, :first.width].float().clamp(0, 1).mul(255).round().byte()
            result = rgb.permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1].copy()
        self.generated += 1
        return result

    def info(self):
        result = {'model': self.options.model, 'sha256': self.digest, 'device': self.device,
                  'backend': self.options.backend, 'precision': self.precision, 'scale': self.options.scale,
                  'generated_frames': self.generated, 'endpoint_uploads': self.uploads,
                  'feature_cache': 'two adjacent endpoint encodings', 'warp_precision': 'fp32'}
        if self.accelerator:
            result['tensorrt'] = self.accelerator.info()
            result['tensorrt']['validations'] = self.quality + result['tensorrt']['validations']
            result['tensorrt']['validation_interval_generated_frames'] = 180
        return result

    def close(self):
        self.accelerator, self.network = None, None
        import torch
        if self.device == 'cuda':
            torch.cuda.empty_cache()

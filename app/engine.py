"""Tiled neural restoration + confidence-gated, motion-aligned detail smoothing."""
from __future__ import annotations

import hashlib
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT

MODEL_FILES = {
    'x2': {'file':'RealESRGAN_x2plus.pth', 'scale':2, 'blocks':23, 'family':'rrdb', 'label':'Real-ESRGAN x2plus'},
    'x4': {'file':'RealESRGAN_x4plus.pth', 'scale':4, 'blocks':23, 'family':'rrdb', 'label':'Real-ESRGAN x4plus'},
    'anime': {'file':'RealESRGAN_x4plus_anime_6B.pth', 'scale':4, 'blocks':6, 'family':'rrdb', 'label':'动漫 RRDB 6B'},
    'general-fast': {'file':'realesr-general-x4v3.pth', 'scale':4, 'blocks':32, 'family':'srvgg', 'label':'通用轻量 SRVGG'},
    'general-clean': {'file':'realesr-general-wdn-x4v3.pth', 'scale':4, 'blocks':32, 'family':'srvgg', 'label':'通用轻量 WDN（弱降噪）'},
    'anime-fast': {'file':'realesr-animevideov3.pth', 'scale':4, 'blocks':16, 'family':'srvgg', 'label':'动漫视频轻量 SRVGG'},
}


def weight_path(key):
    filename = MODEL_FILES[key]['file']
    candidates = [ROOT / 'models' / filename, ROOT.parent / 'pic_polish' / 'models' / filename]
    if os.environ.get('VIDEO_POLISH_MODELS'):
        candidates.insert(0, Path(os.environ['VIDEO_POLISH_MODELS']) / filename)
    return next((p for p in candidates if p.is_file() and p.stat().st_size > 1_000_000), None)


def model_info():
    return [{'id': key, 'file': meta['file'], 'native_scale': meta['scale'], 'family':meta['family'], 'label':meta['label'],
             'available': weight_path(key) is not None,
             'path': str(weight_path(key) or '')} for key, meta in MODEL_FILES.items()]


class Cancelled(Exception):
    pass


class Upscaler:
    def __init__(self):
        self.network = None
        self.key = None
        self.device = None
        self.digest = None
        self.accelerator = None
        self.stage_cb = lambda message: None
        self._fallback_tiles = {}
        self.transfer_stats = {}

    def load(self, key):
        import torch
        from .rrdbnet import load_rrdbnet
        if self.key == key:
            return
        self.close()
        path = weight_path(key)
        if path is None:
            raise RuntimeError(f'缺少 {MODEL_FILES[key]["file"]}；放入 models 或运行 scripts/fetch_models.py。不会自动把插值当作 AI 超分。')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        meta = MODEL_FILES[key]
        if meta['family'] == 'rrdb':
            self.network = load_rrdbnet(path, meta['scale'], self.device, num_block=meta['blocks'])
        else:
            from .srvgg import SRVGGNetCompact
            from .rrdbnet import extract_state_dict
            self.network = SRVGGNetCompact(num_conv=meta['blocks'], upscale=meta['scale'])
            self.network.load_state_dict(extract_state_dict(torch.load(path, map_location='cpu', weights_only=True)), strict=True)
            self.network.eval().requires_grad_(False).to(self.device)
        if self.device == 'cuda':
            self.network.half()
            torch.cuda.reset_peak_memory_stats()
        self.key = key
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        self.digest = digest.hexdigest()

    def enhance(self, bgr, size, model, tile, check, *, tensorrt=False, tensorrt_precision='fp16'):
        import torch
        h, w = bgr.shape[:2]
        base = cv2.resize(bgr, size, interpolation=cv2.INTER_LANCZOS4)
        if max(size[0] / w, size[1] / h) <= 1.001 or model == 'lanczos':
            return base, None, {'backend': 'lanczos', 'reason': 'explicit' if model == 'lanczos' else 'target_not_larger'}
        key = model if model in {'anime', 'general-fast', 'general-clean', 'anime-fast'} else ('x2' if max(size[0] / w, size[1] / h) <= 2 else 'x4')
        self.stage_cb('加载模型')
        self.load(key)
        scale = MODEL_FILES[key]['scale']
        request_tile = max(64, min(2048, tile // 2 * 2))
        fallback_key = (key, tensorrt, tensorrt_precision, h, w, request_tile)
        tile = self._fallback_tiles.get(fallback_key, request_tile)
        started = time.monotonic()
        while True:
            try:
                if tensorrt:
                    if self.device != 'cuda':
                        raise RuntimeError('TensorRT 需要 NVIDIA CUDA 显卡，请关闭加速以使用 CPU')
                    from .tensorrt_backend import TensorRTNetwork
                    shape = (min(h + h % 2, tile + 48), min(w + w % 2, tile + 48))
                    identity = (self.digest, tensorrt_precision, shape)
                    if self.accelerator is None or self.accelerator.identity != identity:
                        self.accelerator = TensorRTNetwork(self.network, self.digest, shape, tensorrt_precision, self.stage_cb, check)
                else:
                    self.accelerator = None
                self.stage_cb('超分与时序稳定')
                result = self._tiled(bgr, scale, tile, check)
                self._fallback_tiles[fallback_key] = tile
                break
            except torch.cuda.OutOfMemoryError:
                if tile <= 64:
                    raise RuntimeError('显存不足；请关闭其他 GPU 程序或降低目标尺寸')
                tile = max(64, tile // 4 * 2)
                self.accelerator = None
                torch.cuda.empty_cache()
        restored = cv2.resize(result, size, interpolation=cv2.INTER_LANCZOS4)
        residual = restored.astype(np.float32) - base.astype(np.float32)
        info = {'backend': 'realesrgan', 'model': MODEL_FILES[key]['file'], 'model_id': key,
                'sha256': self.digest, 'device': self.device, 'tile': tile, 'requested_tile':request_tile,
                'native_scale':scale, 'native_size':[w * scale, h * scale], 'output_size':list(size),
                'inference_engine':'tensorrt' if tensorrt else 'pytorch',
                'inference_seconds':round(time.monotonic() - started, 4), 'transfers':self.transfer_stats}
        if self.device == 'cuda':
            info.update(peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                        peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 1024**2, 1))
        if self.accelerator:
            info['tensorrt'] = self.accelerator.info()
        return base, residual, info

    def _tiled(self, bgr, scale, tile, check):
        import torch
        h, w = bgr.shape[:2]
        # x2 RRDB uses pixel_unshuffle; use even tile origins and padded even frames.
        pad_h, pad_w = h % 2, w % 2
        padded = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        hh, ww = padded.shape[:2]
        context = 24
        infer = self.accelerator or self.network
        tiles = math.ceil(hh / tile) * math.ceil(ww / tile)
        if self.accelerator:
            self.accelerator.stream.wait_stream(torch.cuda.current_stream())
        stream_context = torch.cuda.stream(self.accelerator.stream) if self.accelerator else nullcontext()
        with torch.inference_mode(), stream_context:
            # Same normalization, tile boundaries, context and per-tile network
            # calls as v1. Only movement/stitching changes: one frame upload and
            # one native-resolution mosaic download, never GPU-side resizing.
            rgb_chw = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
            full = torch.from_numpy(rgb_chw).unsqueeze(0).to(self.device)
            full = full.half() if self.device == 'cuda' else full.float()
            full = full / 255
            output = torch.empty((3, hh * scale, ww * scale), dtype=torch.uint8, device=self.device)
            for y in range(0, hh, tile):
                for x in range(0, ww, tile):
                    check()
                    x2, y2 = min(x + tile, ww), min(y + tile, hh)
                    left, top = max(0, x - context), max(0, y - context)
                    right, bottom = min(ww, x2 + context), min(hh, y2 + context)
                    tensor = full[:, :, top:bottom, left:right].contiguous()
                    prediction = infer(tensor).clamp_(0, 1)
                    arr = prediction[0].mul_(255).round_().byte()
                    cx, cy = (x - left) * scale, (y - top) * scale
                    output[:, y * scale:y2 * scale, x * scale:x2 * scale] = arr[:, cy:cy + (y2-y)*scale, cx:cx + (x2-x)*scale]
                    del tensor, prediction
            result = output[:, :h * scale, :w * scale].permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]
        self.transfer_stats = {'host_to_device':1 if self.device == 'cuda' else 0,
                               'device_to_host':1 if self.device == 'cuda' else 0, 'tiles':tiles}
        return result

    def close(self):
        self.accelerator = None
        self.network = None
        self.key = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class TemporalDetails:
    """Smooth only SR residuals, with scene, occlusion, and forward/backward checks.

    The preceding *raw* neural residual is used; smoothing never accumulates
    recursively. Original image content is never averaged across frames.
    """
    def __init__(self, strength=.75, temporal=.4):
        self.strength, self.temporal = strength, temporal
        self.previous_gray = None
        self.previous_residual = None
        self.scene_cuts = 0
        self.mean_weights = []

    def apply(self, source, base, residual):
        if residual is None:
            return base
        h, w = source.shape[:2]
        flow_scale = min(1, 512 / max(h, w))
        sw, sh = max(16, round(w * flow_scale)), max(16, round(h * flow_scale))
        gray = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
        filtered = residual
        if self.temporal and self.previous_gray is not None and self.previous_gray.shape == gray.shape:
            previous = self.previous_gray
            forward = cv2.calcOpticalFlowFarneback(gray, previous, None, .5, 4, 21, 3, 7, 1.5, 0)
            backward = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 4, 21, 3, 7, 1.5, 0)
            yy, xx = np.mgrid[:sh, :sw].astype(np.float32)
            mx, my = xx + forward[:, :, 0], yy + forward[:, :, 1]
            valid = (mx >= 1) & (mx < sw-1) & (my >= 1) & (my < sh-1)
            warped_gray = cv2.remap(previous, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            photo = np.abs(gray.astype(np.float32) - warped_gray.astype(np.float32))
            fb = forward + cv2.remap(backward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            fb_error = np.linalg.norm(fb, axis=2)
            cut = float(np.mean(photo)) > 30 or float(np.mean(photo > 35)) > .55
            if cut:
                self.scene_cuts += 1
            else:
                confidence = np.exp(-photo / 9) * np.exp(-fb_error / 1.0) * valid * (photo < 24) * (fb_error < 2)
                oh, ow = base.shape[:2]
                full_flow = cv2.resize(forward, (ow, oh), interpolation=cv2.INTER_LINEAR)
                full_flow[:, :, 0] *= ow / sw
                full_flow[:, :, 1] *= oh / sh
                gy, gx = np.mgrid[:oh, :ow].astype(np.float32)
                prev = cv2.remap(self.previous_residual.astype(np.float32), gx + full_flow[:, :, 0],
                                 gy + full_flow[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                weight = cv2.resize(confidence, (ow, oh), interpolation=cv2.INTER_LINEAR)[:, :, None] * self.temporal
                # Disagreeing neural details are unreliable and must not leave trails.
                weight *= np.exp(-np.mean(np.abs(prev - residual), axis=2, keepdims=True) / 18)
                filtered = residual * (1 - weight) + prev * weight
                self.mean_weights.append(float(weight.mean()))
        self.previous_gray = gray
        self.previous_residual = residual.astype(np.float16) if self.temporal else None
        return np.clip(base.astype(np.float32) + self.strength * filtered, 0, 255).round().astype(np.uint8)

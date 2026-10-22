"""Tiled neural restoration + confidence-gated, motion-aligned detail smoothing."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import cv2
import numpy as np

from app.config import ROOT

MODEL_FILES = {
    'x2': ('RealESRGAN_x2plus.pth', 2, 23),
    'x4': ('RealESRGAN_x4plus.pth', 4, 23),
    'anime': ('RealESRGAN_x4plus_anime_6B.pth', 4, 6),
}


def weight_path(key):
    filename = MODEL_FILES[key][0]
    candidates = [ROOT / 'models' / filename, ROOT.parent / 'pic_polish' / 'models' / filename]
    if os.environ.get('VIDEO_POLISH_MODELS'):
        candidates.insert(0, Path(os.environ['VIDEO_POLISH_MODELS']) / filename)
    return next((p for p in candidates if p.is_file() and p.stat().st_size > 1_000_000), None)


def model_info():
    return [{'id': key, 'file': meta[0], 'native_scale': meta[1],
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

    def load(self, key):
        import torch
        from app.rrdbnet import load_rrdbnet
        if self.key == key:
            return
        self.close()
        path = weight_path(key)
        if path is None:
            raise RuntimeError(f'缺少 {MODEL_FILES[key][0]}；放入 models 或设置 VIDEO_POLISH_MODELS。不会自动把插值当作 AI 超分。')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        self.network = load_rrdbnet(path, MODEL_FILES[key][1], self.device, num_block=MODEL_FILES[key][2])
        if self.device == 'cuda':
            self.network.half()
            torch.cuda.reset_peak_memory_stats()
        self.key = key
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        self.digest = digest.hexdigest()

    def enhance(self, bgr, size, model, tile, check):
        import torch
        h, w = bgr.shape[:2]
        base = cv2.resize(bgr, size, interpolation=cv2.INTER_LANCZOS4)
        if max(size[0] / w, size[1] / h) <= 1.001 or model == 'lanczos':
            return base, None, {'backend': 'lanczos', 'reason': 'explicit' if model == 'lanczos' else 'target_not_larger'}
        key = 'anime' if model == 'anime' else ('x2' if max(size[0] / w, size[1] / h) <= 2 else 'x4')
        self.load(key)
        scale = MODEL_FILES[key][1]
        tile = tile // 2 * 2
        while True:
            try:
                result = self._tiled(bgr, scale, tile, check)
                break
            except torch.cuda.OutOfMemoryError:
                if tile <= 64:
                    raise RuntimeError('显存不足；请关闭其他 GPU 程序或降低目标尺寸')
                tile = max(64, tile // 4 * 2)
                torch.cuda.empty_cache()
        restored = cv2.resize(result, size, interpolation=cv2.INTER_LANCZOS4)
        residual = restored.astype(np.float32) - base.astype(np.float32)
        return base, residual, {'backend': 'realesrgan', 'model': MODEL_FILES[key][0],
                                'sha256': self.digest, 'device': self.device, 'tile': tile}

    def _tiled(self, bgr, scale, tile, check):
        import torch
        h, w = bgr.shape[:2]
        # x2 RRDB uses pixel_unshuffle; use even tile origins and padded even frames.
        pad_h, pad_w = h % 2, w % 2
        padded = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        hh, ww = padded.shape[:2]
        output = np.empty((hh * scale, ww * scale, 3), np.uint8)
        context = 24
        with torch.inference_mode():
            for y in range(0, hh, tile):
                for x in range(0, ww, tile):
                    check()
                    x2, y2 = min(x + tile, ww), min(y + tile, hh)
                    left, top = max(0, x - context), max(0, y - context)
                    right, bottom = min(ww, x2 + context), min(hh, y2 + context)
                    patch = np.ascontiguousarray(padded[top:bottom, left:right, ::-1].transpose(2, 0, 1))
                    tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                    tensor = tensor.half() if self.device == 'cuda' else tensor.float()
                    prediction = self.network(tensor / 255).clamp_(0, 1)
                    arr = prediction[0].mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                    cx, cy = (x - left) * scale, (y - top) * scale
                    output[y * scale:y2 * scale, x * scale:x2 * scale] = arr[cy:cy + (y2-y)*scale, cx:cx + (x2-x)*scale]
                    del tensor, prediction
        return output[:h * scale, :w * scale]

    def close(self):
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

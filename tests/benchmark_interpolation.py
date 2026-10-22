"""Reproducible real-motion quality and full-pipeline VFI measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from app.config import ROOT
from app.engine import Upscaler
from app.interpolation import RifeInterpolator
from app.media import probe, process_video
from app.quality import compare
from app.target import InterpolationOptions, Options, spatial_plan

FOLDER = ROOT / 'tests' / 'artifacts' / 'v12'
SOURCE = FOLDER / 'sintel-trailer-720p.mp4'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sample_triplets():
    wanted = {i for center in (301, 601, 901) for i in (center - 1, center, center + 1)}
    selected = {}
    with av.open(str(SOURCE)) as reader:
        for index, frame in enumerate(reader.decode(video=0)):
            if index in wanted:
                selected[index] = cv2.resize(frame.to_ndarray(format='bgr24'), (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            if index >= max(wanted):
                break
    return [(selected[i-1], selected[i], selected[i+1]) for i in (301, 601, 901)]


def quality(args):
    triplets = sample_triplets()
    records, references = [], {}
    report = FOLDER / 'quality.json'
    if report.is_file():
        records = json.loads(report.read_text(encoding='utf-8'))
    for model in args.models:
        # Always obtain an FP32 baseline before testing a requested backend.
        for backend, precision in [('pytorch', 'fp32'), (args.backend, args.precision)]:
            label = f'{model}-{backend}-{precision}'
            print('Quality run:', label, flush=True)
            runner = RifeInterpolator(InterpolationOptions(enabled=True, model=model, backend=backend, precision=precision),
                                      stage=lambda s: print(s, flush=True))
            run = {'model': model, 'backend': backend, 'precision': precision, 'size': [1920, 1080],
                   'source': 'Sintel trailer, triplets around frames 301/601/901; resized with Lanczos4, no SR',
                   'samples': [], 'status': 'running'}
            torch.cuda.reset_peak_memory_stats()
            try:
                for index, (a, truth, b) in enumerate(triplets):
                    pa, pb = runner.prepare(a), runner.prepare(b)
                    for t in (.2, .5, .8):
                        started = time.monotonic()
                        output = runner.interpolate(pa, pb, t)
                        elapsed = time.monotonic() - started
                        row = {'pair': index, 'timestep': t, 'seconds': round(elapsed, 5)}
                        if backend == 'pytorch' and precision == 'fp32':
                            references[index, t] = output
                        else:
                            row['vs_pytorch_fp32'] = compare(references[index, t], output)
                        if t == .5:
                            row['vs_withheld_frame'] = compare(truth, output)
                            row['linear_blend_vs_withheld'] = compare(truth, cv2.addWeighted(a, .5, b, .5, 0))
                            if index == 1:
                                cv2.imwrite(str(FOLDER / f'{label}-midpoint.png'), output)
                                cv2.imwrite(str(FOLDER / 'withheld-truth.png'), truth)
                        run['samples'].append(row)
                        print(label, index, t, row, flush=True)
                    del pa, pb
                run['status'] = 'passed'
            except Exception as exc:
                run.update(status='rejected', error=str(exc))
                print(label, 'rejected:', exc, flush=True)
            finally:
                run['engine'] = runner.info()
                run['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                run['torch_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
                runner.close()
                del runner
                records.append(run)
                save_json(report, records)


def make_minute():
    """60s/30fps natural-motion workload; 24fps source resampled, final 7.8s looped.

    Independent 60s stereo test tone is used for exact AAC packet-copy checks.
    This is an engineering workload, not native 30fps acquisition ground truth.
    """
    path = FOLDER / 'sintel-normalized-1080p30-60s.mp4'
    if path.is_file():
        return path
    with av.open(str(path), 'w') as writer:
        stream = writer.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'bf': '0'}
        audio = writer.add_stream('aac', rate=48000)
        audio.layout = 'stereo'
        audio.codec_context.time_base = Fraction(1, 48000)
        input_count = 1253
        reader = av.open(str(SOURCE))
        iterator = reader.decode(video=0)
        current_index, current = -1, None
        for i in range(1800):
            desired = (i * 24 // 30) % input_count
            if desired < current_index:
                reader.close()
                reader = av.open(str(SOURCE))
                iterator, current_index = reader.decode(video=0), -1
            while current_index < desired:
                current = next(iterator)
                current_index += 1
            image = cv2.resize(current.to_ndarray(format='bgr24'), (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 30)
            writer.mux(stream.encode(frame))
        reader.close()
        writer.mux(stream.encode(None))
        for start in range(0, 60 * 48000, 1024):
            count = min(1024, 60 * 48000 - start)
            wave = (.08 * np.sin(np.arange(start, start + count) * 440 * 2 * np.pi / 48000)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.stack((wave, wave)), format='fltp', layout='stereo')
            frame.sample_rate, frame.pts, frame.time_base = 48000, start, Fraction(1, 48000)
            writer.mux(audio.encode(frame))
        writer.mux(audio.encode(None))
    return path


def video(args):
    path = make_minute()
    metadata = probe(path)
    options = Options(workflow='interpolate', interpolation={'target_fps': '60', 'model': args.models[0],
                      'backend': args.backend, 'precision': args.precision})
    engine = Upscaler()
    destination = FOLDER / f'minute-{args.models[0]}-{args.backend}-{args.precision}.mp4'
    last_log = 0
    def progress(count, timestamp, elapsed, backend):
        nonlocal last_log
        if time.monotonic() - last_log > 10:
            print(f'{count}/1800 source frames; {elapsed:.1f}s elapsed; {backend.get("interpolation")}', flush=True)
            last_log = time.monotonic()
    try:
        result = process_video(path, destination, metadata, spatial_plan(metadata, options), options, engine, lambda: None, progress)
    finally:
        engine.close()
    def inspect(p):
        with av.open(str(p)) as reader:
            s = reader.streams.video[0]
            info = {'rate': str(s.average_rate), 'duration': float(s.duration * s.time_base), 'frames': s.frames,
                    'size': [s.width, s.height]}
            info['audio_sha256'] = hashlib.sha256(b''.join(bytes(x) for x in reader.demux(audio=0) if x.dts is not None)).hexdigest()
        return info
    original, output = inspect(path), inspect(destination)
    assert original['frames'] == 1800 and output['frames'] == 3600
    assert output['rate'] == '60' and abs(output['duration'] - 60) < 1/60000
    assert original['audio_sha256'] == output['audio_sha256']
    report = {'source': original, 'output': output, 'evidence': result,
              'source_note': 'Official Sintel trailer (24fps) resampled to 30fps/1080P; tail looped to 60s; generated AAC test tone.',
              'runtime_note': 'This is an interpolation-only workload; not a measurement of super-resolution+interpolation.'}
    save_json(destination.with_suffix('.json'), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['quality', 'video'])
    parser.add_argument('--models', nargs='+', default=['rife-4.25', 'rife-4.25-lite'])
    parser.add_argument('--backend', choices=['pytorch', 'tensorrt'], default='pytorch')
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True, parents=True)
    (quality if args.phase == 'quality' else video)(args)

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import types
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from app.engine import Cancelled, Upscaler
from app.framerate import FrameClock, interpolation_plan, output_time_base
from app.interpolation import RifeInterpolator, is_scene_cut, weight_path
from app.media import probe, process_video
from app.target import InterpolationOptions, Options, Target, spatial_plan
from scripts.network import routes
from tests.fixtures import make_video
from tests.test_api import client, wait_done


def audio_hash(path, index=0):
    with av.open(str(path)) as container:
        return hashlib.sha256(b''.join(bytes(p) for p in container.demux(audio=index) if p.dts is not None)).hexdigest()


def decoded(path):
    with av.open(str(path)) as container:
        return [(Fraction(f.pts) * f.time_base, f.to_ndarray(format='bgr24')) for f in container.decode(video=0)]


@pytest.mark.parametrize('rate', ['60', '60000/1001', '120'])
def test_exact_clock_and_vfr(rate):
    rate = Fraction(rate)
    points = [Fraction(0), Fraction(1, 24), Fraction(11, 100), Fraction(1, 4)]
    clock = FrameClock(rate, 0)
    samples = []
    for a, b in zip(points, points[1:]):
        for timestamp, alpha in clock.positions(a, b):
            assert 0 <= alpha < 1
            assert timestamp == a + alpha * (b - a)
            samples.append(timestamp)
    assert samples == [Fraction(i, 1) / rate for i in range(math.ceil(Fraction(1, 4) * rate))]
    assert (1 / rate / output_time_base(rate)).denominator == 1


def test_validation_and_proxy_policy(monkeypatch):
    for invalid in ['0', '-1', 'nan', 'inf', '1/0', '241', True]:
        with pytest.raises(ValueError):
            InterpolationOptions(target_fps=invalid)
    options = Options(interpolation={'enabled': True})
    assert options.workflow == 'combined'
    assert Options(workflow='interpolate').interpolation.enabled
    with pytest.raises(ValueError):
        interpolation_plan({'rate': '60'}, InterpolationOptions(enabled=True, target_fps='48'))
    monkeypatch.delenv('VIDEO_POLISH_DOWNLOAD_PROXY', raising=False)
    assert routes('https://drive.google.com/test') == ['http://127.0.0.1:7900', 'http://127.0.0.1:7897', None]
    assert routes('https://pypi.tuna.tsinghua.edu.cn/test') == [None]
    assert routes('https://mirrors.aliyun.com/test') == [None]
    monkeypatch.setenv('VIDEO_POLISH_DOWNLOAD_PROXY', 'direct')
    assert routes('https://github.com/test') == [None]


@pytest.mark.parametrize('key', ['rife-4.25', 'rife-4.25-lite'])
def test_fp32_matches_official_architecture(key, monkeypatch):
    # Execute the reviewed, pinned IFNet source only in this independent reference
    # test. Production never imports downloaded Python or training dependencies.
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', False)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    opts = InterpolationOptions(enabled=True, model=key, precision='fp32')
    runner = RifeInterpolator(opts)
    warplayer = types.ModuleType('model.warplayer')
    def warp(image, flow):
        h, w = image.shape[2:]
        x = torch.linspace(-1, 1, w, device=image.device).view(1, 1, 1, w).expand(1, 1, h, w)
        y = torch.linspace(-1, 1, h, device=image.device).view(1, 1, h, 1).expand(1, 1, h, w)
        normal = torch.cat((flow[:, :1] / ((w - 1.) / 2.), flow[:, 1:] / ((h - 1.) / 2.)), 1)
        return torch.nn.functional.grid_sample(image, (torch.cat((x, y), 1) + normal).permute(0, 2, 3, 1),
                                              mode='bilinear', padding_mode='border', align_corners=True)
    warplayer.warp = warp
    monkeypatch.setitem(sys.modules, 'model', types.ModuleType('model'))
    monkeypatch.setitem(sys.modules, 'model.warplayer', warplayer)
    source = weight_path(key).parent / 'IFNet_HDv3.py'
    expected_source = {'rife-4.25': '655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32',
                       'rife-4.25-lite': 'ea5eb5f05a391a2ac26402b16acf9fdb91968f299373f9be24aa4a8c5483d3d3'}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source[key]
    spec = importlib.util.spec_from_file_location('official_rife_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.IFNet().to(runner.device).eval()
    reference.load_state_dict(runner.network.state_dict())
    rng = np.random.default_rng(3)
    a = rng.integers(30, 220, (90, 160, 3), dtype=np.uint8)
    b = np.roll(a, 3, axis=1).copy()
    try:
        with torch.inference_mode():
            pa, pb = runner.prepare(a), runner.prepare(b)
            for t in (.2, .5, .8):
                expected = reference(torch.cat((pa.tensor, pb.tensor), 1), t, runner.network.scales)[-1][-1]
                actual = runner.network(pa.tensor, pb.tensor, torch.full((1, 1, 1, 1), t, device=runner.device), pa.features, pb.features)
                assert torch.max(torch.abs(expected - actual)).item() < 1e-4
    finally:
        del reference
        runner.close()


@pytest.mark.parametrize('fps,vfr,frames,source_rate', [('60', False, 6, 24), ('60', False, 6, 30), ('60000/1001', True, 6, 24), ('60', False, 1, 24), ('4x', False, 2, 24)])
def test_real_video_timing_audio_and_tail(tmp_path, fps, vfr, frames, source_rate):
    source, output = tmp_path / '原片.mp4', tmp_path / '插帧结果.mp4'
    make_video(source, frames=frames, vfr=vfr, rate=source_rate)
    metadata = probe(source)
    options = Options(workflow='interpolate', interpolation={'target_fps': fps, 'model': 'rife-4.25-lite'})
    engine = Upscaler()
    try:
        result = process_video(source, output, metadata, spatial_plan(metadata, options), options, engine, lambda: None, lambda *a: None)
    finally:
        engine.close()
    target_rate = Fraction(result['interpolation']['rate'])
    timeline = decoded(output)
    assert len(timeline) == math.ceil(Fraction(str(result['interpolation']['duration_seconds'])) * target_rate - Fraction(1, 100000))
    assert all(abs(t - Fraction(i, 1) / target_rate) <= output_time_base(target_rate) for i, (t, _) in enumerate(timeline))
    assert timeline[0][1].shape[:2] == (90, 160)
    assert audio_hash(source) == audio_hash(output)
    assert result['engine']['backend'] == 'identity'
    if frames > 2:
        assert result['interpolation']['generated_frames'] > 0
        assert result['interpolation']['endpoint_uploads'] <= frames
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == target_rate
        assert abs(float(stream.duration * stream.time_base) - metadata['duration']) <= float(1 / target_rate) + 1e-5


def test_interpolation_pipeline_serial_equivalence_and_pcm(tmp_path):
    source = tmp_path / 'multi.mkv'
    make_video(source, codec='pcm_s16le', audio_tracks=2)
    metadata = probe(source)
    results = []
    for pipeline in (True, False):
        options = Options(workflow='combined', model='lanczos', pipeline=pipeline,
                          target=Target(preset='custom', edge='long', pixels=240), interpolation={'target_fps': '2x'})
        engine = Upscaler()
        output = tmp_path / f'{pipeline}.mp4'
        try:
            evidence = process_video(source, output, metadata, spatial_plan(metadata, options), options, engine, lambda: None, lambda *a: None)
        finally:
            engine.close()
        results.append((decoded(output), output))
        assert evidence['frames'] == 6 and evidence['output_frames'] >= 12
        with av.open(str(output)) as container:
            assert [s.metadata.get('language') for s in container.streams.audio] == ['zho', 'eng']
            assert all(s.codec_context.name == 'aac' for s in container.streams.audio)
    assert [t for t, _ in results[0][0]] == [t for t, _ in results[1][0]]
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(results[0][0], results[1][0]))
    assert all(audio_hash(results[0][1], i) == audio_hash(results[1][1], i) for i in range(2))


def test_cut_and_cancellation(tmp_path):
    a = np.zeros((90, 160, 3), np.uint8)
    assert is_scene_cut(a, a + 240)
    assert not is_scene_cut(a, a)
    source, output = tmp_path / 'source.mp4', tmp_path / 'out.mp4'
    make_video(source)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls > 20:
            raise Cancelled()
    options = Options(workflow='interpolate')
    engine = Upscaler()
    try:
        with pytest.raises(Cancelled):
            process_video(source, output, probe(source), {'width': 160, 'height': 90}, options, engine, check, lambda *a: None)
    finally:
        engine.close()
    assert not output.exists() and not output.with_suffix('.partial.mp4').exists()


def test_hard_cut_never_morphs_shots(tmp_path):
    source, output = tmp_path / 'cut.mp4', tmp_path / 'out.mp4'
    with av.open(str(source), 'w') as writer:
        stream = writer.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 160, 90, 'yuv420p'
        stream.options = {'crf': '14', 'bf': '0'}
        for i, value in enumerate((0, 240)):
            frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), value, np.uint8), format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 24)
            writer.mux(stream.encode(frame))
        writer.mux(stream.encode(None))
    engine = Upscaler()
    try:
        result = process_video(source, output, probe(source), {'width': 160, 'height': 90},
                               Options(workflow='interpolate', interpolation={'target_fps': '60'}), engine, lambda: None, lambda *a: None)
    finally:
        engine.close()
    assert result['interpolation']['scene_cuts'] == 1
    assert result['interpolation']['generated_frames'] == 0
    source_means = [float(image.mean()) for _, image in decoded(source)]
    # Compare to observed shot colors, allowing H.264/RGB-YUV round-trip error.
    assert all(abs(float(image.mean()) - source_means[0 if timestamp < Fraction(1, 24) else 1]) < 6
               for timestamp, image in decoded(output))


def test_combined_only_superresolves_originals(tmp_path):
    source, output = tmp_path / 'in.mp4', tmp_path / 'out.mp4'
    make_video(source, frames=3)
    class CountingUpscaler(Upscaler):
        calls = 0
        def enhance(self, *args, **kwargs):
            self.calls += 1
            return super().enhance(*args, **kwargs)
    engine = CountingUpscaler()
    options = Options(workflow='combined', tile=256, target=Target(preset='custom', edge='long', pixels=240))
    try:
        evidence = process_video(source, output, probe(source), {'width': 240, 'height': 136}, options, engine, lambda: None, lambda *a: None)
    finally:
        engine.close()
    assert engine.calls == 3
    assert evidence['output_frames'] == 6
    assert evidence['engine']['native_scale'] == 2
    assert evidence['engine']['native_size'] == [320, 180]
    assert evidence['interpolation']['generated_frames'] == 2


def test_interpolation_api_plan_job_and_download(client, tmp_path):
    source = tmp_path / '输入.mp4'
    make_video(source, frames=6, rate=30)
    response = client.post('/api/uploads', files={'files': (source.name, source.read_bytes())})
    assert response.status_code == 201
    job_id = response.json()['id']
    settings = {'target_fps': '60', 'model': 'rife-4.25-lite'}
    response = client.post('/api/plan', json={'upload_id': job_id, 'workflow': 'interpolate', 'interpolation': settings})
    assert response.status_code == 200, response.text
    planned = response.json()['files'][0]
    assert (planned['target']['width'], planned['target']['height']) == (160, 90)
    assert planned['interpolation']['rate'] == '60'
    response = client.post('/api/jobs', json={'upload_id': job_id, 'options': {'workflow': 'interpolate', 'interpolation': settings}})
    assert response.status_code == 202, response.text
    result = wait_done(client, job_id)
    assert result['status'] == 'done', result
    item = result['files'][0]
    assert item['evidence']['output_frames'] == 12
    assert item['output_name'].endswith('_160x90_60fps.mp4')
    assert client.get(item['download_url'], headers={'Range': 'bytes=0-99'}).status_code == 206
    assert client.delete('/api/jobs/' + job_id).status_code == 200


def test_nonzero_video_start_preserves_audio_offset(tmp_path):
    source, output = tmp_path / 'offset.mp4', tmp_path / 'offset-out.mp4'
    with av.open(str(source), 'w') as writer:
        video = writer.add_stream('libx264', rate=24)
        video.width, video.height, video.pix_fmt = 160, 90, 'yuv420p'
        video.options = {'bf': '0', 'crf': '18'}
        audio = writer.add_stream('aac', rate=48000)
        audio.layout = 'stereo'
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), 70 + i, np.uint8), format='bgr24')
            frame.pts, frame.time_base = 24 + i, Fraction(1, 24)
            writer.mux(video.encode(frame))
        writer.mux(video.encode(None))
        for start in range(0, 72000, 1024):
            frame = av.AudioFrame.from_ndarray(np.zeros((2, min(1024, 72000 - start)), np.float32), format='fltp', layout='stereo')
            frame.sample_rate, frame.pts, frame.time_base = 48000, start, Fraction(1, 48000)
            writer.mux(audio.encode(frame))
        writer.mux(audio.encode(None))
    metadata = probe(source)
    options = Options(workflow='interpolate', interpolation={'target_fps': '60'})
    engine = Upscaler()
    try:
        result = process_video(source, output, metadata, spatial_plan(metadata, options), options, engine, lambda: None, lambda *a: None)
    finally:
        engine.close()
    assert result['output_frames'] == 15
    assert abs(float(decoded(output)[0][0]) - 1.) < 1/90000
    assert audio_hash(source) == audio_hash(output)

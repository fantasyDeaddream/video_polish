"""Live 1.2 API, proxy routes, and Windows/WSL lifecycle evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from fractions import Fraction

import av
import httpx
import numpy as np

from app.config import ROOT
from app.interpolation import RIFE_MODELS, weight_path
from app.media import probe
from scripts.network import opener, routes

FOLDER = ROOT / 'tests' / 'artifacts' / 'v12'


def save(name, value):
    (FOLDER / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def audio_hash(path):
    with av.open(str(path)) as reader:
        return hashlib.sha256(b''.join(bytes(p) for p in reader.demux(audio=0) if p.dts is not None)).hexdigest()


def short_source():
    path = FOLDER / 'sintel-short-720p24.mp4'
    if path.is_file():
        return path
    with av.open(str(FOLDER / 'sintel-trailer-720p.mp4')) as reader, av.open(str(path), 'w') as writer:
        reader.seek(int(12 / reader.streams.video[0].time_base), stream=reader.streams.video[0])
        video = writer.add_stream('libx264', rate=24)
        video.width, video.height, video.pix_fmt = 1280, 720, 'yuv420p'
        video.options = {'crf': '18', 'bf': '0'}
        audio = writer.add_stream('aac', rate=48000)
        audio.layout = 'stereo'
        iterator = reader.decode(video=0)
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(next(iterator).to_ndarray(format='bgr24'), format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 24)
            writer.mux(video.encode(frame))
        writer.mux(video.encode(None))
        for start in range(0, 24000, 1024):
            count = min(1024, 24000 - start)
            tone = (.1 * np.sin(np.arange(start, start + count) * 440 * 2 * np.pi / 48000)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.stack((tone, tone)), format='fltp', layout='stereo')
            frame.sample_rate, frame.pts, frame.time_base = 48000, start, Fraction(1, 48000)
            writer.mux(audio.encode(frame))
        writer.mux(audio.encode(None))
    return path


def upload(client, source):
    with source.open('rb') as handle:
        response = client.post('/api/uploads', files={'files': (source.name, handle, 'video/mp4')})
    response.raise_for_status()
    return response.json()['id']


def wait(client, job_id, condition, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get('/api/jobs/' + job_id)
        response.raise_for_status()
        record = response.json()
        if condition(record):
            return record
        if record['status'] in {'failed', 'cancelled', 'partial'}:
            raise RuntimeError(str(record))
        time.sleep(.2)
    raise TimeoutError(job_id)


def e2e(client, ui_ids=()):
    source = short_source()
    records = []
    for workflow in ('upscale', 'combined'):
        job_id = upload(client, source)
        options = {'workflow': workflow, 'target': {'preset': '1080p'}, 'model': 'auto', 'tile': 512,
                   'tensorrt': True, 'interpolation': {'target_fps': '60', 'model': 'rife-4.25', 'backend': 'tensorrt'}}
        client.post('/api/jobs', json={'upload_id': job_id, 'options': options}).raise_for_status()
        record = wait(client, job_id, lambda j: j['status'] == 'done')
        item = record['files'][0]
        output = FOLDER / f'http-{workflow}-1080p.mp4'
        with client.stream('GET', item['download_url']) as response, output.open('wb') as handle:
            response.raise_for_status()
            for data in response.iter_bytes(1024 * 1024):
                handle.write(data)
        metadata = probe(output)
        expected_frames = 30 if workflow == 'combined' else 12
        assert metadata['frames'] == expected_frames and (metadata['width'], metadata['height']) == (1920, 1080)
        assert metadata['rate'] == ('60' if workflow == 'combined' else '24')
        assert audio_hash(source) == audio_hash(output)
        assert item['evidence']['engine']['native_scale'] == 2
        assert item['evidence']['engine']['native_size'] == [2560, 1440]
        assert item['evidence']['frames'] == 12
        if workflow == 'combined':
            assert item['evidence']['interpolation']['tensorrt']['loaded_from_cache']
        assert client.get(item['preview_url'], headers={'Range': 'bytes=0-99'}).status_code == 206
        records.append({'job': record, 'output': str(output), 'metadata': metadata, 'audio_packet_copy': True})
        print(workflow, 'passed:', item['evidence'], flush=True)
    # Persist the owned UI jobs and results before lifecycle cleanup.
    ui = [j for j in client.get('/api/jobs').json()['jobs'] if j['id'] in ui_ids]
    save('http-e2e.json', {'cases': records, 'ui_jobs': ui})


def prepare_stop(client, label):
    jobs = client.get('/api/jobs').json()['jobs']
    if any(j['status'] in {'running', 'queued'} for j in jobs):
        raise RuntimeError('Do not interrupt existing work; complete active jobs before lifecycle testing')
    source = FOLDER / 'sintel-normalized-1080p30-60s.mp4'
    if not source.is_file():
        raise RuntimeError('Run the full-minute benchmark first')
    pending = upload(client, ROOT / 'tests/artifacts/portrait.mp4')
    active = upload(client, source)
    client.post('/api/jobs', json={'upload_id': active, 'options': {'workflow': 'interpolate',
                'interpolation': {'model': 'rife-4.25-lite', 'target_fps': '60', 'backend': 'tensorrt'}}}).raise_for_status()
    record = wait(client, active, lambda j: j['status'] == 'running' and j['files'][0].get('output_frames', 0) > 10, timeout=60)
    (ROOT / 'cache/tmp/v12-stop.partial.mp4').write_bytes(b'owned lifecycle test video cache')
    save(f'lifecycle-{label}.json', {'active_job': active, 'pending_job': pending,
         'processed_before_stop': record['files'][0]['processed_frames'],
         'output_before_stop': record['files'][0]['output_frames'],
         'rife_engines': [p.name for p in (ROOT / '.runtime/rife-tensorrt').glob('*.engine')]})
    print('Active RIFE TensorRT job, pending upload, and intermediate ready for', label, 'stop test.', flush=True)


def verify_stop(label):
    path = FOLDER / f'lifecycle-{label}.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    assert not (ROOT / '.runtime/server.json').exists()
    assert not [p for p in (ROOT / 'cache').rglob('*') if p.is_file()]
    assert all((ROOT / '.runtime/rife-tensorrt' / name).is_file() for name in record['rife_engines'])
    assert all(hashlib.sha256(weight_path(key).read_bytes()).hexdigest() == meta['sha256'] for key, meta in RIFE_MODELS.items())
    assert (FOLDER / 'web-saved-interpolation.mp4').is_file()
    assert (FOLDER / 'http-combined-1080p.mp4').is_file()
    record.update(service_stopped=True, remaining_video_cache_files=0, compiled_engines_preserved=True,
                  model_hashes_preserved=True, saved_videos_preserved=True)
    save(path.name, record)
    print(label, 'lifecycle passed', flush=True)


def proxy_test():
    url = 'https://api.github.com/repos/hzwer/Practical-RIFE'
    result = {'non_cn_routes': routes(url), 'cn_routes': routes('https://pypi.tuna.tsinghua.edu.cn/simple/'), 'ports': []}
    for port in (7900, 7897):
        started = time.monotonic()
        with opener(f'http://127.0.0.1:{port}').open(urllib.request.Request(url, headers={'User-Agent': 'VideoPolish/1.2'}), timeout=20) as response:
            payload = response.read()
            result['ports'].append({'port': port, 'status': response.status, 'bytes': len(payload), 'seconds': time.monotonic() - started})
    assert all(p['status'] == 200 for p in result['ports'])
    save('proxy.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['e2e', 'prepare-stop', 'verify-stop', 'proxy'])
    parser.add_argument('--label', default='windows')
    parser.add_argument('--url', default='http://127.0.0.1:8800')
    parser.add_argument('--ui-job-ids', nargs='*', default=[])
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    if args.phase == 'verify-stop':
        verify_stop(args.label)
    elif args.phase == 'proxy':
        proxy_test()
    else:
        with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
            if args.phase == 'e2e':
                e2e(client, args.ui_job_ids)
            else:
                prepare_stop(client, args.label)

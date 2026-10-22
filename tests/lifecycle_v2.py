"""Real stop-script verification with active TensorRT and bounded video queues."""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction

import av
import httpx

from app.config import ROOT

parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['prepare','verify'])
parser.add_argument('--url',default='http://127.0.0.1:8800')
args=parser.parse_args()
folder=ROOT/'tests/artifacts/v2'
report_file=folder/'lifecycle.json'
if args.action=='prepare':
    with httpx.Client(base_url=args.url,timeout=30,trust_env=False) as client:
        previous=client.get('/api/jobs').json()['jobs']
        if previous:
            raise RuntimeError('Preserve existing jobs: clear only known test jobs before lifecycle verification')
        source=folder/'stop-test-720p.mp4'
        with av.open(str(ROOT/'tests/artifacts/sample_720p.mp4')) as reader:
            images=[f.to_ndarray(format='bgr24') for f in reader.decode(video=0)]
        with av.open(str(source),'w') as writer:
            stream=writer.add_stream('libx264',rate=24)
            stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'ultrafast','bf':'0'}
            for i in range(96):
                frame=av.VideoFrame.from_ndarray(images[i%len(images)],format='bgr24')
                frame.pts=i;frame.time_base=Fraction(1,24)
                writer.mux(stream.encode(frame))
            writer.mux(stream.encode(None))
        def upload(path):
            with path.open('rb') as handle:
                response=client.post('/api/uploads',files={'files':(path.name,handle,'video/mp4')})
            response.raise_for_status();return response.json()['id']
        pending=upload(ROOT/'tests/artifacts/portrait.mp4')
        active=upload(source)
        client.post('/api/jobs',json={'upload_id':active,'options':{'target':{'preset':'1080p'},'tile':512,'tensorrt':True,'pipeline':True}}).raise_for_status()
        (ROOT/'cache/tmp/lifecycle.partial.mp4').write_bytes(b'owned test intermediate')
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+active).json()
            if job['status']=='running' and job['files'][0].get('processed_frames',0)>=1:break
            if job['status'] in {'failed','done','cancelled'}:raise RuntimeError(str(job))
            time.sleep(.2)
        assert job['status']=='running'
        report={'active_job':active,'pending_job':pending,'processed_frames_at_stop':job['files'][0].get('processed_frames'),
                'engine':job['files'][0].get('engine'),
                'engines_before':len(list((ROOT/'.runtime/tensorrt').glob('*.engine')))}
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Active TensorRT job, pending input and intermediate ready for stop-script test.',flush=True)
else:
    report=json.loads(report_file.read_text(encoding='utf-8'))
    remaining=[str(p) for p in (ROOT/'cache').rglob('*') if p.is_file()]
    assert not remaining,remaining
    assert not (ROOT/'.runtime/server.json').exists()
    assert len(list((ROOT/'.runtime/tensorrt').glob('*.engine')))>=report['engines_before']
    assert (ROOT/'models/realesr-general-x4v3.pth').is_file()
    assert (folder/'v2-tensorrt.mp4').is_file()
    report.update(cache_files_after=0,service_stopped=True,engine_cache_preserved=True,weights_preserved=True,saved_video_preserved=True)
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('V2 lifecycle passed: active TensorRT/queues stopped; video cache empty; compiled engines and saved copies preserved.')

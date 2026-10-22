"""Test a real HTTP TensorRT job, including first-build UI responsiveness."""
import argparse
import json
import time
from pathlib import Path

import httpx

from app.config import ROOT
from app.media import probe

parser=argparse.ArgumentParser()
parser.add_argument('--tile',type=int,default=768)
parser.add_argument('--url',default='http://127.0.0.1:8800')
args=parser.parse_args()
folder=ROOT/'tests/artifacts/v2'
folder.mkdir(parents=True,exist_ok=True)
with httpx.Client(base_url=args.url,timeout=15,trust_env=False) as client:
    health=client.get('/api/health');health.raise_for_status()
    assert health.json()['acceleration']['available']
    with (ROOT/'tests/artifacts/sample_720p.mp4').open('rb') as handle:
        response=client.post('/api/uploads',files={'files':('sample_720p.mp4',handle,'video/mp4')})
    response.raise_for_status();job_id=response.json()['id']
    options={'target':{'preset':'1080p'},'tile':args.tile,'tensorrt':True,'tensorrt_precision':'fp16','pipeline':True}
    response=client.post('/api/jobs',json={'upload_id':job_id,'options':options});response.raise_for_status()
    deadline=time.monotonic()+600
    latencies=[];failures=[];last_stage=None
    while time.monotonic()<deadline:
        started=time.monotonic()
        try:
            health=client.get('/api/health',timeout=3);health.raise_for_status()
            latencies.append(time.monotonic()-started)
            response=client.get('/api/jobs/'+job_id,timeout=3);response.raise_for_status();job=response.json()
        except httpx.TimeoutException as exc:
            failures.append(str(exc));print('HTTP timeout during build',flush=True);continue
        stage=job['files'][0].get('stage')
        if stage!=last_stage:
            print(f'{job["status"]}: {stage}',flush=True);last_stage=stage
        if job['status'] in {'done','partial','failed','cancelled'}:break
        time.sleep(1)
    assert job['status']=='done',job
    item=job['files'][0];engine=item['evidence']['engine']
    assert engine['inference_engine']=='tensorrt' and engine['native_size']==[2560,1440]
    assert engine['output_size']==[1920,1080] and item['evidence']['frames']==12
    assert engine['tensorrt']['quality_checked']
    output=folder/f'http-trt-{args.tile}.mp4'
    with client.stream('GET',item['download_url']) as response,output.open('wb') as handle:
        response.raise_for_status()
        for chunk in response.iter_bytes(1024*1024):handle.write(chunk)
    metadata=probe(output)
    assert (metadata['width'],metadata['height'],metadata['audio_tracks'])==(1920,1080,1)
    report={'job_id':job_id,'options':options,'metadata':metadata,'evidence':item['evidence'],
            'health_max_latency_seconds':max(latencies),'health_checks':len(latencies),'health_timeouts':len(failures)}
    (folder/f'http-trt-{args.tile}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not failures,report
    print(f'HTTP TensorRT passed; {len(latencies)} health checks, max latency {max(latencies):.3f}s',flush=True)

"""Prepare an active GPU job, then verify a real stop script removed all videos."""
import argparse
import json
import time
from pathlib import Path

import httpx

from app.config import ROOT


parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['prepare','verify'])
parser.add_argument('--url',default='http://127.0.0.1:8801')
args=parser.parse_args()
report_path=ROOT/'tests'/'artifacts'/'lifecycle-report.json'
if args.action=='prepare':
    with httpx.Client(base_url=args.url,timeout=60,trust_env=False) as client:
        def upload(name):
            with (ROOT/'tests'/'artifacts'/name).open('rb') as handle:
                response=client.post('/api/uploads',files={'files':(name,handle,'video/mp4')})
            response.raise_for_status();return response.json()['id']
        pending=upload('portrait.mp4')
        job_id=upload('sample_720p.mp4')
        response=client.post('/api/jobs',json={'upload_id':job_id,'options':{'target':{'preset':'1080p'}}});response.raise_for_status()
        (ROOT/'cache'/'tmp'/'stop-test.partial.mp4').write_bytes(b'test intermediate video cache')
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+job_id).json()
            if job['status']=='running' and job['files'][0].get('processed_frames',0)>=1:break
            time.sleep(.2)
        assert job['status']=='running',job
        report={'url':args.url,'active_status':job['status'],'processed_frames':job['files'][0].get('processed_frames',0),
                'pending_batch':pending,'active_batch':job_id,'before_cache_files':len(list((ROOT/'cache').rglob('*.mp4')))}
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Active GPU job and pending upload ready; run real stop script now.')
else:
    report=json.loads(report_path.read_text(encoding='utf-8'))
    remaining=[str(p) for p in (ROOT/'cache').rglob('*') if p.is_file()]
    assert not remaining,remaining
    assert not (ROOT/'.runtime'/'server.json').exists()
    assert (ROOT/'tests'/'artifacts'/'result_1080p.mp4').is_file()
    try:
        httpx.get(args.url+'/api/health',timeout=2,trust_env=False)
        raise AssertionError('Server still responding')
    except (httpx.ConnectError, httpx.ConnectTimeout):
        pass
    report.update(after_cache_files=0,service_stopped=True,saved_copy_preserved=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Lifecycle passed: active worker stopped; cache empty; saved copy preserved.')

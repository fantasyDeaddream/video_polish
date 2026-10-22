"""Run against a script-managed live service; write reproducible media evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import httpx
import numpy as np

from app.config import ROOT
from tests.fixtures import make_video,video_times


def make_photo_clip(path):
    photo=ROOT.parent/'pic_polish'/'samples'/'photo_1280x720.jpg'
    image=cv2.imdecode(np.fromfile(str(photo),np.uint8),cv2.IMREAD_COLOR)
    image=cv2.resize(image,(1280,720))
    with av.open(str(path),'w') as output:
        stream=output.add_stream('libx264',rate=24);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'18','bf':'0'}
        audio=output.add_stream('aac',rate=48000);audio.layout='stereo'
        for i in range(12):
            frame=av.VideoFrame.from_ndarray(np.roll(image,i,axis=1),format='bgr24');frame.pts=i;frame.time_base=Fraction(1,24)
            output.mux(stream.encode(frame))
        output.mux(stream.encode(None))
        for start in range(0,24000,1024):
            count=min(1024,24000-start);wave=(np.sin(np.arange(start,start+count)*2*np.pi*440/48000)*.1).astype(np.float32)
            frame=av.AudioFrame.from_ndarray(np.stack([wave,wave]),format='fltp',layout='stereo');frame.sample_rate=48000;frame.pts=start;frame.time_base=Fraction(1,48000)
            output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8800');args=parser.parse_args()
    folder=ROOT/'tests'/'artifacts';folder.mkdir(exist_ok=True)
    landscape=folder/'sample_720p.mp4';portrait=folder/'portrait.mp4';vfr=folder/'vfr.mp4'
    make_photo_clip(landscape);make_video(portrait,90,160,frames=6,audio=False);make_video(vfr,vfr=True)
    report={'url':args.url,'time':time.time(),'cases':[]}
    with httpx.Client(base_url=args.url,timeout=60,trust_env=False) as client:
        health=client.get('/api/health');health.raise_for_status();report['health']=health.json()
        handles=[p.open('rb') for p in [landscape,portrait]]
        try:response=client.post('/api/uploads',files=[('files',(p.name,h,'video/mp4')) for p,h in zip([landscape,portrait],handles)])
        finally:
            for h in handles:h.close()
        response.raise_for_status();batch=response.json();job_id=batch['id']
        assert [(f['metadata']['width'],f['metadata']['height']) for f in batch['files']]==[(1280,720),(90,160)]
        plans={}
        for preset in ['720p','1080p','2k','4k']:
            response=client.post('/api/plan',json={'upload_id':job_id,'target':{'preset':preset}});response.raise_for_status();plans[preset]=[f['target'] for f in response.json()['files']]
        assert plans['1080p'][0]['width']==1920 and plans['1080p'][1]['width']==1080
        assert plans['2k'][0]['width']==2560 and plans['4k'][0]['width']==3840
        # Neural x2 for the photo-derived 720p clip; neural x4 for the small portrait.
        response=client.post('/api/jobs',json={'upload_id':job_id,'options':{'target':{'preset':'1080p'},'model':'auto'}});response.raise_for_status()
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            result=client.get('/api/jobs/'+job_id).json()
            print(json.dumps({'status':result['status'],'frames':[f.get('processed_frames',0) for f in result['files']]},ensure_ascii=True),flush=True)
            if result['status'] in {'done','partial','failed','cancelled'}:break
            time.sleep(4)
        assert result['status']=='done',result
        for item in result['files']:
            response=client.get(item['download_url']);response.raise_for_status()
            saved=folder/('result_1080p.mp4' if item['index']==0 else 'result_portrait.mp4');saved.write_bytes(response.content)
            meta=__import__('app.media',fromlist=['probe']).probe(saved)
            source=[landscape,portrait][item['index']]
            assert np.allclose(video_times(source),video_times(saved),atol=1/90000)
            assert meta['audio_tracks']==(1 if item['index']==0 else 0)
            assert item['evidence']['engine']['backend']=='realesrgan'
            report['cases'].append({'input':str(source),'output':str(saved),'metadata':meta,'evidence':item['evidence'],'sha256':hashlib.sha256(response.content).hexdigest()})
        # Inputs have been reclaimed immediately; outputs remain downloadable.
        assert not any((ROOT/'cache'/'uploads'/job_id).iterdir())
        report['plans']=plans
        report['job_id']=job_id
    (folder/'e2e-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('E2E passed; report: tests/artifacts/e2e-report.json',flush=True)


if __name__=='__main__':main()

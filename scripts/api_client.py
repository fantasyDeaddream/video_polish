"""Example batch API client: uploads files, polls, and streams results to disk."""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import httpx


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('videos',nargs='+',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:8800')
    parser.add_argument('--preset',choices=['720p','1080p','2k','4k','custom'],default='1080p')
    parser.add_argument('--edge',choices=['short','long'],default='short')
    parser.add_argument('--pixels',type=int,default=1080)
    parser.add_argument('--model',choices=['auto','quality','anime','general-fast','general-clean','anime-fast','lanczos'],default='auto')
    parser.add_argument('--tile',type=int,default=256)
    parser.add_argument('--tensorrt',action='store_true')
    parser.add_argument('--trt-precision',choices=['fp16','fp32'],default='fp16')
    parser.add_argument('--no-pipeline',action='store_true')
    parser.add_argument('--workflow', choices=['upscale','interpolate','combined'], default='upscale')
    parser.add_argument('--fps', default='2x', help='2x, 4x, 60, or a fraction such as 60000/1001')
    parser.add_argument('--rife-model', choices=['rife-4.25','rife-4.25-lite'], default='rife-4.25')
    parser.add_argument('--rife-backend', choices=['pytorch','tensorrt'], default='pytorch')
    parser.add_argument('--rife-precision', choices=['fp16','fp32'], default='fp16')
    parser.add_argument('--rife-scale', type=float, choices=[1.,.5,.25], default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with httpx.Client(base_url=args.url,timeout=300,trust_env=False) as client, contextlib.ExitStack() as stack:
        files=[('files',(path.name,stack.enter_context(path.open('rb')),'application/octet-stream')) for path in args.videos]
        response=client.post('/api/uploads',files=files);response.raise_for_status()
        batch=response.json();job_id=batch['id']
        target={'preset':args.preset,'edge':args.edge,'pixels':args.pixels}
        interpolation={'enabled':args.workflow!='upscale','model':args.rife_model,'target_fps':args.fps,
                       'backend':args.rife_backend,'precision':args.rife_precision,'scale':args.rife_scale}
        response=client.post('/api/plan',json={'upload_id':job_id,'target':target,'workflow':args.workflow,'interpolation':interpolation});response.raise_for_status()
        print(json.dumps(response.json(),ensure_ascii=False,indent=2))
        response=client.post('/api/jobs',json={'upload_id':job_id,'options':{'target':target,'model':args.model,'tile':args.tile,
                             'tensorrt':args.tensorrt,'tensorrt_precision':args.trt_precision,'pipeline':not args.no_pipeline,
                             'workflow':args.workflow,'interpolation':interpolation}});response.raise_for_status()
        while True:
            response=client.get('/api/jobs/'+job_id);response.raise_for_status();job=response.json()
            print(f'{job_id}: {job["status"]}, {job["progress"]*100:.1f}%',flush=True)
            if job['status'] in {'done','partial','failed','cancelled'}:break
            time.sleep(2)
        saved=0
        for item in job['files']:
            if item['status']!='done':
                print(f'{item["name"]}: {item.get("error",item["status"])}')
                continue
            destination=args.output/item['output_name']
            suffix=1
            while destination.exists() or destination.with_suffix('.partial.mp4').exists():
                destination=args.output/f'{Path(item["output_name"]).stem}_{suffix}.mp4';suffix+=1
            partial=destination.with_suffix('.partial.mp4')
            try:
                with client.stream('GET',item['download_url']) as response,partial.open('xb') as handle:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024*1024):handle.write(chunk)
                partial.replace(destination)
                saved+=1
                print(f'Saved: {destination}')
            finally:partial.unlink(missing_ok=True)
        if not saved:raise SystemExit(1)


if __name__=='__main__':main()

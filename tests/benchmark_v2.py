"""Reproducible warm throughput and reference-agreement benchmarks on this GPU.

The model comparison measures agreement with v1, not ground-truth fidelity.
All native x2/x4 outputs are still resized on CPU with the original Lanczos4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from app.config import ROOT
from app.engine import Upscaler, TemporalDetails
from app.media import probe, process_video
from app.quality import compare
from app.target import Options
from tests.reference_v1_engine import Upscaler as OldUpscaler
from tests.reference_v1_media import process_video as old_process_video

FOLDER=ROOT/'tests'/'artifacts'/'v2'
FOLDER.mkdir(parents=True,exist_ok=True)
REPORT=FOLDER/'benchmark.json'


def save(report):
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def frames(path):
    with av.open(str(path)) as reader:
        return [frame.to_ndarray(format='bgr24') for frame in reader.decode(video=0)]


def result(base,detail):
    return np.clip(base.astype(np.float32)+.75*detail,0,255).round().astype(np.uint8)


def one_case(images,model,tile,*,reference=False,tensorrt=False,precision='fp16',repeats=3):
    engine=OldUpscaler() if reference else Upscaler()
    if not reference:
        engine.stage_cb=lambda message:print(f'[{model}, {tile}, {precision}] {message}',flush=True) if 'TensorRT' in message or 'ONNX' in message else None
    extra={} if reference else {'tensorrt':tensorrt,'tensorrt_precision':precision}
    initialized=time.perf_counter()
    try:
        first=engine.enhance(images[0],(1920,1080),model,tile,lambda:None,**extra)
        cold=time.perf_counter()-initialized
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times=[]
        native_info=None
        for index in range(repeats):
            started=time.perf_counter()
            base,detail,native_info=engine.enhance(images[index%len(images)],(1920,1080),model,tile,lambda:None,**extra)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        out=result(first[0],first[1])
        measurements={'model':model,'requested_tile':tile,'actual_tile':native_info.get('tile'),
                      'reference_v1':reference,'tensorrt':tensorrt,'precision':precision if tensorrt else 'fp16',
                      'warmup_or_build_seconds':round(cold,4),'seconds_per_frame':times,
                      'median_seconds':statistics.median(times),'fps':1/statistics.median(times),
                      'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,2),
                      'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/1024**2,2),'engine':native_info}
        free,total=torch.cuda.mem_get_info()
        measurements['device_used_after_inference_mib']=round((total-free)/1024**2,2)
        print(json.dumps({k:measurements[k] for k in ['model','requested_tile','actual_tile','tensorrt','fps','peak_allocated_mib']},ensure_ascii=False),flush=True)
        return measurements,out
    finally:
        engine.close()
        torch.cuda.empty_cache()


def video_case(source,name,options,reference=False):
    engine=OldUpscaler() if reference else Upscaler()
    destination=FOLDER/f'{name}.mp4'
    metadata=probe(source)
    function=old_process_video if reference else process_video
    try:
        engine.load(options.model if options.model in {'anime','general-fast','general-clean','anime-fast'} else 'x2')
        # Exclude first-forward kernel initialization from the steady video run.
        sample=frames(source)[0]
        extra={} if reference else {'tensorrt':options.tensorrt,'tensorrt_precision':options.tensorrt_precision}
        engine.enhance(sample,(1920,1080),options.model,options.tile,lambda:None,**extra)
        torch.cuda.synchronize()
        started=time.perf_counter()
        evidence=function(source,destination,metadata,{'width':1920,'height':1080},options,engine,lambda:None,lambda *args:None)
        elapsed=time.perf_counter()-started
        return {'name':name,'seconds':elapsed,'fps':evidence['frames']/elapsed,'evidence':evidence,'file':str(destination)}
    finally:engine.close()


def video_agreement(a,b):
    reference,candidate=frames(a),frames(b)
    assert len(reference)==len(candidate)
    metrics=[compare(x,y) for x,y in zip(reference,candidate)]
    def timestamps(path):
        with av.open(str(path)) as c:return [float(f.pts*f.time_base) for f in c.decode(video=0)]
    def audio(path):
        with av.open(str(path)) as c:return hashlib.sha256(b''.join(bytes(p) for p in c.demux(audio=0) if p.dts is not None)).hexdigest()
    return {'frames':len(metrics),'all_decoded_pixels_identical':all(m['identical'] for m in metrics),
            'max_abs':max(m['max_abs'] for m in metrics),'mean_mae':statistics.mean(m['mae'] for m in metrics),
            'mean_ssim_luma':statistics.mean(m['ssim_luma'] for m in metrics),
            'mean_psnr_db':statistics.mean(m['psnr_db'] for m in metrics if m['psnr_db'] is not None) if any(m['psnr_db'] is not None for m in metrics) else None,
            'timestamps_identical':timestamps(a)==timestamps(b),'aac_packets_identical':audio(a)==audio(b)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['engineering','pipeline','tensorrt','light-videos','all'],default='all')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    source=ROOT/'tests'/'artifacts'/'sample_720p.mp4'
    images=frames(source)[:3]
    if not images:raise RuntimeError('Run tests/e2e.py once to create sample_720p.mp4')
    report=json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
    report.update(date='2026-10-21',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                  source=str(source),input_size=[1280,720],output_size=[1920,1080],repeats=args.repeats,
                  warning='Photo-derived short clip; model-to-model PSNR/SSIM measure agreement, not true restoration quality.')
    if args.phase in {'engineering','all'}:
        measured,reference=one_case(images,'auto',512,reference=True,repeats=args.repeats)
        report['reference']=measured;save(report)
        cv2.imencode('.png',reference)[1].tofile(str(FOLDER/'reference-frame.png'))
        report['tiles']=[]
        for tile in [256,512,768,1024,1536,2048]:
            measured,out=one_case(images,'auto',tile,repeats=args.repeats)
            measured['agreement_with_v1_tile512']=compare(reference,out)
            if tile==512:
                assert measured['agreement_with_v1_tile512']['identical']
            report['tiles'].append(measured);save(report)
        report['lightweight']=[]
        for model in ['general-fast','general-clean','anime-fast']:
            measured,out=one_case(images,model,512,repeats=args.repeats)
            measured['agreement_with_heavy_model']=compare(reference,out)
            cv2.imencode('.png',out)[1].tofile(str(FOLDER/f'{model}-frame.png'))
            report['lightweight'].append(measured);save(report)
        videos=[]
        for name,pipeline,original in [('v1-reference',False,True),('v2-serial',False,False),('v2-pipeline',True,False)]:
            videos.append(video_case(source,name,Options(tile=512,pipeline=pipeline),reference=original))
            report['videos']=videos;save(report)
        report['video_agreement']={name:video_agreement(videos[0]['file'],video['file']) for name,video in zip(['v2-serial','v2-pipeline'],videos[1:])}
        assert all(v['all_decoded_pixels_identical'] and v['timestamps_identical'] and v['aac_packets_identical'] for v in report['video_agreement'].values())
        save(report)
    if args.phase == 'pipeline':
        candidate=video_case(source,'v2-pipeline',Options(tile=512,pipeline=True))
        report['videos']=[video for video in report['videos'] if video['name']!='v2-pipeline']+[candidate]
        report['video_agreement']['v2-pipeline']=video_agreement(FOLDER/'v1-reference.mp4',candidate['file'])
        assert report['video_agreement']['v2-pipeline']['all_decoded_pixels_identical']
        save(report)
    if args.phase in {'tensorrt','all'}:
        reference=cv2.imdecode(np.fromfile(str(FOLDER/'reference-frame.png'),np.uint8),cv2.IMREAD_COLOR)
        cold=report.setdefault('tensorrt_cold_builds',[])
        for previous in report.get('tensorrt',[]):
            info=previous['engine'].get('tensorrt',{})
            if not info.get('loaded_from_cache',True) and not any(x.get('engine_file')==info.get('engine_file') for x in cold):
                cold.append(info)
        report['tensorrt']=[]
        for model,tile,precision in [('auto',512,'fp16'),('auto',1536,'fp16'),('general-fast',512,'fp16'),('auto',512,'fp32')]:
            measured,out=one_case(images,model,tile,tensorrt=True,precision=precision,repeats=args.repeats)
            measured['agreement_with_v1_tile512']=compare(reference,out)
            if model=='general-fast':
                same=cv2.imdecode(np.fromfile(str(FOLDER/'general-fast-frame.png'),np.uint8),cv2.IMREAD_COLOR)
            elif tile==512:
                same=reference
            else:
                _,same=one_case(images,model,tile,repeats=1)
            measured['agreement_same_model_same_tile']=compare(same,out)
            cv2.imencode('.png',out)[1].tofile(str(FOLDER/f'trt-{model}-{tile}-{precision}.png'))
            report['tensorrt'].append(measured);save(report)
        trt_video=video_case(source,'v2-tensorrt',Options(tile=512,tensorrt=True,pipeline=True))
        report['tensorrt_video']=trt_video
        report['tensorrt_video_agreement']=video_agreement(FOLDER/'v1-reference.mp4',trt_video['file'])
        save(report)
    if args.phase in {'light-videos','all'}:
        report['lightweight_videos']=[]
        for model,use_trt in [('general-fast',False),('general-fast',True),('anime-fast',False)]:
            name=f'{model}-'+('trt' if use_trt else 'torch')
            measured=video_case(source,name,Options(model=model,tile=512,tensorrt=use_trt,pipeline=True))
            measured.update(model=model,tensorrt=use_trt)
            report['lightweight_videos'].append(measured);save(report)
    print(f'Benchmark complete: {REPORT}',flush=True)


if __name__=='__main__':main()

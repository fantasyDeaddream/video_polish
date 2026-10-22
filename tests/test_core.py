from __future__ import annotations

import hashlib
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from app.engine import TemporalDetails, Upscaler
from app.media import probe, process_video
from app.target import Options, Target, plan
from tests.fixtures import make_video, video_times


@pytest.mark.parametrize('source,preset,expected',[
    ((1280,720),'1080p',(1920,1080)),((720,1280),'1080p',(1080,1920)),
    ((1920,1080),'720p',(1280,720)),((1280,720),'2k',(2560,1440)),
    ((720,1280),'4k',(2160,3840)),((1000,1000),'4k',(3840,3840)),
])
def test_dimensions(source,preset,expected):
    p=plan(*source,Target(preset=preset))
    assert (p['width'],p['height'])==expected


def test_custom_odd_anamorphic():
    p=plan(853,479,Target(preset='custom',edge='long',pixels=1601))
    assert p['width']==1602 and p['height']%2==0 and p['ratio_error_percent']<.2
    p=plan(720,576,Target(preset='1080p'),sar=16/15)
    assert (p['width'],p['height'])==(1440,1080)
    with pytest.raises(ValueError):plan(4096,32,Target(preset='custom',edge='short',pixels=7680))


def test_temporal_detail_stability_and_cut():
    h,w=90,160
    rng=np.random.default_rng(7)
    source=rng.integers(40,210,(h,w,3),dtype=np.uint8)
    base=cv2.resize(source,(w*2,h*2))
    stable=TemporalDetails(1,.65)
    raw,smoothed=[],[]
    for i in range(16):
        residual=np.full(base.shape,5 if i%2 else -5,np.float32)
        result=stable.apply(source,base,residual)
        raw.append(float(residual.mean()));smoothed.append(float((result.astype(np.float32)-base).mean()))
    assert np.std(smoothed[2:])<np.std(raw[2:])*.85
    # A hard cut must not borrow residuals from the previous shot.
    new_source=np.full_like(source,245)
    new_base=np.full_like(base,245)
    result=stable.apply(new_source,new_base,np.full_like(base, -8,dtype=np.float32))
    assert stable.scene_cuts==1
    assert np.all(result==237)


def test_vfr_audio_and_unicode(tmp_path):
    source=tmp_path/'原视频 vfr.mp4';output=tmp_path/'结果.mp4'
    make_video(source,vfr=True)
    metadata=probe(source)
    opts=Options(model='lanczos',target=Target(preset='custom',edge='long',pixels=240))
    target=plan(metadata['width'],metadata['height'],opts.target)
    engine=Upscaler()
    try: evidence=process_video(source,output,metadata,target,opts,engine,lambda:None,lambda *args:None)
    finally: engine.close()
    assert evidence['frames']==6 and evidence['engine']['backend']=='lanczos'
    assert np.allclose(video_times(source),video_times(output),atol=1/90000)
    with av.open(str(output)) as container:
        assert len(container.streams.audio)==1 and container.streams.audio[0].codec_context.name=='aac'
        assert (container.streams.video[0].width,container.streams.video[0].height)==(240,136)
    def audio_hash(path):
        with av.open(str(path)) as c:
            return hashlib.sha256(b''.join(bytes(p) for p in c.demux(audio=0) if p.dts is not None)).hexdigest()
    assert audio_hash(source)==audio_hash(output)


def test_cancel_removes_partial(tmp_path):
    from app.engine import Cancelled
    source=tmp_path/'in.mp4';output=tmp_path/'out.mp4';make_video(source)
    calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls>4:raise Cancelled()
    with pytest.raises(Cancelled):
        process_video(source,output,probe(source),{'width':240,'height':136},Options(model='lanczos'),Upscaler(),check,lambda *args:None)
    assert not output.exists() and not output.with_suffix('.partial.mp4').exists()


def test_transcodes_all_non_aac_tracks(tmp_path):
    source=tmp_path/'pcm.mkv';output=tmp_path/'aac.mp4'
    make_video(source,codec='pcm_s16le',audio_tracks=2)
    engine=Upscaler()
    try:
        process_video(source,output,probe(source),{'width':240,'height':136},Options(model='lanczos'),engine,lambda:None,lambda *args:None)
    finally:engine.close()
    with av.open(str(output)) as container:
        assert len(container.streams.audio)==2
        assert all(s.codec_context.name=='aac' for s in container.streams.audio)
        assert [s.metadata.get('language') for s in container.streams.audio]==['zho','eng']
    for index in range(2):
        with av.open(str(output)) as container:
            assert abs(sum(f.samples for f in container.decode(audio=index))-12000)<=1024


def test_real_neural_inference():
    # Real official checkpoint, strict loading, and a genuine CUDA forward pass.
    engine=Upscaler()
    source=np.random.default_rng(4).integers(30,220,(35,63,3),dtype=np.uint8)
    try:
        base,detail,info=engine.enhance(source,(96,54),'auto',64,lambda:None)
        assert info['backend']=='realesrgan' and info['model']=='RealESRGAN_x2plus.pth'
        assert len(info['sha256'])==64 and base.shape==detail.shape==(54,96,3)
        assert np.isfinite(detail).all() and float(np.mean(abs(detail)))>1
    finally:engine.close()

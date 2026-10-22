from __future__ import annotations

import hashlib
import threading

import av
import cv2
import numpy as np
import pytest

from app.engine import Upscaler
from app.media import probe, process_video
from app.pipeline import VideoPipeline
from app.target import Options
from tests.fixtures import make_video, video_times
from tests.reference_v1_engine import Upscaler as ReferenceUpscaler
from tests.reference_v1_media import process_video as reference_process_video


@pytest.mark.parametrize('tile',[768,1024,1536,2048])
def test_extended_tiles(tile):
    assert Options(tile=tile).tile==tile
    with pytest.raises(ValueError):Options(tile=2049)


@pytest.mark.parametrize('model',['auto','anime'])
def test_transfer_optimization_exact(model):
    image=np.random.default_rng(34).integers(0,255,(95,131,3),dtype=np.uint8)
    old,new=ReferenceUpscaler(),Upscaler()
    try:
        original=old.enhance(image,(196,142),model,64,lambda:None)
        optimized=new.enhance(image,(196,142),model,64,lambda:None)
        assert np.array_equal(original[0],optimized[0])
        assert np.array_equal(original[1],optimized[1])
        assert optimized[2]['transfers']['host_to_device']==1
        assert optimized[2]['transfers']['device_to_host']==1
        assert optimized[2]['transfers']['tiles']>1
    finally:old.close();new.close()


@pytest.mark.parametrize('model',['general-fast','general-clean','anime-fast'])
def test_lightweight_real_checkpoint(model):
    engine=Upscaler()
    try:
        source=np.random.default_rng(6).integers(0,255,(35,63,3),dtype=np.uint8)
        base,detail,info=engine.enhance(source,(96,54),model,64,lambda:None)
        assert info['model_id']==model and info['native_scale']==4
        assert info['native_size']==[252,140] and np.isfinite(detail).all()
    finally:engine.close()


def test_fallback_tile_remembered(monkeypatch):
    import torch
    engine=Upscaler()
    attempted=[]
    def load(key):
        engine.device='cpu';engine.digest='test';engine.key=key
    def tiled(image,scale,tile,check):
        attempted.append(tile)
        if tile>512:raise torch.cuda.OutOfMemoryError('test allocation failure')
        return cv2.resize(image,(image.shape[1]*scale,image.shape[0]*scale))
    monkeypatch.setattr(engine,'load',load)
    monkeypatch.setattr(engine,'_tiled',tiled)
    source=np.zeros((32,48,3),np.uint8)
    engine.enhance(source,(96,64),'auto',2048,lambda:None)
    first=list(attempted);attempted.clear()
    result=engine.enhance(source,(96,64),'auto',2048,lambda:None)
    assert first==[2048,1024,512] and attempted==[512]
    assert result[2]['requested_tile']==2048 and result[2]['tile']==512


def test_pipeline_matches_reference_video_and_audio(tmp_path):
    source=tmp_path/'source.mp4';a=tmp_path/'reference.mp4';b=tmp_path/'pipeline.mp4'
    make_video(source,vfr=True)
    options=Options(model='auto',tile=64)
    metadata=probe(source)
    old,new=ReferenceUpscaler(),Upscaler()
    try:
        reference_process_video(source,a,metadata,{'width':240,'height':136},options,old,lambda:None,lambda *args:None)
        process_video(source,b,metadata,{'width':240,'height':136},options,new,lambda:None,lambda *args:None)
    finally:old.close();new.close()
    with av.open(str(a)) as ca,av.open(str(b)) as cb:
        frames_a=[f.to_ndarray(format='bgr24') for f in ca.decode(video=0)]
        frames_b=[f.to_ndarray(format='bgr24') for f in cb.decode(video=0)]
    assert len(frames_a)==len(frames_b)==6
    assert all(np.array_equal(x,y) for x,y in zip(frames_a,frames_b))
    assert np.allclose(video_times(a),video_times(b),atol=1/90000)
    def audio_hash(path):
        with av.open(str(path)) as c:
            return hashlib.sha256(b''.join(bytes(p) for p in c.demux(audio=0) if p.dts is not None)).hexdigest()
    assert audio_hash(a)==audio_hash(b)


def test_pipeline_propagates_decoder_failure():
    def decode():
        yield 1
        raise ValueError('decoder test failure')
    with pytest.raises(ValueError,match='decoder test failure'):
        with VideoPipeline(decode,lambda event:None,lambda:None,lambda:None) as pipeline:
            for event in pipeline:pipeline.submit(event)
    assert not any(t.name in {'video-decode','video-encode'} for t in threading.enumerate())

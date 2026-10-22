from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


def make_video(path: Path, width=160, height=90, frames=6, rate=24, audio=True, vfr=False, codec='aac', audio_tracks=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    interval = Fraction(90000, 1) / Fraction(rate)
    timestamps = [round(i * interval) + (750 if vfr and i >= frames//2 else 0) for i in range(frames)]
    length = float((timestamps[-1] + interval) / 90000)
    with av.open(str(path), 'w') as output:
        video = output.add_stream('libx264', rate=rate)
        video.width, video.height = width, height
        video.pix_fmt = 'yuv420p' if width % 2 == 0 and height % 2 == 0 else 'yuv444p'
        video.codec_context.time_base = Fraction(1, 90000)
        video.options = {'crf': '16', 'bf': '0'}
        if audio:
            audio_streams=[]
            for track in range(audio_tracks):
                stream=output.add_stream(codec,rate=48000)
                stream.layout='stereo'
                stream.codec_context.time_base=Fraction(1,48000)
                stream.metadata['language']='zho' if track==0 else 'eng'
                audio_streams.append(stream)
        for i, pts in enumerate(timestamps):
            yy, xx = np.mgrid[:height, :width]
            arr = np.stack([(xx * 255 / width), (yy * 255 / height), ((xx + yy + i * 3) % 128) * 2], axis=2).astype(np.uint8)
            cv2.rectangle(arr, (int(width*.2)+i, int(height*.2)), (int(width*.5)+i, int(height*.6)), (130, 185, 210), -1)
            cv2.putText(arr, 'VIDEO 2026', (max(2,width//15),height*4//5), cv2.FONT_HERSHEY_SIMPLEX, max(.25,width/700), (230,230,230), max(1,width//700), cv2.LINE_AA)
            frame=av.VideoFrame.from_ndarray(arr, format='rgb24')
            frame.pts,frame.time_base=pts,Fraction(1,90000)
            output.mux(video.encode(frame))
        output.mux(video.encode(None))
        if audio:
            for stream in audio_streams:
                n=round(length*48000)
                for start in range(0,n,1024):
                    count=min(1024,n-start)
                    wave=(np.sin(np.arange(start,start+count)*440*2*np.pi/48000)*.15).astype(np.float32)
                    frame=av.AudioFrame.from_ndarray(np.stack([wave,wave]),format='fltp',layout='stereo')
                    frame.sample_rate=48000;frame.pts=start;frame.time_base=Fraction(1,48000)
                    output.mux(stream.encode(frame))
                output.mux(stream.encode(None))
    return timestamps


def video_times(path):
    with av.open(str(path)) as container:
        return [float(f.pts*f.time_base) for f in container.decode(video=0)]

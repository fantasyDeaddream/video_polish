"""Bounded-memory PyAV decoding/encoding; preserve presentation times and audio."""
from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .engine import TemporalDetails


def rotation_of(frame, stream):
    rotation = getattr(frame, 'rotation', 0) or stream.metadata.get('rotate', 0)
    rotation = int(round(float(rotation) / 90) * 90) % 360
    return rotation


def rotated(array, rotation):
    if rotation == 90:
        return cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(array, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
    return array


def probe(path: Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('没有可解码的视频轨道')
        stream = container.streams.video[0]
        if stream.codec_context.width * stream.codec_context.height > 34_000_000:
            raise ValueError('输入超过 3400 万像素上限')
        frame = next(container.decode(stream), None)
        if frame is None:
            raise ValueError('视频没有可解码帧')
        if int(stream.codec_context.color_trc) in (16, 18):
            raise ValueError('当前版本只处理 SDR 视频；HDR/PQ/HLG 请先进行正确的色调映射')
        if frame.interlaced_frame:
            raise ValueError('当前版本要求逐行视频；请先对隔行视频去隔行')
        rotation = rotation_of(frame, stream)
        width, height = frame.width, frame.height
        sar = float(stream.sample_aspect_ratio or Fraction(1))
        if rotation in (90, 270):
            width, height, sar = height, width, 1 / sar
        rate = stream.average_rate or stream.guessed_rate or Fraction(25)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration or 0) / av.time_base
        audio = list(container.streams.audio)
        return {'width': width, 'height': height, 'coded_width': frame.width, 'coded_height': frame.height,
                'sar': sar, 'rotation': rotation, 'fps': round(float(rate), 5), 'rate': str(rate),
                'duration': round(duration, 6), 'frames': stream.frames or None,
                'codec': stream.codec_context.name, 'audio_tracks': len(audio),
                'audio_codecs': [s.codec_context.name for s in audio], 'size_bytes': path.stat().st_size,
                'pixel_format': frame.format.name}


def process_video(source, destination, metadata, target, options, upscaler, check, progress):
    if options.interpolation.enabled:
        from .interpolated_media import process_interpolated_video
        return process_interpolated_video(source, destination, metadata, target, options, upscaler, check, progress)
    started = time.monotonic()
    size = target['width'], target['height']
    stabilizer = TemporalDetails(options.strength, options.temporal)
    count, backend = 0, {}
    partial = destination.with_suffix('.partial.mp4')
    time_base = Fraction(1, 90000)
    # Keep all audio tracks. AAC is packet-copied; other codecs are converted to AAC.
    try:
        with av.open(str(source)) as reader, av.open(str(partial), 'w', options={'movflags': '+faststart'}) as writer:
            video = reader.streams.video[0]
            video.thread_type = 'AUTO'
            rate = Fraction(metadata['rate'])
            out = writer.add_stream('libx264', rate=rate)
            out.width, out.height = size
            out.pix_fmt = 'yuv420p'
            out.time_base = out.codec_context.time_base = time_base
            out.codec_context.thread_count = 4
            out.codec_context.sample_aspect_ratio = Fraction(1)
            out.codec_context.colorspace = 1
            out.codec_context.color_primaries = 1
            out.codec_context.color_trc = 1
            out.codec_context.color_range = 1
            out.options = {'crf': str(options.crf), 'preset': 'medium', 'bf': '0'}
            audios = {}
            for audio in reader.streams.audio:
                if audio.codec_context.name == 'aac':
                    audios[audio.index] = (writer.add_stream_from_template(audio), None)
                else:
                    sample_rate = audio.codec_context.sample_rate or 48000
                    if sample_rate not in {7350, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000}:
                        sample_rate = 48000
                    encoded = writer.add_stream('aac', rate=sample_rate)
                    layout = audio.codec_context.layout
                    # PCM containers often describe only "2 channels", which AAC
                    # cannot encode as an unspecified channel layout.
                    known = {'mono', 'stereo', '2.1', '3.0', '4.0', 'quad', '5.0', '5.1', '6.1', '7.1'}
                    encoded.layout = layout.name if layout.name in known else {1:'mono', 2:'stereo', 3:'2.1', 4:'quad', 5:'5.0', 6:'5.1', 8:'7.1'}.get(len(layout.channels), 'stereo')
                    encoded.codec_context.time_base = Fraction(1, sample_rate)
                    encoded.bit_rate = 192000
                    resampler = av.AudioResampler(format='fltp', layout=encoded.layout.name, rate=sample_rate)
                    audios[audio.index] = (encoded, resampler)
                audios[audio.index][0].metadata.update(audio.metadata)
            origin = Fraction(reader.start_time or 0, av.time_base)
            # Prefer the container's common origin, preserving A/V start offsets.
            last_pts = None
            first_timestamp = None
            last_timestamp = None
            def decode_events():
                for packet in reader.demux([video, *reader.streams.audio]):
                    if packet.stream.index == video.index:
                        for frame in packet.decode():
                            yield ('video', frame)
                    elif packet.stream.index in audios:
                        yield ('audio', packet, packet.stream.index)

            def encode_event(event):
                check()
                if event[0] == 'video':
                    _, image, base, detail, pts, duration = event
                    # CPU optical flow, detail fusion, color conversion and
                    # encoding run in order while the GPU handles the next frame.
                    result = stabilizer.apply(image, base, detail)
                    output_frame = av.VideoFrame.from_ndarray(result, format='bgr24')
                    output_frame.pts, output_frame.time_base = pts, time_base
                    output_frame = output_frame.reformat(format='yuv420p', dst_colorspace='ITU709')
                    if duration:
                        output_frame.duration = duration
                    output_frame.colorspace = 1
                    output_frame.color_range = 1
                    writer.mux(out.encode(output_frame))
                else:
                    _, packet, index = event
                    audio_out, resampler = audios[index]
                    if resampler is None:
                        if packet.dts is not None:
                            shift = round(origin / packet.time_base)
                            packet.pts = packet.pts - shift if packet.pts is not None else None
                            packet.dts -= shift
                            packet.stream = audio_out
                            writer.mux(packet)
                    else:
                        for audio_frame in packet.decode():
                            if audio_frame.pts is not None:
                                audio_frame.pts -= round(origin / audio_frame.time_base)
                            for resampled in resampler.resample(audio_frame):
                                for audio_packet in audio_out.encode(resampled):
                                    writer.mux(audio_packet)
            def finish_encode():
                for audio_out, resampler in audios.values():
                    if resampler is not None:
                        for resampled in resampler.resample(None):
                            writer.mux(audio_out.encode(resampled))
                        writer.mux(audio_out.encode(None))
                writer.mux(out.encode(None))

            def transform_event(event):
                nonlocal count, backend, last_pts, last_timestamp, first_timestamp
                check()
                if event[0] != 'video':
                    return event
                frame = event[1]
                if frame.width != metadata['coded_width'] or frame.height != metadata['coded_height']:
                    raise ValueError('视频中途改变分辨率，暂不支持')
                timestamp = Fraction(frame.pts) * frame.time_base - origin if frame.pts is not None else Fraction(count, 1) / rate
                pts = round(timestamp / time_base)
                if last_pts is not None and pts <= last_pts:
                    raise ValueError('视频时间戳重复或倒退，无法保证正确同步')
                image = rotated(frame.to_ndarray(format='bgr24'), metadata['rotation'])
                base, detail, backend = upscaler.enhance(image, size, options.model, options.tile, check,
                                                       tensorrt=options.tensorrt, tensorrt_precision=options.tensorrt_precision)
                duration = max(1, round(frame.duration * frame.time_base / time_base)) if frame.duration else 0
                count += 1
                last_pts, last_timestamp = pts, float(timestamp)
                if first_timestamp is None:
                    first_timestamp = float(timestamp)
                progress(count, max(0, float(timestamp)), time.monotonic() - started, backend)
                return ('video', image, base, detail, pts, duration)

            if options.pipeline:
                from .pipeline import VideoPipeline
                with VideoPipeline(decode_events, encode_event, finish_encode, check) as pipeline:
                    for event in pipeline:
                        pipeline.submit(transform_event(event))
            else:
                for event in decode_events():
                    encode_event(transform_event(event))
                finish_encode()
            if not count:
                raise ValueError('没有成功处理的视频帧')
        check()
        partial.replace(destination)
        return {'frames': count, 'elapsed_seconds': round(time.monotonic() - started, 3), 'engine': backend, 'pipeline':options.pipeline,
                'scene_cuts': stabilizer.scene_cuts, 'audio_tracks': metadata['audio_tracks'],
                'audio': 'AAC copied; other codecs converted to AAC', 'first_timestamp': first_timestamp,
                'last_timestamp': last_timestamp, 'temporal_mean_weight': round(float(np.mean(stabilizer.mean_weights)), 4) if stabilizer.mean_weights else 0}
    finally:
        partial.unlink(missing_ok=True)

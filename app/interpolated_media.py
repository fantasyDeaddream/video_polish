"""SR-originals then RIFE at final size, with a single ordered GPU owner."""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np

from .engine import TemporalDetails
from .framerate import FrameClock, interpolation_plan, output_time_base
from .interpolation import RifeInterpolator, is_scene_cut
from .media import rotated
from .pipeline import VideoPipeline


@dataclass
class SourceFrame:
    timestamp: Fraction
    duration: Fraction
    original: object
    result: object
    prepared: object = None


def _audio_streams(reader, writer):
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
            known = {'mono', 'stereo', '2.1', '3.0', '4.0', 'quad', '5.0', '5.1', '6.1', '7.1'}
            encoded.layout = layout.name if layout.name in known else {1: 'mono', 2: 'stereo', 3: '2.1', 4: 'quad', 5: '5.0', 6: '5.1', 8: '7.1'}.get(len(layout.channels), 'stereo')
            encoded.codec_context.time_base = Fraction(1, sample_rate)
            encoded.bit_rate = 192000
            audios[audio.index] = (encoded, av.AudioResampler(format='fltp', layout=encoded.layout.name, rate=sample_rate))
        audios[audio.index][0].metadata.update(audio.metadata)
    return audios


def process_interpolated_video(source, destination, metadata, target, options, upscaler, check, progress):
    started = time.monotonic()
    settings = options.interpolation
    planned = interpolation_plan(metadata, settings)
    rate, source_rate = Fraction(planned['rate']), Fraction(metadata['rate'])
    time_base = output_time_base(rate)
    size = target['width'], target['height']
    stabilizer = TemporalDetails(options.strength, options.temporal)
    partial = destination.with_suffix('.partial.mp4')
    count, output_count, cuts, repeated, held, anchors = 0, 0, 0, 0, 0, 0
    previous, clock, interpolator = None, None, None
    backend = {'backend': 'identity', 'reason': 'interpolation_only'}
    first_timestamp = last_timestamp = end_timestamp = None
    last_encoded_pts = None
    try:
        with av.open(str(source)) as reader, av.open(str(partial), 'w', options={'movflags': '+faststart'}) as writer:
            video = reader.streams.video[0]
            video.thread_type = 'AUTO'
            origin = Fraction(reader.start_time or 0, av.time_base)
            source_end = ((video.start_time or 0) + video.duration) * video.time_base - origin if video.duration is not None else None
            out = writer.add_stream('libx264', rate=rate)
            out.width, out.height, out.pix_fmt = *size, 'yuv420p'
            out.time_base = out.codec_context.time_base = time_base
            out.codec_context.thread_count = 4
            out.codec_context.sample_aspect_ratio = Fraction(1)
            out.codec_context.colorspace = out.codec_context.color_primaries = out.codec_context.color_trc = 1
            out.codec_context.color_range = 1
            out.options = {'crf': str(options.crf), 'preset': 'medium', 'bf': '0'}
            audios = _audio_streams(reader, writer)

            def decode_events():
                for packet in reader.demux([video, *reader.streams.audio]):
                    if packet.stream.index == video.index:
                        for frame in packet.decode():
                            yield 'video', frame
                    elif packet.stream.index in audios:
                        yield 'audio', packet, packet.stream.index

            def encode_event(event):
                nonlocal last_encoded_pts
                check()
                if event[0] == 'video':
                    _, image, timestamp, duration = event
                    pts = round(timestamp / time_base)
                    if last_encoded_pts is not None and pts <= last_encoded_pts:
                        raise ValueError('输出时间戳重复或倒退')
                    frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                    frame.pts, frame.time_base = pts, time_base
                    frame = frame.reformat(format='yuv420p', dst_colorspace='ITU709')
                    frame.colorspace = frame.color_range = 1
                    frame.duration = max(1, round(duration / time_base))
                    writer.mux(out.encode(frame))
                    last_encoded_pts = pts
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
                        for frame in packet.decode():
                            if frame.pts is not None:
                                frame.pts -= round(origin / frame.time_base)
                            for resampled in resampler.resample(frame):
                                writer.mux(audio_out.encode(resampled))

            def finish_encode():
                for audio_out, resampler in audios.values():
                    if resampler is not None:
                        for resampled in resampler.resample(None):
                            writer.mux(audio_out.encode(resampled))
                        writer.mux(audio_out.encode(None))
                writer.mux(out.encode(None))

            def notify():
                info = dict(backend)
                info['interpolation'] = {'model': settings.model, 'backend': settings.backend, 'rate': str(rate),
                                         'output_frames': output_count, 'generated_frames': interpolator.generated if interpolator else 0}
                timestamp = last_timestamp - first_timestamp if first_timestamp is not None else Fraction(0)
                progress(count, float(timestamp), time.monotonic() - started, info)

            def emit_image(emit, image, timestamp, duration=None):
                nonlocal output_count
                check()
                emit(('video', image, timestamp, duration or 1 / rate))
                output_count += 1

            def consume(event, emit):
                nonlocal previous, clock, count, backend, interpolator, cuts, repeated, held, anchors
                nonlocal first_timestamp, last_timestamp
                check()
                if event[0] != 'video':
                    emit(event)
                    return
                frame = event[1]
                if (frame.width, frame.height) != (metadata['coded_width'], metadata['coded_height']):
                    raise ValueError('视频中途改变分辨率，暂不支持')
                timestamp = Fraction(frame.pts) * frame.time_base - origin if frame.pts is not None else count / source_rate
                if previous is not None and timestamp <= previous.timestamp:
                    raise ValueError('视频时间戳重复或倒退，无法保证正确同步')
                image = rotated(frame.to_ndarray(format='bgr24'), metadata['rotation'])
                if options.workflow == 'interpolate':
                    result = image if (image.shape[1], image.shape[0]) == size else cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
                else:
                    base, detail, backend = upscaler.enhance(image, size, options.model, options.tile, check,
                                                           tensorrt=options.tensorrt, tensorrt_precision=options.tensorrt_precision)
                    result = stabilizer.apply(image, base, detail)
                duration = Fraction(frame.duration) * frame.time_base if frame.duration else Fraction(1, 1) / source_rate
                current = SourceFrame(timestamp, duration, image, result)
                count += 1
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                last_timestamp = timestamp
                if previous is None:
                    clock = FrameClock(rate, timestamp)
                else:
                    cut = settings.scene_cut_detection and is_scene_cut(previous.original, current.original)
                    identical = np.array_equal(previous.result, current.result)
                    cuts += bool(cut)
                    repeated += bool(identical)
                    for output_time, alpha in clock.positions(previous.timestamp, timestamp):
                        if alpha == 0:
                            value = previous.result
                            anchors += 1
                        elif cut or identical:
                            value = previous.result
                            held += 1
                        else:
                            if interpolator is None:
                                interpolator = RifeInterpolator(settings, check, upscaler.stage_cb)
                            if previous.prepared is None:
                                previous.prepared = interpolator.prepare(previous.result)
                            if current.prepared is None:
                                current.prepared = interpolator.prepare(current.result)
                            value = interpolator.interpolate(previous.prepared, current.prepared, alpha)
                        emit_image(emit, value, output_time)
                previous = current
                notify()

            def finish_timeline(emit):
                nonlocal end_timestamp, held, anchors
                if previous is None:
                    raise ValueError('没有成功处理的视频帧')
                end_timestamp = source_end if source_end is not None and source_end > previous.timestamp else previous.timestamp + previous.duration
                for timestamp, alpha in clock.positions(previous.timestamp, end_timestamp):
                    if alpha == 0:
                        anchors += 1
                    else:
                        held += 1
                    emit_image(emit, previous.result, timestamp, min(1 / rate, end_timestamp - timestamp))
                notify()

            if options.pipeline:
                with VideoPipeline(decode_events, encode_event, finish_encode, check) as pipeline:
                    for event in pipeline:
                        consume(event, pipeline.submit)
                    finish_timeline(pipeline.submit)
            else:
                for event in decode_events():
                    consume(event, encode_event)
                finish_timeline(encode_event)
                finish_encode()
        check()
        partial.replace(destination)
        details = interpolator.info() if interpolator else {'model': settings.model, 'backend': settings.backend,
                   'generated_frames': 0, 'reason': 'only anchors, repeated frames or scene cuts; no neural inference needed'}
        details.update(enabled=True, rate=str(rate), fps=float(rate), output_frames=output_count,
                       anchor_frames=anchors, held_frames=held, scene_cuts=cuts, repeated_pairs=repeated,
                       order='sr_then_vfi', source_end=float(end_timestamp), tail_policy='hold_last_frame',
                       duration_seconds=float(end_timestamp - first_timestamp))
        return {'frames': count, 'output_frames': output_count, 'elapsed_seconds': round(time.monotonic() - started, 3),
                'engine': backend, 'pipeline': options.pipeline, 'workflow': options.workflow, 'interpolation': details,
                'scene_cuts': cuts, 'audio_tracks': metadata['audio_tracks'], 'audio': 'AAC copied; other codecs converted to AAC',
                'first_timestamp': float(first_timestamp), 'last_timestamp': float(last_timestamp)}
    except Exception as exc:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise RuntimeError('插帧显存不足。可选择 lite、减小插帧内部尺度，或降低目标尺寸；不会自动降低画质设置。') from exc
        raise
    finally:
        previous = None
        if interpolator:
            interpolator.close()
        partial.unlink(missing_ok=True)

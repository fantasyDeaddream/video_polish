"""Exact target-frame scheduling; input PTS, rather than average fps, brackets t."""
from __future__ import annotations

import math
from fractions import Fraction


def parse_rate(text):
    try:
        if len(str(text)) > 24:
            raise ValueError()
        rate = Fraction(str(text))
        if not 1 <= rate <= 240 or rate.numerator > 1_000_000 or rate.denominator > 100_000:
            raise ValueError()
        return rate
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError('目标帧率应为 1–240，可输入 60 或 60000/1001') from exc


def interpolation_plan(metadata, options):
    source = Fraction(metadata['rate'])
    if not options.enabled:
        return {'enabled': False, 'source_rate': str(source), 'rate': str(source), 'fps': float(source)}
    rate = source * int(options.target_fps[0]) if options.target_fps in {'2x', '4x'} else parse_rate(options.target_fps)
    if rate > 240:
        raise ValueError('插帧输出超过 240fps，请降低目标帧率')
    if rate < source:
        raise ValueError(f'目标帧率 {rate} 低于原片 {source}；插帧模式不执行降帧')
    duration = Fraction(str(metadata.get('duration') or 0))
    count = math.ceil(duration * rate) if duration else None
    return {'enabled': True, 'source_rate': str(source), 'rate': str(rate), 'fps': round(float(rate), 6),
            'estimated_frames': count, 'duration_seconds': float(duration), 'model': options.model,
            'backend': options.backend, 'order': 'sr_then_vfi', 'note': '保持播放时长；按原始 PTS 重采样，末帧尾段保持。'}


def output_time_base(rate):
    denominator = math.lcm(90000, rate.numerator)
    if denominator > 2_000_000_000:
        denominator = rate.numerator * 1000
    return Fraction(1, denominator)


class FrameClock:
    def __init__(self, rate, start):
        self.rate, self.start = Fraction(rate), Fraction(start)
        self.index = 0

    def positions(self, left, right):
        """Yield a half-open interval, so each target sample appears exactly once."""
        left, right = Fraction(left), Fraction(right)
        if right <= left:
            raise ValueError('视频时间戳重复或倒退')
        while (timestamp := self.start + self.index / self.rate) < right:
            if timestamp < left:
                raise ValueError('目标时钟落在输入帧区间之外')
            self.index += 1
            yield timestamp, (timestamp - left) / (right - left)

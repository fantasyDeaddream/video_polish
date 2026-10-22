"""Explicit image comparisons; these measure agreement, not recovered truth."""
import cv2
import numpy as np


def compare(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError('Quality comparison requires identical shapes')
    delta = candidate.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(delta * delta, dtype=np.float64))
    result = {'identical':bool(np.array_equal(reference, candidate)),
              'mae':float(np.mean(np.abs(delta), dtype=np.float64)),
              'max_abs':float(np.max(np.abs(delta))),
              'changed_percent':float(np.count_nonzero(delta) / delta.size * 100),
              'psnr_db':float(10 * np.log10(255**2 / mse)) if mse else None}
    if reference.ndim == 3:
        a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        a, b = reference.astype(np.float32), candidate.astype(np.float32)
    ma = cv2.GaussianBlur(a, (11,11), 1.5)
    mb = cv2.GaussianBlur(b, (11,11), 1.5)
    va = cv2.GaussianBlur(a*a, (11,11), 1.5) - ma*ma
    vb = cv2.GaussianBlur(b*b, (11,11), 1.5) - mb*mb
    cov = cv2.GaussianBlur(a*b, (11,11), 1.5) - ma*mb
    c1, c2 = (.01*255)**2, (.03*255)**2
    score = ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
    if min(score.shape) > 10:
        score = score[5:-5,5:-5]
    result['ssim_luma'] = float(np.mean(score, dtype=np.float64))
    return result

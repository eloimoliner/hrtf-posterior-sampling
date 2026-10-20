"""Python version of Julius O. Smith's mps (minimum-phase spectrum): https://ccrma.stanford.edu/~jos/fp/Matlab_listing_mps_m.html"""
import numpy as np
import numpy.fft as fft

ft = fft.fft
ift = fft.ifft


def mps(s, cutoff_dB, axis):
    """Minimum-phase spectrum from the complex spectrum s."""
    n = len(s.shape)
    permutation = list(range(n))
    permutation[n - 1] = axis
    permutation[axis] = n - 1
    sm = s.transpose(permutation)

    sm = clipdb(sm, cutoff_dB)
    sm = np.log(sm)
    sm = ift(sm, axis=-1)
    sm = fold(sm)
    sm = ft(sm, axis=-1)
    sm = np.exp(sm)

    return sm.transpose(permutation)


def clipdb(s, cutoff_dB):
    """Floors the magnitude of s at cutoff_dB relative to its peak (https://ccrma.stanford.edu/~jos/fp/Matlab_listing_clipdb_m.html)."""
    assert cutoff_dB < 0
    clipped = s
    as_ = np.abs(s)
    mas = np.max(as_.flatten())
    if mas == 0:
        return clipped
    thresh = mas * 10 ** (cutoff_dB / 20)
    clipped[as_ < thresh] = thresh
    return clipped


def fold(r):
    """Folds the left wing of a vector in FFT buffer format onto the right wing (https://ccrma.stanford.edu/~jos/fp/Matlab_listing_fold_m.html)."""
    n = r.shape[-1]
    if n < 3:
        return r
    assert np.mod(n, 2) == 0

    rf = np.concatenate((r[..., 1:r.shape[-1] // 2], np.zeros((*r.shape[:-1], 1), dtype=r.dtype)), axis=-1)
    rf += np.conj(r[..., -1:r.shape[-1] // 2 - 1:-1])
    rw = np.concatenate((r[..., :1], rf, np.zeros((*r.shape[:-1], r.shape[-1] // 2 - 1), dtype=r.dtype)), axis=-1)
    return rw

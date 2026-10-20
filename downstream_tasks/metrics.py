"""HRTF metrics of Table I (pointwise, on complex spectra); averaged over ears and frequencies in scripts/evaluate.py."""
import numpy as np


def pointwise_relative_error_dB(x_hat, x):
    """LRE: 10 log10(|x_hat - x|^2 / |x|^2)."""
    error = x_hat - x
    squared_error = (np.conj(error) * error).real
    eps = np.finfo(squared_error.dtype).eps
    return 10 * np.log10(squared_error + eps) - 10 * np.log10((np.conj(x) * x).real + eps)


def pointwise_log_magnitude_distance_dB(x_hat, x):
    """LMD: |10 log10(|x_hat|^2) - 10 log10(|x|^2)|."""
    pow_hat = (np.conj(x_hat) * x_hat).real
    pow = (np.conj(x) * x).real
    eps = np.finfo(pow.dtype).eps
    return np.abs(10 * np.log10(pow_hat + eps) - 10 * np.log10(pow + eps))


def mean_std(data):
    """Mean and standard deviation, ignoring NaNs."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return np.mean(data), np.std(data)

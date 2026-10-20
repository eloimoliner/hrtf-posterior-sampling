"""Feature transforms applied to the observation y and its estimate y_hat before computing the loss."""
import logging

import einops
import torch

logger = logging.getLogger(__name__)


def compressed_stft(x, win_length, hop_length, fft_length, compression_factor):
    """STFT with power-law compressed magnitude |X|^compression_factor."""
    assert compression_factor is not None and 0. < compression_factor <= 1., f"Unexpected compression factor value: {compression_factor}"

    window = torch.hann_window(win_length).to(x.device)
    X = stft(x, window, fft_length, win_length, hop_length, pad=False, normalize=True, window_prepaded=False)
    X = (X.abs() + 1e-8) ** compression_factor * torch.exp(1j * X.angle())
    return X


def stft(x, window, n_fft, win_length, hop_length, pad=True, normalize=True, window_prepaded=True):
    """
    x:          tensor with shape (batch, channels, samples)
    window:     tensor with shape (n_fft,) if window_prepaded, else (win_length,)
    returns:    tensor with shape (batch, channels, n_fft // 2 + 1, frames)
    """
    B, C, T = x.shape
    x = einops.rearrange(x, 'b c t -> (b c) t', b=B, c=C)

    if pad:
        xpadded = torch.nn.functional.pad(x, (0, win_length))
    else:
        xpadded = x

    # with a zero-padded window, win_length=n_fft is passed to torch.stft
    if window_prepaded:
        X = torch.stft(xpadded, n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=True,
                       onesided=True, return_complex=True, normalized=False, pad_mode='constant')
    else:
        X = torch.stft(xpadded, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True,
                       onesided=True, return_complex=True, normalized=False, pad_mode='constant')

    if normalize:
        X /= torch.sqrt(torch.sum(window ** 2))

    X = einops.rearrange(X, '(b c) f t -> b c f t', b=B, c=C)
    return X


def istft(X, window, n_fft, win_length, hop_length, length=None, pad=True, normalize=True, window_prepaded=True):
    """
    X:          tensor with shape (batch, channels, n_fft // 2 + 1, frames)
    returns:    tensor with shape (batch, channels, samples)
    """
    if length is None:
        logger.warning('length is None, istft may crash')
        length_param = None
    elif pad:
        length_param = length + win_length // 2
    else:
        length_param = length

    window = window.to(X.device)
    B, C, _, _ = X.shape
    X = einops.rearrange(X, 'b c f t -> (b c) f t', b=B, c=C)

    if normalize:
        X *= torch.sqrt(torch.sum(window ** 2))
    if window_prepaded:
        x = torch.istft(X, n_fft, hop_length=hop_length, win_length=n_fft, window=window, onesided=True, center=True,
                        normalized=False, return_complex=False, length=length_param)
    else:
        x = torch.istft(X, n_fft, hop_length=hop_length, win_length=win_length, window=window, onesided=True,
                        center=True, normalized=False, return_complex=False, length=length_param)

    if pad:
        x = x[..., win_length // 2:]  # delay caused by centering the STFT

    x = einops.rearrange(x, '(b c) t -> b c t', b=B, c=C)
    return x

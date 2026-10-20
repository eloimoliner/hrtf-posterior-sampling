import logging

import librosa
import numpy as np
import torch
from omegaconf.listconfig import ListConfig

from utils.coordinate_system import cart2sph__matlab

logger = logging.getLogger(__name__)


def apply_in_place_on_each_tensor(tensor_collection, in_place_transform, tensor_names=None):
    """
    Applies in_place_transform to each tensor of a dict (optionally only to those in tensor_names) or of a list.
    """
    if isinstance(tensor_collection, dict):
        if tensor_names is not None:
            assert isinstance(tensor_names, list) or isinstance(tensor_names, ListConfig)
            for tensor_name in tensor_names:
                assert tensor_name in tensor_collection.keys(), f'parameter "{tensor_name}" not found in {tensor_collection.keys()}'
            collection = {key: value for key, value in tensor_collection.items() if key in tensor_names}
        else:
            collection = tensor_collection
        for tensor in collection.values():
            in_place_transform(tensor)

    elif isinstance(tensor_collection, list):
        assert tensor_names is None, 'tensor collection is a list: cannot filter by parameter name'
        for tensor in tensor_collection:
            in_place_transform(tensor)

    else:
        raise ValueError


def get_azimuth_from_metadata(file_doa):
    """Azimuth in degrees, in (-180, 180], from a cartesian DoA file."""
    doa = np.load(file_doa)
    azimuth = cart2sph__matlab(*doa)[0]  # in [0, 2 pi)
    azimuth = 360 - azimuth * 180 / np.pi
    if azimuth > 180:
        azimuth = azimuth - 360
    return torch.tensor(azimuth)


def get_itdist_from_metadata(file_anthropometry):
    """Inter-tragus distance (HUTUBS anthropometric measure x1) in metres."""
    anthropometry = np.load(file_anthropometry, allow_pickle=True)
    it_dist = anthropometry.item().get('x1') / 100
    return torch.tensor(it_dist)


def delay_from_doa_and_itdist(doa, itd_ist, sample_rate):
    """ITD in samples from the azimuth (degrees) and the inter-tragus distance (m)."""
    c = 343
    delay_s = torch.sin(doa * torch.pi / 180) * itd_ist / c
    return delay_s * sample_rate


def device(force_cpu):
    if force_cpu:
        return torch.device('cpu')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dtype(str):
    if str == 'float32':
        return torch.float32
    elif str == 'float64':
        return torch.float64
    else:
        raise ValueError


def time_alignment_brir(brir, hrir, upsample_factor=8):
    """
    Delay of the BRIR w.r.t. the HRIR (cross-correlation peak, at upsample_factor times the sample rate).
    Returns the BRIR advanced by that delay and the delay in samples.
    """
    if upsample_factor == 1:
        Y, X = brir, hrir
    else:
        Y = librosa.resample(brir.detach().cpu().numpy(), orig_sr=1, target_sr=upsample_factor, axis=-1)
        X = librosa.resample(hrir.detach().cpu().numpy(), orig_sr=1, target_sr=upsample_factor, axis=-1)
        Y, X = torch.tensor(Y), torch.tensor(X)

    if Y.shape[-1] > X.shape[-1]:
        X = torch.nn.functional.pad(X, (0, Y.shape[-1] - X.shape[-1]))
    assert Y.shape == X.shape

    n_taps = Y.shape[-1]
    Y = torch.fft.rfft(Y, n=n_taps + n_taps - 1, dim=-1)
    X = torch.fft.rfft(X, n=n_taps + n_taps - 1, dim=-1)
    ic_0 = torch.fft.irfft(torch.conj(X[0]) * Y[0])
    ic_1 = torch.fft.irfft(torch.conj(X[1]) * Y[1])

    delay = (ic_0 + ic_1).argmax().item()
    delay = delay // upsample_factor

    omegas = 2 * torch.pi * torch.arange(0, n_taps // 2 + 1, device=hrir.device) / n_taps

    def apply_delay(x, d, n_pad):
        h = torch.exp(-1j * omegas * (n_pad + d))
        x_f = torch.fft.rfft(x, n=n_pad, dim=-1)
        return torch.fft.irfft(h * x_f, n=n_pad, dim=-1)

    y_aligned = apply_delay(brir, -delay, n_pad=n_taps)
    return y_aligned, delay

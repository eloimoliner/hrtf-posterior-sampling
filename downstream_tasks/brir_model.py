"""Forward model: speech s -> reverberant binaural speech y, through the parametric BRIR model."""
import logging

import torch
import torchcde

from datasets.pointwise_hrir import to_complex
from downstream_tasks.feature_transforms import istft

logger = logging.getLogger(__name__)


def fast_convolve(s, h, mode):
    """
    FFT convolution of a single-channel signal with one or more filters.

    s:          tensor with shape (1, 1, n_taps_1)
    h:          filter(s) with shape (channels, n_taps_2) or (n_taps_2,)
    mode:       'same' (output length n_taps_1) or 'full' (n_taps_1 + n_taps_2 - 1)
    returns:    tensor with shape (1, channels, output length)
    """
    assert s.ndim == 3, 'expecting a input signal with shape (batch, channels, taps)'
    assert s.shape[:2] == (1, 1), 'expects signal with single batch and channel'

    def convolve_(s_, h_):
        B = h_.unsqueeze(0).unsqueeze(0).to(s_.device)

        N = s_.size(2)
        M = B.size(2)
        fft_size = torch.tensor(N + M - 1)
        fft_size = int(2**torch.ceil(torch.log2(fft_size)))

        X = torch.fft.fft(s_, fft_size, dim=2)
        H = torch.fft.fft(B, fft_size, dim=2)
        y = torch.fft.ifft(X * H, fft_size, dim=2).real

        if mode == 'same':
            y = y[..., :N]
        elif mode == 'full':
            y = y[..., :N + M - 1]
        else:
            raise ValueError('mode must be either "same" or "full"')
        return y

    if h.ndim == 2:
        y = torch.concatenate([convolve_(s, h_=fir) for fir in h], dim=1)
    elif h.ndim == 1:
        y = convolve_(s, h_=h)
    else:
        raise ValueError
    return y


def time_aligned_hrtf_to_hrir(TA_HRTF, delay, fixed_delay_offset=0.0):
    """Time-aligned HRTF (4, n_bins - 1) -> HRIR (2, taps), with the ITD delay applied to the later ear and both ears delayed by fixed_delay_offset."""
    H_ipd_ild = to_complex(TA_HRTF)
    # add the Nyquist bin (zero)
    H_ipd_ild = torch.cat((H_ipd_ild, torch.zeros((2, 1), device=H_ipd_ild.device, dtype=H_ipd_ild.dtype)), dim=-1)

    n_bins = H_ipd_ild.shape[-1]
    fft_size = 2 * (n_bins - 1)
    omegas = 2 * torch.pi * torch.arange(0, n_bins, device=TA_HRTF.device) / fft_size
    H_direct_path_left = H_ipd_ild[0, :] * torch.exp(-1j * omegas * (torch.nn.ReLU()(delay) + fixed_delay_offset))
    H_direct_path_right = H_ipd_ild[1, :] * torch.exp(-1j * omegas * (torch.nn.ReLU()(-delay) + fixed_delay_offset))
    H_direct_path = torch.stack((H_direct_path_left, H_direct_path_right))

    hrir = torch.fft.irfft(H_direct_path, dim=-1)
    return hrir


def measurement_model(s, hrtf, psi, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                            sample_rate, mixing_time_in_seconds, fixed_delay_offset, hrtf_sigma, hrtf_mu):
    """Forward model with the HRTF given in normalised features (as produced by the diffusion prior)."""
    assert hrtf.ndim == 3, 'expecting tensor hrtf of shape (batch, channels, frequency)'
    assert hrtf.shape[0] == 1, 'expecting single batch signal'

    hrtf = hrtf * hrtf_sigma.unsqueeze(0).to(hrtf.device) + hrtf_mu.unsqueeze(0).to(hrtf.device)
    psi_prime = {**psi, 'hrtf': hrtf[0]}

    return convolve_with_brir_model(s, psi_prime, n_fft, hop_length, win_length, Nf, EQ_freqs,
                                                                  fix_EQ_extremes, sample_rate, mixing_time_in_seconds, fixed_delay_offset)


def convolve_with_brir_model(s, psi, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                                                           sample_rate, mixing_time_in_seconds, fixed_delay_offset):
    """
    Reverberant binaural measurement y_hat = h(psi)_a * s with the BRIR model of Eq. (2):

        h(psi)_a = delta_{t_left} * (g * [delta_0, delta_{t_itd}] * F^-1(a) + r(chi))

    a is the time-aligned HRTF, g the gain, t_itd the ITD and t_left the time of flight to the left ear. r is the
    reverberant component of Eq. (3), with decays and weights shared by both ears and phases per ear; it is faded in
    over the mixing time and starts fixed_delay_offset samples after the direct path. Early reflections are not modelled.
    """
    assert s.ndim == 3, 'expecting tensor s of shape (batch, channels, taps)'
    assert s.shape[1] == 1, 'expecting single channel signal'
    assert isinstance(psi, dict), 'expecting psi to be a dictionary'
    assert psi['hrtf'].shape[0] == 4, 'expecting 4 channels: 2 ears * (real, imaginary)'

    # direct path
    t_left = psi['t_left'].to(s.device)
    t_itd = psi['t_itd'].to(s.device)
    hrir = time_aligned_hrtf_to_hrir(psi['hrtf'], t_itd, t_left)
    h_direct_path = psi['g'] * hrir

    # reverberant component r
    window = torch.hann_window(win_length).to(s.device)
    window = torch.nn.functional.pad(window, (0, n_fft - win_length), mode='constant', value=0)

    psi_l = {'phases': psi['phases_left'], 'decays': psi['decays'], 'weights': psi['weights']}
    psi_r = {'phases': psi['phases_right'], 'decays': psi['decays'], 'weights': psi['weights']}

    H_l = reverberant_component_stft(psi_l, window, n_fft, win_length, hop_length, Nf, EQ_freqs, sample_rate)
    H_r = reverberant_component_stft(psi_r, window, n_fft, win_length, hop_length, Nf, EQ_freqs, sample_rate)
    r_l = reverberant_component_from_stft(H=H_l, window_padded=window, n_fft=n_fft, win_length=win_length, hop_length=hop_length, Nf=Nf)
    r_r = reverberant_component_from_stft(H=H_r, window_padded=window, n_fft=n_fft, win_length=win_length, hop_length=hop_length, Nf=Nf)
    r = torch.stack((r_l, r_r))

    # fade-in up to the mixing time
    n_mixing_taps = int(mixing_time_in_seconds * sample_rate)
    r[:, :n_mixing_taps] = torch.arange(0.0, 1.0, 1 / n_mixing_taps, device=s.device, dtype=s.dtype) * r[:, :n_mixing_taps]

    # delay the diffuse part by fixed_delay_offset, pad the direct path to the same length, and sum
    r = torch.cat((torch.zeros((2, fixed_delay_offset), device=s.device, dtype=s.dtype), r), dim=-1)
    n_zeros = r.shape[-1] - h_direct_path.shape[-1]
    h_direct_path = torch.cat((h_direct_path, torch.zeros((2, n_zeros), device=s.device, dtype=s.dtype)), dim=-1)
    h = h_direct_path + r

    y = fast_convolve(s, h, mode='same')
    return y


def reverberant_component_stft(psi, window, n_fft, win_length, hop_length, Nf, EQ_freqs, sample_rate):
    """
    STFT of the diffuse RIR from its parameters.

    psi:      {"phases": (n_fft // 2 + 1, Nf), "decays": (bands,), "weights": (bands,)}, bands = len(EQ_freqs) - 2
    returns:    complex tensor with shape (n_fft // 2 + 1, Nf)
    """
    phases, decay, weights = psi['phases'], psi['decays'], psi['weights']

    device = phases.device
    assert phases.device == decay.device == weights.device, 'expecting all parameters to be on the same device'

    # band edges / centre frequencies in Hz, and FFT frequencies
    EQ_freqs = torch.tensor(EQ_freqs).to(torch.float32).to(device)
    freqs = torch.fft.rfftfreq(n_fft, d=1 / sample_rate).to(device)

    # magnitude: exponential decay per band over the Nf frames, zero at the first and last band
    time_indices = torch.arange(0, Nf, dtype=torch.float, device=decay.device)
    exponential_decay_curves = torch.exp(decay).unsqueeze(0) ** (-time_indices.unsqueeze(-1))

    damped_magnitudes = weights.unsqueeze(0) * exponential_decay_curves
    damped_magnitudes = torch.cat((torch.zeros((Nf, 1), device=device, dtype=decay.dtype),
                                   damped_magnitudes,
                                   torch.zeros((Nf, 1), device=device, dtype=decay.dtype)),
                                  dim=-1)
    log_damped_magnitudes = torch.log(damped_magnitudes + 1e-6)

    # interpolate the log magnitudes linearly over frequency
    coeffs = torchcde.linear_interpolation_coeffs(log_damped_magnitudes.unsqueeze(-1))
    linear_interpolator = torchcde.LinearInterpolation(coeffs, t=EQ_freqs)
    A = linear_interpolator.evaluate(freqs)
    A = A.squeeze(-1).transpose(0, 1)
    A = torch.exp(A)
    A = A.clone()  # avoids an in-place operation error in correct_OLA

    if torch.isnan(A).any():
        logger.warning('encountered nan values in interpolated magnitude spectrogram')

    A = correct_OLA(A, window, win_length, hop_length)

    H = A * torch.exp(1j * phases)
    return H


def reverberant_component_from_stft(H, window_padded, n_fft, win_length, hop_length, Nf):
    """Inverse STFT of the RIR spectrogram H with shape (n_fft // 2 + 1, Nf); returns the RIR in the time domain."""
    H = torch.nn.functional.pad(H, (1, 1))

    assert window_padded.shape[0] == n_fft, 'expecting window_padded to have the same length as n_fft'
    length_rir = hop_length * Nf

    H = H.unsqueeze(0).unsqueeze(0)
    h = istft(H, window_padded, n_fft, win_length, hop_length, length=length_rir, pad=False, normalize=False)
    h = h.squeeze(0).squeeze(0)
    h = torch.nn.functional.pad(h, (0, hop_length))
    return h


def correct_OLA(A, window, win_length, hop_length):
    """
    Compensates the first frames for the partial window overlap: with a hop of 25% of the window, the RIR onset is at
    75%, 50% and 25% of the first three windows, which would otherwise make the decay too steep at the start.
    """
    K = int(win_length / (hop_length) - 1)
    win_sum = torch.sum(window)
    for k in range(0, K):
        A[:, k] /= win_sum / torch.sum(window[int((K - k) * hop_length):])
    return A

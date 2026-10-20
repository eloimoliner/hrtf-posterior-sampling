"""Initialisers of the HRTF diffusion state and of the BRIR model parameters psi."""
import logging

import torch

from datasets.pointwise_hrir import to_time_aligned_hrtf_helper
from downstream_tasks.feature_transforms import stft

logger = logging.getLogger(__name__)


def initial_diffusion_state(shape, device, sampler):
    x_T = sampler.initialize_x(shape=tuple(shape), device=device, schedule=sampler.create_schedule())
    return x_T


def init_reverberant_component(decays_init, weights_init, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                      sample_rate, device="cpu"):
    """Single-channel BUDDy RIR parameters: {"decays", "weights", "phases"}."""
    if fix_EQ_extremes:
        num_bands = len(EQ_freqs) - 2
    else:
        num_bands = len(EQ_freqs)

    # decays: T60 in seconds -> decay per STFT frame (6.908 = ln(1000), i.e. 60 dB)
    if len(decays_init) == 1:
        decays_init = num_bands * [decays_init[0]]
    else:
        assert len(decays_init) == num_bands, "decays_init must be a list of length 1 or num_bands"
    decays_init = torch.Tensor(decays_init)
    decays_init = 6.908 / (decays_init * (sample_rate / hop_length))
    decays_init = decays_init.detach().to(device)
    decays = torch.nn.Parameter(decays_init, requires_grad=True)

    # weights
    if len(weights_init) == 1:
        weights_init = num_bands * [weights_init[0]]
    else:
        assert len(weights_init) == num_bands, "weights_init must be a list of length 1 or num_bands"
    weights_init = torch.Tensor(weights_init).detach().to(device)
    weights = torch.nn.Parameter(weights_init, requires_grad=True)

    # phases: those of the STFT of white noise
    length_rir = hop_length * Nf
    noise = torch.randn((length_rir)).to(device)
    window = torch.hann_window(win_length).to(device)
    window = torch.cat((window, torch.zeros(n_fft - win_length).to(device)))
    N = stft(noise.view(1, 1, -1), window, n_fft, win_length, hop_length, pad=False, normalize=True, window_prepaded=True)
    N = N[0, 0, :, 1:]
    phases_init = torch.angle(N).detach().to(device)
    phases = torch.nn.Parameter(phases_init, requires_grad=True)

    return dict(decays=decays, weights=weights, phases=phases)


def init_reverberant_component_binaural(decays_init, weights_init, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                             independent_weights, independent_decays, independent_phases, sample_rate, device="cpu", seed=42):
    """Stereo BUDDy RIR parameters: init_reverberant_component for each ear, keeping one set of the shared parameters."""
    torch.manual_seed(seed)

    psi_left = init_reverberant_component(decays_init=decays_init, weights_init=weights_init, n_fft=n_fft,
                                   hop_length=hop_length, win_length=win_length, Nf=Nf, EQ_freqs=EQ_freqs,
                                   fix_EQ_extremes=fix_EQ_extremes, sample_rate=sample_rate, device=device)
    psi_right = init_reverberant_component(decays_init=decays_init, weights_init=weights_init, n_fft=n_fft,
                                    hop_length=hop_length, win_length=win_length, Nf=Nf, EQ_freqs=EQ_freqs,
                                    fix_EQ_extremes=fix_EQ_extremes, sample_rate=sample_rate, device=device)
    psi = dict()

    if independent_weights:
        psi.update(dict(weights_left=psi_left['weights'], weights_right=psi_right['weights']))
    else:
        psi.update(dict(weights=psi_left['weights']))

    if independent_decays:
        psi.update(dict(decays_left=psi_left['decays'], decays_right=psi_right['decays']))
    else:
        psi.update(dict(decays=psi_left['decays']))

    if independent_phases:
        psi.update(dict(phases_left=psi_left['phases'], phases_right=psi_right['phases']))
    else:
        psi.update(dict(phases=psi_left['phases']))

    return psi


def init_brir_parameters(decays_init, weights_init, g_init, t_itd_init, t_left_init,
                         n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes, sample_rate, device):
    """
    BRIR model parameters psi: reverberant component (decays and weights shared by the ears, phases per ear),
    ITD t_itd, gain g and time of flight to the left ear t_left (in samples).
    """
    psi = init_reverberant_component_binaural(decays_init, weights_init, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                                              independent_weights=False,
                                              independent_decays=False,
                                              independent_phases=True,
                                              sample_rate=sample_rate, device=device)

    t_itd = torch.tensor(t_itd_init).to(device)
    t_itd = torch.nn.Parameter(t_itd, requires_grad=True)
    logger.info(f'initializing t_itd to: {t_itd}')
    psi.update(dict(t_itd=t_itd))

    g = torch.ones((1,), device=device) * g_init
    g = torch.nn.Parameter(g, requires_grad=True)
    psi.update(dict(g=g))

    t_left = torch.nn.Parameter(torch.tensor(t_left_init).to(device), requires_grad=True)
    psi.update(dict(t_left=t_left))
    return psi


def time_aligned_hrtf(hrir, n_taps, sample_rate):
    """Time-aligned HRTF of a (2, taps) HRIR: returns (ITD delay in samples, (1, 4, n_taps // 2 + 1) real features)."""
    hrir = hrir.cpu().numpy()
    delay_vec, cpx_spec = to_time_aligned_hrtf_helper(hrir, sample_rate, n_taps)

    delay_vec = torch.from_numpy(delay_vec)
    cpx_spec = torch.from_numpy(cpx_spec)

    if torch.min(delay_vec) > 1:
        logger.warning("The hrir has a delay of more than 1 sample")
    delay = torch.max(delay_vec)
    if torch.argmax(delay_vec) == 1:  # right ear delayed
        delay = -delay

    ta_HRTF = torch.view_as_real(cpx_spec)
    ta_HRTF = ta_HRTF.permute(2, 0, 1).contiguous().view(4, -1)

    return delay, ta_HRTF.unsqueeze(0)

"""HRTF estimation with a score-based prior (arXiv 2410.01562): the experiment of Table I, at 44.1 kHz.

Posterior sampling of the time-aligned HRTF with the DoA-conditioned diffusion prior (EDM, Euler steps with DPS
likelihood guidance), alternated with the optimisation of the parametric BRIR model at each diffusion step.
"""
import logging
import random
from functools import partial

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from downstream_tasks.brir_model import measurement_model, fast_convolve
from downstream_tasks.initialization import init_brir_parameters, time_aligned_hrtf
from downstream_tasks.utils import get_itdist_from_metadata, delay_from_doa_and_itdist, get_azimuth_from_metadata, time_alignment_brir

logger = logging.getLogger(__name__)


@hydra.main(config_path="./", config_name="config", version_base=str(hydra.__version__))
def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    config = hydra.utils.instantiate(config)

    estimate_hrtf(**config.kwargs, device=config.device, dtype=config.dtype)


def estimate_hrtf(speech_file, brir_file, hrir_file, doa_file, anthropometry_file, save_hrtf_file, save_gt_hrtf_file, save_parameters_file,
                  sample_rate, sampler, initial_state, likelihood_loss, losses, gradient_transforms,
                  optimizer, post_update_transforms, zeta, n_its, device, dtype, hrtf_sigma, hrtf_mu,
                  EQ_freqs, decays_init, weights_init, g_init, t_left_init, n_fft, hop_length, win_length, Nf,
                  mixing_time_in_seconds, fixed_delay_offset):
    """Estimates the time-aligned HRTF a from the reverberant binaural measurement y = BRIR * s and the excitation signal s."""

    logger.info(f'device is {device}')
    assert sample_rate == 44100
    n_taps = 256

    # Load HRIR (ground truth, only used for evaluation and for the BRIR time alignment)
    with open(hrir_file, 'rb') as f:
        x_hrir, sr = np.load(f, allow_pickle=True).item().values()
    if sr != sample_rate:
        x_hrir = librosa.resample(x_hrir, orig_sr=sr, target_sr=sample_rate)
    x_hrir = torch.tensor(x_hrir).float().to(device)

    # Load BRIR
    x_brir, sr = sf.read(brir_file)
    if sr != sample_rate:
        x_brir = librosa.resample(x_brir.T, orig_sr=sr, target_sr=sample_rate).T
    x_brir = torch.tensor(x_brir).float().permute(1, 0).to(device)

    _, prop_delay = time_alignment_brir(x_brir, x_hrir)

    # Load the excitation signal s (dry speech)
    x_speech, sr = sf.read(speech_file)
    if sr != sample_rate:
        x_speech = librosa.resample(x_speech, orig_sr=sr, target_sr=sample_rate)
    x_speech = torch.tensor(x_speech).float().view(1, 1, -1).to(device)

    # Initialise the BRIR model parameters psi; the ITD is initialised from the DoA
    doa = get_azimuth_from_metadata(doa_file)
    it_dist = get_itdist_from_metadata(anthropometry_file)
    t_itd_init = delay_from_doa_and_itdist(doa, it_dist, sample_rate)

    fix_EQ_extremes = True
    psi = init_brir_parameters(decays_init, weights_init, g_init, t_itd_init, t_left_init,
                               n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes, sample_rate, device)

    # Ground-truth time-aligned HRTF (for evaluation)
    _, gt_HRTF = time_aligned_hrtf(x_hrir, n_taps=n_taps, sample_rate=sample_rate)
    hrtf_gt = gt_HRTF[..., 0:-1].squeeze(0)

    # Measurement y: speech convolved with the BRIR
    y = fast_convolve(x_speech, x_brir, mode="same")

    # Model of the measurement: speech * BRIR model h(psi)_a
    simulate_measurement = lambda s, hrtf, psi_i: measurement_model(s, hrtf, psi_i, n_fft, hop_length, win_length, Nf, EQ_freqs, fix_EQ_extremes,
                                                                    sample_rate, mixing_time_in_seconds, fixed_delay_offset, hrtf_sigma, hrtf_mu)

    logger.info('making sampler')
    _, sampler = sampler(device=device)

    doa = np.load(doa_file)
    doa = torch.tensor(doa).to(device).unsqueeze(0).to(dtype)
    sampler.load_DoA(doa)

    # Initial HRTF: an unconditional prior sample plus the sampler's initial noise
    datum_hat = sampler.predict_unconditional(shape=(4, 4, 128), device=device)
    hrtf_sigma = hrtf_sigma.to(device)
    hrtf_mu = hrtf_mu.to(device)
    datum_hat *= hrtf_sigma
    datum_hat += hrtf_mu
    hrtf_T = datum_hat[0].unsqueeze(0) + initial_state(sampler=sampler, device=device)

    logger.info('setting up optimizer')
    for key, value in psi.items():
        assert value.is_leaf, f'parameter {key} of the BRIR model must be leaf'
        assert value.requires_grad, f'parameter {key} of the BRIR model must be gradient-optimizable'
    optimizer = optimizer(params=[tensor for tensor in psi.values()])

    logger.info('setting-up scheduler')
    t_hrtf = sampler.create_schedule().to(device)
    gamma_hrtf = sampler.get_gamma(t_hrtf, Stmin=sampler.Stmin, Stmax=sampler.Stmax, Schurn=sampler.Schurn).to(device)

    logger.info('running posterior sampling')
    step = partial(posterior_sampling_step, sampler=sampler, simulate_measurement=simulate_measurement, psi=psi,
                   losses=losses, gradient_transforms=gradient_transforms,
                   optimizer=optimizer, likelihood_loss=likelihood_loss,
                   zeta=zeta, n_its=n_its,
                   post_update_transforms=post_update_transforms)

    # The dry speech, delayed by the propagation delay to match the observation
    x_speech_aligned = torch.cat([torch.zeros(1, 1, prop_delay).to(device), x_speech], dim=-1)
    x_speech_aligned = x_speech_aligned[:, :, 0:y.shape[-1]]
    x_i = x_speech_aligned.to(device)

    hrtf_i = hrtf_T.to(device)

    for i in tqdm(range(0, sampler.T, 1)):
        hrtf_i_prime, t_hrtf_i_prime = sampler.stochastic_timestep(x=hrtf_i, t=t_hrtf[i], gamma=gamma_hrtf[i])
        _, hrtf_hat_i, hrtf_i = step(y=y, hrtf_i_prime=hrtf_i_prime, x_i_prime=x_i, t_hrtf_i_prime=t_hrtf_i_prime, t_hrtf_i_plus_1=t_hrtf[i + 1])

    # The estimate is the last Tweedie estimate (normalised features; de-normalise with hrtf_mu/hrtf_sigma)
    hrtf_final = hrtf_hat_i.squeeze(0).detach().cpu().numpy()
    np.save(save_hrtf_file, hrtf_final)
    torch.save(psi, save_parameters_file)
    np.save(save_gt_hrtf_file, hrtf_gt.squeeze(0).detach().cpu().numpy())


def log_likelihood_gradient(squared_error_norm, x, zeta, retain_graph=False):
    """Gradient of the reconstruction error w.r.t. the diffusion state x, scaled by zeta * sqrt(sum(x.shape)) / ||grad||."""
    grad = torch.autograd.grad(outputs=squared_error_norm, inputs=x, retain_graph=retain_graph)[0]
    grad = grad.detach()
    diffusion_guidance = lambda z: z * ((sum(x.shape) ** 0.5) / torch.norm(grad) + 1e-8)
    likelihood_score = diffusion_guidance(zeta) * grad
    return likelihood_score


def posterior_sampling_step(y, hrtf_i_prime, x_i_prime, t_hrtf_i_prime, t_hrtf_i_plus_1, sampler, simulate_measurement, psi, losses,
                            gradient_transforms, optimizer, post_update_transforms, likelihood_loss, zeta, n_its):
    """
    One step n of the inference algorithm (Fig. 1): n_its optimisation iterations of the BRIR parameters psi given the
    one-step denoising (Tweedie) estimate a_0_hat, then a step of the posterior ODE (9) with the log-likelihood gradient.
    [1]: Karras et al., 2022 (EDM), [2]: Chung et al., 2023 (DPS).

    returns: squared error norm, Tweedie estimate hrtf_hat, next diffusion state hrtf_i_plus_1
    """
    assert not hrtf_i_prime.requires_grad
    hrtf_i_prime.requires_grad = True

    # step 7 from Algorithm 2 [1] (EDM): denoised estimate D(x_i_prime; t_i_prime)
    hrtf_hat = sampler.get_Tweedie_estimate(hrtf_i_prime, t_hrtf_i_prime)

    x_hat = x_i_prime  # known dry speech

    psi = optimize_brir_parameters(y=y, hrtf=hrtf_hat.clone().detach(), x=x_hat.clone().detach(), simulate_measurement=simulate_measurement,
                                   psi=psi, n_its=n_its, losses=losses,
                                   gradient_transforms=gradient_transforms,
                                   optimizer=optimizer, post_update_transforms=post_update_transforms)

    # DPS
    # computing the update term of step 7 from the Algorithm 1 in [2] (DPS)
    y_hat = simulate_measurement(x_hat, hrtf_hat, psi)
    squared_error_norm = likelihood_loss(x=y, x_hat=y_hat)

    log_likelihood_grad = log_likelihood_gradient(squared_error_norm=squared_error_norm, x=hrtf_i_prime, zeta=zeta, retain_graph=False)

    hrtf_i_prime = hrtf_i_prime.detach()

    score_hrtf = sampler.Tweedie2score(hrtf_hat, hrtf_i_prime, t_hrtf_i_prime)
    d_i_hrtf = sampler.diff_params._ode_integrand(hrtf_i_prime, t_hrtf_i_prime, score_hrtf)

    # adding update term of step 7 from the Algorithm 1 in [2] (DPS)
    d_i_hrtf += log_likelihood_grad

    # EDM
    # step 8 from Algorithm 2 [1]
    hrtf_i_plus_1 = hrtf_i_prime + (t_hrtf_i_plus_1 - t_hrtf_i_prime) * d_i_hrtf

    return squared_error_norm, hrtf_hat.detach(), hrtf_i_plus_1.detach()


def optimize_brir_parameters(y, hrtf, x, simulate_measurement, psi, n_its, losses,
                             gradient_transforms, optimizer, post_update_transforms):
    """
    Adam updates of the BRIR model parameters psi (n_its iterations).

    x:              speech with shape (batch, channels, time), where channels=1 for single channel audio
    hrtf:           current HRTF Tweedie estimate (fixed here)
    psi:            parameters of the BRIR model, dict of torch tensors
    """
    for _ in range(n_its):

        for key, value in psi.items():
            assert value.requires_grad, f'parameter {key} of the BRIR model must be gradient-optimizable'

        y_hat = simulate_measurement(x, hrtf, psi)
        assert y_hat.shape == y.shape, 'expecting matching shapes for y_hat and y'

        loss_terms = {key: value(x_hat=y_hat, x=y) for key, value in losses.items()}
        loss = 0.0
        for value in loss_terms.values():
            loss += value

        if torch.isnan(loss).any():
            logger.warning('nan found in loss')

        optimizer.zero_grad()
        loss.backward()

        for transform in gradient_transforms.values():
            with torch.no_grad():
                transform(psi)

        optimizer.step()

        for transform in post_update_transforms.values():
            with torch.no_grad():
                transform(psi)

    return psi


if __name__ == "__main__":
    main()

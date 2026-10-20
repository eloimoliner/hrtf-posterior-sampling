"""Time-aligned HRTF features (HRIR spectrum with the pure delay removed), and the training dataloader of the prior."""
import json
import os
import random
from functools import partial

import librosa
import numpy as np
import torch
from numpy.fft import fft as ft
from numpy.fft import ifft as ift
from torch.utils.data import DataLoader, Dataset

from datasets.mps import mps


def to_time_aligned_hrtf_helper(h, sampling_rate, n_taps):
    """Returns the pure delay (samples) of each HRIR in h (..., taps) and its complex spectrum envelope (..., n_taps // 2 + 1)."""

    def get_group_delay(z, fftsize):
        """
        computes group delay https://ccrma.stanford.edu/~jos/fp/Group_Delay.html
            D(z, omega) = - d/domega angle(z)(omega)

        using finite differences

        returns group quantified as number of samples
        """
        phasor = np.roll(z, shift=-1) / z
        delta_omega = 2 * np.pi / fftsize

        gp = - np.angle(phasor) / delta_omega
        return gp

    def pure_delay_wegd(h, fftsize, frequencies, sampling_rate, minimum_phase_processor, axis=-1):
        """
        Weighted excess group delay (WEGD) method for extraction of pure delay component of hrirs according to

            Nam et al. - 2008 - A Method for Estimating Interaural Time Difference for Binaural Synthesis.

        Result is the power spectrum weighted average of the group delay over specified frequencies

        :param frequencies:                     None -> weighted excess group delay over the entire audio range
                                                [min freq. (Hz), max freq. (Hz)] -> over the specified frequency range
        :param sampling_rate:                   sampling rate of h
        :minimum_phase_processor:               processor taking H=ft(h, fftsize) and returning minimum phase version
        :param axis:                            axis of filter taps in h
        :return:                                returns the pure delay (in samples)
         """

        permutation = list(range(len(h.shape)))
        permutation[-1] = axis
        permutation[axis] = len(h.shape) - 1

        h = h.transpose(permutation)

        # computing minimum phase (mp) component
        H = ft(h, n=fftsize, axis=-1)
        h_mp = minimum_phase_processor(h)
        H_mp = ft(h_mp, n=fftsize, axis=-1)

        carrier = H / H_mp
        group_delay = get_group_delay(carrier, fftsize)

        # power-weighted average of the group delay
        if isinstance(frequencies, tuple) or isinstance(frequencies, list):
            n0 = int(frequencies[0] * fftsize / sampling_rate)
            nf = int(frequencies[1] * fftsize / sampling_rate)
            pure_delay = (group_delay[..., n0:nf + 1] * np.abs(H[..., n0:nf + 1]) ** 2).sum(-1, keepdims=True) / \
                         np.abs(np.abs(H[..., n0:nf + 1]) ** 2).sum(-1, keepdims=True)

        else:
            assert frequencies is None
            pure_delay = (group_delay * np.abs(H ** 2)).sum(-1, keepdims=True) / np.abs(np.abs(H) ** 2).sum(-1, keepdims=True)

        return pure_delay.transpose(permutation)

    def minimum_phase_jos(h, fftsize, mps_cutoff_dB, discard_second_half=False):
        """
        :param h:                               set of hrir
        :param mps_cutoff_dB:                   parameter of JOS's minimum phase spectrum function
        :return:                                set of minimum phase HRTF spectra, with shape (..., frequency bins)
        """
        H = ft(h, n=fftsize, axis=-1)
        H_mp = mps(H, cutoff_dB=mps_cutoff_dB, axis=-1)
        h_mp = ift(H_mp, axis=-1).real
        if discard_second_half:
            return h_mp[..., :h.shape[-1] // 2]
        else:
            return h_mp[..., :h.shape[-1]]

    def demodulate(h, pure_delay, sample_rate, fftsize):
        """
        :param h:                           hrir set with shape (..., filter taps)
        :param pure_delay:                  pure delay in samples for each hrir filter
        :param sample_rate:                 sample rate of hrir filters
        :param fftsize:                     int
        :return:                            complex spectrum envelope
        """

        if pure_delay.shape == h.shape:

            def make_demodulating_phasor():
                cum_tau = np.cumsum(np.concatenate((np.zeros((*pure_delay.shape[:-1], 1), dtype=pure_delay.dtype),
                                                    pure_delay[..., 1:fftsize // 2 + 1]), axis=-1), axis=-1)
                return np.exp(2j * np.pi * cum_tau * sample_rate / fftsize)

        else:

            def make_demodulating_phasor():
                omegas = 2 * np.pi * np.arange(fftsize // 2 + 1) / fftsize
                return np.exp(1j * omegas * pure_delay)

        spectrum_envelope = ft(h, n=fftsize)[..., :fftsize // 2 + 1] * make_demodulating_phasor()

        return spectrum_envelope

    assert n_taps >= h.shape[-1], 'Expecting n_taps to introduce zeropadding, here it would truncate the hrir...'

    pure_delay = pure_delay_wegd(h=h,
                                 fftsize=n_taps,
                                 frequencies=[0, 1100],
                                 sampling_rate=sampling_rate,
                                 minimum_phase_processor=partial(minimum_phase_jos,
                                                                 fftsize=n_taps * 8,
                                                                 mps_cutoff_dB=-40),
                                 axis=-1)

    complex_envelope = demodulate(h=h,
                                  pure_delay=pure_delay,
                                  sample_rate=sampling_rate,
                                  fftsize=n_taps)

    return pure_delay, complex_envelope


def to_complex(complex_envelope):
    """(..., 4, bins) real features [re L, re R, im L, im R] -> (..., 2, bins) complex."""
    assert complex_envelope.shape[-2] == 4, 'expecting ear channel to be at axis -2'
    return complex_envelope[..., :2, :] + 1j * complex_envelope[..., 2:, :]


# ---------------------------------------------------------------------------------------------------------------------
# Training data: pointwise (DoA, HRIR) pairs -> (DoA, normalised time-aligned HRTF features)
# ---------------------------------------------------------------------------------------------------------------------

class PointwiseHRIR(Dataset):
    """(coordinates, hrir) pairs listed in <dataset_dir>/<split>_entries.json."""

    def __init__(self, dataset_dir, split='all', cardinality=None, transform=None, seed=0):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.dataset_dir = dataset_dir
        entries_file = {'all': 'dataset_entries.json', 'train': 'train_entries.json',
                        'test': 'test_entries.json', 'valid': 'valid_entries.json'}[split]
        with open(os.path.join(self.dataset_dir, entries_file), 'r') as f:
            self.dataset_entries = json.load(f)

        if cardinality is not None:
            assert cardinality < len(self.dataset_entries)
            self.dataset_entries = self.dataset_entries[:cardinality]

        self.transform = transform

    def __len__(self):
        return len(self.dataset_entries)

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_dir, self.dataset_entries[idx][0]), 'rb') as f:
            coordinates = np.load(f)
        with open(os.path.join(self.dataset_dir, self.dataset_entries[idx][1]), 'rb') as f:
            hrir = np.load(f)

        if self.transform:
            return self.transform((coordinates, hrir))
        else:
            return coordinates, hrir


def to_torch_tensor(args):
    return [torch.tensor(item) for item in args]


def numpy_resample(datapoint_tuple, dataset_sample_rate, target_sample_rate):
    coordinates, hrir = datapoint_tuple
    assert hrir.ndim == 2 and hrir.shape[-1] == 2
    f = partial(librosa.resample, orig_sr=dataset_sample_rate, target_sr=target_sample_rate)
    hrir = np.stack((f(hrir[:, 0]), f(hrir[:, 1])), axis=-1)
    return coordinates, hrir


def whiten(datapoint_tuple, train_set_mu, train_set_sigma, tuple_index):
    x = datapoint_tuple[tuple_index]
    x -= train_set_mu
    x /= train_set_sigma
    return *datapoint_tuple[:tuple_index], x, *datapoint_tuple[tuple_index + 1:]


def to_unit_norm(datapoint_tuple, tuple_index):
    x = datapoint_tuple[tuple_index]
    x /= (x ** 2).sum(-1, keepdim=True) ** 0.5
    return *datapoint_tuple[:tuple_index], x, *datapoint_tuple[tuple_index + 1:]


def to_time_aligned_hrtf(datapoint_tuple, sampling_rate, n_taps):
    coordinates, h = datapoint_tuple
    assert h.ndim == 2 and h.shape[-1] == 2
    h = h.transpose(1, 0)
    pure_delay, complex_envelope = to_time_aligned_hrtf_helper(h, sampling_rate, n_taps)
    return coordinates, pure_delay, complex_envelope


def drop_nyquist(datapoint_tuple):
    coordinates, pure_delay, complex_envelope = datapoint_tuple
    assert complex_envelope.shape[-1] % 2 == 1, 'expecting odd number of bins for positive frequency side of spectrum'
    return coordinates, pure_delay, complex_envelope[..., :-1]


def to_real(datapoint_tuple):
    coordinates, pure_delay, complex_envelope = datapoint_tuple
    assert complex_envelope.shape[0] == 2, 'expecting ear channel to be at axis 0'
    complex_envelope = torch.concatenate((complex_envelope.real, complex_envelope.imag), axis=0)
    return coordinates, pure_delay, complex_envelope


def compose(transforms):
    def f(x):
        for t in transforms:
            x = t(x)
        return x
    return f


def pointwise_time_aligned_hrtf_envelope_dataloader_v2(dataset_dir, split, cardinality, batch_size, n_taps, shuffle, seed,
                                                       num_workers, prefetch_factor, target_sample_rate, dataset_sample_rate,
                                                       dtype, mu, sigma, return_mu_and_sigma=False,
                                                       doa_dropout_rate=None, doa_noise_sigma=None):
    """
    Batches of dict(context=DoA (B, 3), features=normalised time-aligned HRTF (B, 4, n_taps // 2)).

    DoA augmentation: Gaussian noise of std doa_noise_sigma on the unit-norm DoA
    (then renormalised), and the DoA set to zero with probability doa_dropout_rate.
    """

    def collate_fn(batch):
        coordinates, pure_delay, complex_envelope = list(zip(*batch))
        coordinates = torch.stack(coordinates).to(dtype=dtype)
        complex_envelope = torch.stack(complex_envelope).to(dtype=dtype)

        if doa_noise_sigma:
            noise = torch.normal(mean=torch.zeros_like(coordinates), std=doa_noise_sigma * torch.ones_like(coordinates))
            coordinates += noise
            coordinates /= (coordinates ** 2).sum(dim=-1, keepdim=True).sqrt() + torch.finfo(coordinates.dtype).eps

        if doa_dropout_rate is not None:
            assert 0.0 <= doa_dropout_rate <= 1.0
            keep = torch.bernoulli((1.0 - doa_dropout_rate) * torch.ones((coordinates.shape[0], 1)))
            coordinates *= keep

        if return_mu_and_sigma:
            return dict(context=coordinates, features=complex_envelope), mu.to(dtype=dtype), sigma.to(dtype=dtype)
        else:
            return dict(context=coordinates, features=complex_envelope)

    transform_chain = [partial(numpy_resample, dataset_sample_rate=dataset_sample_rate, target_sample_rate=target_sample_rate),
                       partial(to_time_aligned_hrtf, sampling_rate=target_sample_rate, n_taps=n_taps),
                       drop_nyquist,
                       to_torch_tensor,
                       to_real,
                       partial(whiten, train_set_mu=mu, train_set_sigma=sigma, tuple_index=2),  # normalise the features
                       partial(to_unit_norm, tuple_index=0)]                                    # unit-norm DoA

    return DataLoader(dataset=PointwiseHRIR(dataset_dir=dataset_dir, split=split, cardinality=cardinality, seed=seed,
                                            transform=compose(transform_chain)),
                      batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True, timeout=0, prefetch_factor=prefetch_factor,
                      shuffle=shuffle,
                      drop_last=False,
                      collate_fn=collate_fn)

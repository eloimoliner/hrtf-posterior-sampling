"""Baselines of Table I:
  - Random:            HRTF of a random training subject, at the test DoA
  - Generic:           HRTF of the generic subject (pp41, the centroid of the training set), at the test DoA
  - Nearest Neighbour: training HRTF at the test DoA with the lowest mean LRE w.r.t. the true HRTF

Outputs <out_dir>/{random,generic,nearest_neighbour}/<basename>.npy in the same (4, 128) real layout as the
true HRTF saved by hrtf_estimation.py.
"""
import os
from argparse import ArgumentParser
from glob import glob

import librosa
import numpy as np
import sofar
from tqdm import tqdm

from datasets.hutubs import hutubs_simulated_files_from_plit, get_source_position_in_cartesian_coordinates_from_sofa
from datasets.pointwise_hrir import to_time_aligned_hrtf_helper
from downstream_tasks.metrics import pointwise_relative_error_dB

SAMPLE_RATE = 44100
N_TAPS = 256
GENERIC_SUBJECT = 'pp41'


def to_features(hrir, fs):
    """(2, N) HRIR -> (4, 128) real time-aligned HRTF, as for the ground truth in hrtf_estimation.py."""
    if fs != SAMPLE_RATE:
        hrir = librosa.resample(hrir, orig_sr=fs, target_sr=SAMPLE_RATE, axis=-1)
    _, complex_envelope = to_time_aligned_hrtf_helper(hrir, SAMPLE_RATE, N_TAPS)
    complex_envelope = complex_envelope[..., :-1]                      # drop Nyquist
    return np.concatenate((complex_envelope.real, complex_envelope.imag), axis=0).astype(np.float32)


def to_complex(x):
    return x[..., :2, :] + 1j * x[..., 2:, :]


def load_train_set(sofa_dir):
    """Returns {subject: (unit-norm DoAs (M, 3), HRIRs (M, 2, N), fs)} for the HUTUBS simulated training split."""
    train = {}
    for path in sorted(hutubs_simulated_files_from_plit(os.path.join(sofa_dir, 'hutubs'), 'train', 'simulated')):
        s = sofar.read_sofa(path, verify=False)
        doa = get_source_position_in_cartesian_coordinates_from_sofa(s)
        doa = doa / np.sqrt((doa ** 2).sum(-1, keepdims=True))
        train[os.path.basename(path).split('_')[0]] = (doa, s.Data_IR, s.Data_SamplingRate)
    return train


def hrtf_at_doa(subject_entry, doa):
    doas, hrirs, fs = subject_entry
    i = np.argmin(((doas - doa) ** 2).sum(-1))
    assert np.allclose(doas[i], doa, atol=1e-3), 'found no exact match with doa'
    return to_features(hrirs[i], fs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sofa_dir', required=True, help='directory containing hutubs/ (SOFA files)')
    parser.add_argument('--test_dir', required=True, help='test set with hrir/ and doa/ (e.g. test_v3p1_600)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train = load_train_set(args.sofa_dir)
    subjects = sorted(train)
    print(f'{len(subjects)} training subjects')
    assert GENERIC_SUBJECT in train

    for name in ['random', 'generic', 'nearest_neighbour', 'gt']:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)

    for hrir_file in tqdm(sorted(glob(os.path.join(args.test_dir, 'hrir', '*.np')))):
        basename = os.path.basename(hrir_file)[:-len('.np')]
        if not os.path.exists(os.path.join(args.test_dir, 'brir', basename + '_receiver_0.wav')):
            continue                                                   # same 599 tasks as the main pipeline
        hrir, fs = np.load(hrir_file, allow_pickle=True).item().values()
        doa = np.load(os.path.join(args.test_dir, 'doa', basename + '.np'))
        doa = doa / np.sqrt((doa ** 2).sum())
        gt = to_features(hrir, fs)

        candidates = {s: hrtf_at_doa(train[s], doa) for s in subjects}
        mean_lre = {s: pointwise_relative_error_dB(to_complex(c), to_complex(gt)).mean() for s, c in candidates.items()}

        np.save(os.path.join(args.out_dir, 'gt', basename + '.npy'), gt)
        np.save(os.path.join(args.out_dir, 'random', basename + '.npy'), candidates[subjects[rng.integers(len(subjects))]])
        np.save(os.path.join(args.out_dir, 'generic', basename + '.npy'), candidates[GENERIC_SUBJECT])
        np.save(os.path.join(args.out_dir, 'nearest_neighbour', basename + '.npy'), candidates[min(mean_lre, key=mean_lre.get)])

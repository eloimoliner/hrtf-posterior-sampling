"""HUTUBS subject split and SOFA helpers (used by the baselines)."""
import os

import numpy as np

from utils.coordinate_system import sph2cart__matlab

# subject split of the HRTF prior (HUTUBS simulated); pp88 and pp96 repeat pp1 and pp22 and are excluded
TRAIN = ['pp10', 'pp11', 'pp12', 'pp13', 'pp14', 'pp15', 'pp16', 'pp17', 'pp19', 'pp2', 'pp20', 'pp21', 'pp22',
         'pp23', 'pp24', 'pp25', 'pp26', 'pp29', 'pp3', 'pp31', 'pp32', 'pp33', 'pp34', 'pp35', 'pp36', 'pp37',
         'pp38', 'pp39', 'pp40', 'pp41', 'pp42', 'pp43', 'pp44', 'pp45', 'pp46', 'pp47', 'pp48', 'pp49', 'pp5',
         'pp50', 'pp51', 'pp52', 'pp54', 'pp55', 'pp56', 'pp57', 'pp58', 'pp59', 'pp6', 'pp60', 'pp61', 'pp62',
         'pp63', 'pp64', 'pp66', 'pp68', 'pp69', 'pp7', 'pp70', 'pp71', 'pp72', 'pp73', 'pp74', 'pp75', 'pp76',
         'pp77', 'pp78', 'pp79', 'pp8', 'pp80', 'pp81', 'pp82', 'pp83', 'pp84', 'pp85', 'pp86', 'pp87', 'pp89',
         'pp9', 'pp90', 'pp91', 'pp92', 'pp93', 'pp94', 'pp95']
TEST = ['pp28', 'pp30', 'pp4', 'pp53', 'pp65']
VALID = ['pp1', 'pp18', 'pp27', 'pp67']


def hutubs_simulated_files_from_plit(dataset_dir, split, basename_suffix):
    assert basename_suffix in ['simulated', 'measured']
    subjects = {'train': TRAIN, 'test': TEST, 'valid': VALID, 'all': TRAIN + VALID + TEST}[split]
    basenames = [b for b in os.listdir(dataset_dir) if b.split('_')[-1] == f'{basename_suffix}.sofa']
    return [os.path.join(dataset_dir, b) for b in basenames if b.split('_')[0] in subjects]


def get_source_position_in_cartesian_coordinates_from_sofa(sofa_object):

    if sofa_object.SourcePosition_Type == 'cartesian' and sofa_object.SourcePosition_Units == 'meter':

        x1, x2, x3 = sofa_object.SourcePosition.transpose()
        x_out = np.stack((x1, x2, x3), axis=-1)

    elif sofa_object.SourcePosition_Type == 'spherical' and sofa_object.SourcePosition_Units == 'degree, degree, metre':

        azimuth, elevation, radius = sofa_object.SourcePosition.transpose()
        assert np.all(radius >= 1), 'expect hrtf measured at location on sphere of at least unit radius'
        x_out = sph2cart__matlab(azimuth=np.deg2rad(azimuth), elevation=np.deg2rad(elevation), radius=radius)
        x_out = np.stack(x_out, axis=-1)

    return x_out

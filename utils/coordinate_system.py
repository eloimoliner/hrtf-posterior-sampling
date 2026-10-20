import numpy as np


def cart2sph__sfs(x, y, z):
    """Cartesian to spherical coordinates, as in the sfs package: azimuth in [-pi, pi], colatitude in [0, pi], radius."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / r)
    return alpha, beta, r


def cart2sph__matlab(x, y, z):
    """Equivalent of MATLAB's cart2sph: azimuth in [0, 2 pi), elevation, radius."""
    azimuth, elevation, radius = cart2sph__sfs(x=x, y=y, z=z)
    azimuth = np.fmod(azimuth + 2 * np.pi, 2 * np.pi)
    elevation = - elevation + np.pi / 2
    return azimuth, elevation, radius


def sph2cart__matlab(azimuth, elevation, radius):
    """Equivalent of MATLAB's sph2cart."""
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return x, y, z

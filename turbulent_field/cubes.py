import numpy as np
from numpy.fft import fftfreq


def set_r_cube(n_side, L):
    """Compute momentum space 3D cubes of n_side and half-size L

    Arguments:
        n_side -- size of the cube side (normally, a power of 2: 64, 128, or 256)
        L -- [arb. length units] half-scale of the cube side

    Returns:
        coordinate cube of size n_side x n_side x n_side and 2L x 2L x 2L
    """
    # set (endpoint=False) to account for periodic boundary conditions
    xyz = np.array([np.linspace(-L, L, n_side, endpoint=False) for i in range(3)])

    r_grid = np.stack(np.meshgrid(*xyz, indexing='ij'), axis=-1)  # 3D coordinate grid
    return r_grid


def set_k_cube(n_side, L):
    """Compute momentum space 3D cubes of n_side and half-size L

    Arguments:
        n_side -- size of the cube side (normally, a power of 2: 64, 128, or 256)
        L -- [arb. length units] half-scale of the cube side

    Returns:
        momentum cube of size n_side^3 and (2pi N/L)^3
    """
    dx = 2 * L / n_side

    k_x = 2 * np.pi * fftfreq(n_side, d=dx)

    k_grid = np.stack(np.meshgrid(k_x, k_x, k_x, indexing='ij'), axis=-1)
    return k_grid
import numpy as np
from numpy.fft import fftfreq, rfftfreq, irfftn


def set_cubes(n_side, L):
    """Compute coordinate and momentum space 3D cubes of n_side and half-size L

    Arguments:
        n_side -- size of the cube side (normally, a power of 2: 64, 128, or 256)
        L -- [arb. length units] half-scale of the cube side

    Returns:
        tuple: (coordinate cube, momentum cube) of size n_side x n_side x n_side and 2L x 2L x 2L
    """
    n = (n_side, n_side, n_side)  # grid shape

    L_x, L_y, L_z = L, L, L
    xyz_min = (-L_x, -L_y, -L_z)  # [pc], minimal values
    xyz_max = (+L_x, +L_y, +L_z)  # [pc], maximal values

    # set (endpoint=False) to account for periodic boundary conditions
    xyz = np.array([np.linspace(xyz_min[i], xyz_max[i], n[i], endpoint=False) for i in range(len(n))])

    r_grid = np.stack(np.meshgrid(*xyz, indexing='ij'))  # 3D coordinate grid

    Nx, Ny, Nz = r_grid.shape[1:]
    dx, dy, dz = xyz[:, 1] - xyz[:, 0]

    k_x = 2 * np.pi * fftfreq(Nx, d=dx)
    k_y = 2 * np.pi * fftfreq(Ny, d=dy)
    k_z = 2 * np.pi * rfftfreq(Nz, d=dz)

    k_grid = np.stack(
        np.meshgrid(k_x, k_y, k_z, indexing='ij')
    )
    return r_grid, k_grid


def set_isotropic_spectrum(r_grid, k_grid, power_index,
                           k_min=None, k_max=None, seed=None):
    """Set the isotropic spectrum by reweigthing the normally distributed components

    Arguments:
        r_grid -- grid in coordinate space
        k_grid -- corresponding grid in Fourier (momentum) space
        power_index -- isotropic spectrum power index p: P(k) ~ k^{-p}

    Keyword Arguments:
        k_min -- minimal momentum <-> injection scale (default: {None})
        k_max -- maximal momentum <-> dissipation scale (default: {None})
        seed -- seed value for the random generator. If None, the fresh unpredictable state will be set. (default: {None})

    Returns:
        vector field in coordinate space with a known power-spectrum in the momentum space
    """ 
    # Step 1. Generate white noise field
    # (use default numpy random generator)
    
    rng = np.random.default_rng(seed)
    
    b_re = rng.normal(size=k_grid.shape)  # real part
    b_im = rng.normal(size=k_grid.shape)  # imaginary part

    b_field = 1/np.sqrt(2) * (b_re + 1j * b_im)
    
    # Step 2. Reweighting
    k_norm_grid = np.sqrt(np.sum(k_grid**2, axis=0))  # k-vector norms
    
    # define injection and dissipation scales
    k_min = 0 if k_min is None else k_min  # if k_min is undefined set to 0
    
    k_max_grid = np.maximum(k_norm_grid)  # maximal value on the grid
    k_max = k_max_grid if k_max is None else k_max # if k_max is undefined, set to max value on the grid
    
    mask = (k_norm_grid > k_min) * (k_norm_grid < k_max)  # cut the spectrum between k_min and k_max

    power_grid = np.zeros_like(k_norm_grid)  # set the power grid
    power_grid[mask] = k_norm_grid[mask]**(-power_index)  # set spectrum

    b_prime_field = b_field * np.sqrt(power_grid[None, ...])
    
    # Step 3. Orthogonal projection
    n_grid = np.zeros_like(k_grid)
    n_grid[:, mask] = k_grid[:, mask] / k_norm_grid[None, mask]

    n_dot_b = np.sum(n_grid * b_prime_field, axis=0)  # scalar product

    B_k_field = b_prime_field - n_grid * n_dot_b[None, ...]  # divergence-free field
    B_k_field[:, 0, 0, 0] = np.array([0, 0, 0])  # send the zero-mode to 0
    
    # Step 4. Inverse Fourier Transform
    B_field = np.zeros_like(r_grid)
    for i in range(3):
        B_field[i] = irfftn(B_k_field[i], s=r_grid.shape[1:], norm='backward')
    
    B_norm = np.average(B_field**2, axis=(1,2,3))
    B_field /= np.sqrt(3 * B_norm)[:, None, None, None]
        
    return B_field


def get_field_cube(n_side, L, power_index=11/3):
    r_grid, k_grid = set_cubes(n_side, L)
    b_field = set_isotropic_spectrum(r_grid, k_grid, power_index)
    return b_field

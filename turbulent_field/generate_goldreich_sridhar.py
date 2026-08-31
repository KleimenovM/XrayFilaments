import numpy as np
from numpy.fft import ifftn

from cubes import set_k_cube


def set_anisotropic_spectrum(k_grid, k_inj, model,
                             power_index = 10/3, seed=None):
    """Set the isotropic spectrum by reweigthing the normally distributed components

    Arguments:
        k_grid -- corresponding grid in Fourier (momentum) space
        k_inj  -- turbulence injection scale
        model -- "Chandran" (10.1103/PhysRevLett.85.4656) or "YL02" (10.1103/PhysRevLett.89.281102)

    Keyword Arguments:
        power_index -- power-spectrum for transverse modes in GS95 turbulence (default: 10/3)
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
    b_field[:, :, 0] = b_re[:, :, 0]  # real values in the xy plane
    
    # Step 2. Reweighting
    k_prp_grid = np.sqrt(k_grid[..., 0]**2 + k_grid[..., 1]**2)  # x & y
    k_prl_grid = np.abs(k_grid[..., 2])  # z
    
    power_grid = np.zeros_like(k_prl_grid)
    mask = (k_prp_grid > 0) * (k_prl_grid > 0)
    
    if model == 'YL02': # use exponential formulation from (Yan & Lazarian, 2002)
        power_grid[mask] = k_prp_grid[mask] ** (-power_index) * np.exp(-k_prl_grid[mask] * k_inj**(-1/3) * k_prp_grid[mask]**(-2/3))
    elif model == "Chandran": # use gate formulation from (Chandran, 2000)
        power_grid[mask] = k_prp_grid[mask] ** (-power_index) * np.heaviside(-k_prl_grid[mask] * k_inj**(-1/3) * k_prp_grid[mask]**(-2/3), 1.0)

    b_prime_field = b_field * np.sqrt(power_grid[..., None])
    
    # Step 3. Orthogonal projection
    n_grid = np.zeros_like(k_grid)
    k_norm_grid = np.sqrt(np.sum(k_grid**2, axis=-1))  # k-vector norms
    n_grid[mask] = k_grid[mask] / k_norm_grid[mask, None]

    n_dot_b = np.sum(n_grid * b_prime_field, axis=-1)  # scalar product

    B_k_field = b_prime_field - n_grid * n_dot_b[..., None]  # divergence-free field
    B_k_field[0, 0, 0] = np.array([0, 0, 0])  # send the zero-mode to 0
    
    # Step 4. Inverse Fourier Transform
    B_field = ifftn(B_k_field,
                    s=k_grid.shape[:-1],
                    axes=(0,1,2),
                    norm='backward').real
    
    B_norm = np.average(B_field**2,
                        axis=(0,1,2))
    
    B_field /= np.sqrt(3 * B_norm)[None, None, None, :]
        
    return B_field


def get_aniso_cube(n_side, L, k_inj, model, power_index=10/3, seed=None):
    """Get an isotropic magnetic field in the coordinate grid of size n_side x n_side x n_side
    and spatial scale 2L x 2L x 2L with a power-spectrum P(|k|) = Ak^{-power_index},
    and inertial range from k_min to k_max

    Arguments:
        n_side -- size of the cube side (normally, a power of 2: 64, 128, or 256)
        L -- [arb. length units] half-scale of the cube side

    Keyword Arguments:
        power_index -- isotropic spectrum power index p: P(k) ~ k^{-p} (default: {-11/3}, Kolmogorov turbulence)
        seed -- seed value for the random generator. If None, the fresh unpredictable state will be set. (default: {None})

    Returns:
        Normalized vector field in coordinate space with a known power-spectrum in the momentum space
        as a numpy array of (3, n_side, n_side, n_side)
    """
    k_grid = set_k_cube(n_side, L)
    b_field = set_anisotropic_spectrum(k_grid, k_inj, model, power_index, seed)
    return b_field


if __name__ == "__main__":
    print("Nor for direct use.")

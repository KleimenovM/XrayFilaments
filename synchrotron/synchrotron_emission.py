# synchrotron/synchrotron_emission.py

import numpy as np

import astropy.units as u
from astropy.constants import codata2010 as cst

from scipy.special import gamma

from config.constants import CST_HC, CST_e, CST_m_e


def first_synchrotron_function_approximation(x):
    """
    Analytical approximation of the first synchrotron function
    F(x) = x * int_x^inf K_{5/3}(t) dt,
    where K is the modified Bessel function of the second kind.
    see [Fouka, Ouichaoui, 2013] (relative error < 0.26%).
    
    Parameters:
    -----------
    x: float or array-like
        The argument of the synchrotron function.
        
    Returns:
    --------    F: float or array-like
        The value of the first synchrotron function at x.
    """
    a1 = np.array([-0.97947838884478688,
                   -0.83333239129525072,
                   0.15541796026816246])
    
    a2 = np.array([-4.69247165562628882e-2,
                   -0.70055018056462881,
                   1.03876297841949544e-2])

    # theta function approximations: delta_1 = 1 for x << 1, delta_2 = 1 for x >> 1 (0 else)

    H1 = a1[0] * x + a1[1] * x**(1/2) + a1[2] * x**(1/3)
    H2 = a2[0] * x + a2[1] * x**(1/2) + a2[2] * x**(1/3)

    delta_1 = np.exp(H1)
    delta_2 = 1 - np.exp(H2)

    # the asymptotes
    F1 = np.pi * 2**(5/3) / (np.sqrt(3) * gamma(1/3))
    F2 = np.sqrt(np.pi/2)

    F_low = F1 * x**(1/3)
    F_high = F2 * np.exp(-x) * x**(1/2)

    # return the convolution
    return delta_1 * F_low + delta_2 * F_high


def characteristic_synchrotron_energy(e, b, alpha=90*u.deg):
    """
    Calculate the characteristic synchrotron energy for an electron of energy e in a magnetic field b and pitch angle alpha.
    
    Parameters:
    -----------
    e: astropy Quantity [eV]
    b: astropy Quantity [G]
    alpha: astropy Quantity [deg]
    
    Returns:
    --------    eps_c: astropy Quantity [eV]
        The characteristic synchrotron energy.
    """
    return (3 * CST_e * b * CST_HC * np.sin(alpha) / (4 * np.pi * CST_m_e**3) * e**2).to(u.eV)


def synchrotron_peak_energy(e, b, alpha=90*u.deg):
    """
    Calculate the synchrotron peak energy for an electron of energy e in a magnetic field b and pitch angle alpha.
    The peak energy is approximately 0.29 times the characteristic energy.
    
    Parameters:
    -----------
    e: astropy Quantity [eV]
    b: astropy Quantity [G]
    alpha: astropy Quantity [deg]
    
    Returns:
    --------    eps_peak: astropy Quantity [eV]
        The synchrotron peak energy.
    """
    
    return 0.29 * characteristic_synchrotron_energy(e, b, alpha)


def single_electron_synchrotron_power(electron_energy,
                                      photon_energy,
                                      bfield,
                                      alpha=None, n_alpha=20):
    """
    Calculation of single electron synchrotron power.
    
    Can handle:
    - fixed alpha (float) → power for fixed angle
    - averaged alpha (None) → integration over pitch angle
    
    Parameters:
    -----------
    electron_energy: array-like [Ne] or array-like (1 x Ne)
        The energy of the electrons.
    photon_energy: array-like [Nph] or array-like (Nph x 1)
        The energy of the emitted photons.
    bfield: astropy Quantity [G]
        The magnetic field strength.
    alpha: float or None
        The pitch angle in radians. If None, the power is averaged over pitch angles.
    n_alpha: int
        The number of points for integration if alpha is None.  
        
    Returns:
    --------    P: array-like (Nph x Ne)
        The synchrotron power emitted by a single electron of energy E at photon energy eps.
    """   
    # fixed pitch angle
    if alpha is not None:
        E_b = electron_energy[None, :]
        eps_b = photon_energy[:, None]
        
        # critical energy
        eps_c = characteristic_synchrotron_energy(E_b, bfield, alpha=alpha)  # shape (1 x Ne)
        x = (eps_b / eps_c).to("").value  # (Nph x Ne)
        F = first_synchrotron_function_approximation(x) * np.sin(alpha)  # shape (Nph x Ne)
        
    # averaged pitch angle
    else:
        # electron and photon energy drids
        E_b = electron_energy[None, :, None]
        eps_b = photon_energy[:, None, None]
        
        # angle grid
        alpha_grid = np.linspace(1e-2, np.pi/2, n_alpha)
        alpha_b = alpha_grid[None, None, :]
        
        eps_c = characteristic_synchrotron_energy(E_b, bfield, alpha=alpha_b)  # shape (1 x Ne x Nalpha)
        x_b = (eps_b / eps_c).to("").value  # shape (Nph x Ne x Nalpha)
        
        F_raw = first_synchrotron_function_approximation(x_b) * np.sin(alpha_b)  # (Nph x Ne x Nalpha)
        
        # integrate over pitch angle with sin(alpha) weighting (from homogeneous distribution of pitch angles)
        # normalize by the integral of sin**2(alpha) from [-alpha to alpha], which is equal to 1
        F = np.trapezoid(F_raw * np.sin(alpha_b), alpha_grid, axis=2)  # shape (Nph x Ne)

    # dimensionful prefactor
    A = np.sqrt(3) * CST_e**3 * bfield * cst.c / (CST_HC * CST_m_e)
    P = A * F  # shape (Nph x Ne)
    return P.to(1/u.s)


def electron_synchrotron_emission_luminosity(electron_energy,
                                             electron_density,
                                             photon_energy,
                                             bfield,
                                             alpha=None, n_alpha=20):
    """
    Calculate the synchrotron luminosity from a distribution of electrons.
    
    Parameters:
    -----------
    electron_energy: array-like [Ne]
        The energy of the electrons.
    electron_density: array-like [Ne]
        The number density of electrons per energy bin.
    photon_energy: array-like [Nph]
        The energy of the emitted photons.
    bfield: astropy Quantity [G]
        The magnetic field strength.
    alpha: float or None
        The pitch angle in radians. If None, the power is averaged over pitch angles.
    n_alpha: int
        The number of points for integration if alpha is None.
    
    Returns:
    --------    L: array-like [Nph]
        The synchrotron luminosity at each photon energy.   
    """
    # synchrotron power (Nph x Ne)
    P = single_electron_synchrotron_power(electron_energy, photon_energy, bfield,
                                          alpha=alpha, n_alpha=n_alpha)

    # integration over electron energy to get luminosity (Nph,)
    lum = np.trapezoid(P * electron_density[None, :], electron_energy, axis=1)
    return lum.to(1/u.s)  # shape (Nph,)


if __name__ == '__main__':
    print("Not for direct use.")
    
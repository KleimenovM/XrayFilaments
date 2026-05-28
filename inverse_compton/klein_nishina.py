# inverse_compton/klein_nishina.py

import numpy as np
from astropy.constants import codata2010 as cst
import astropy.units as u

from config.constants import CST_m_e, T_CMB
from src.black_body_radiation import bbr_density


KN_CST = 3 * cst.sigma_T * cst.c / 4


def klein_nishina_profile_function(x1, g_e1):
    """
    Compute the Klein-Nishina profile.

    Args:
        x1 (array_like or float): Input parameter for the Klein-Nishina profile.
        g_e1 (array_like or float): Scaled electron energy parameter.

    Returns:
        ndarray or float: Klein-Nishina profile values for the given inputs.
    """
    mask = (x1 > 0) * (x1 < 1)  # Ensure x1 is within the valid range (0, 1)
    
    q1 = x1 / (1 + g_e1 - g_e1 * x1)
    q1 = q1.clip(1e-32, 1)  # Avoid numerical issues by clipping q1 to a reasonable range
    
    result = (2 * q1 * np.log(q1) + (1 + 2 * q1) * (1 - q1) + 1 / 2 * (g_e1 * q1) ** 2 / (1 + g_e1 * q1) * (1 - q1))
    result_masked = np.where(mask, result, 0)  # Set values outside the valid range to 0
    
    return result_masked


def klein_nishina_on_a_given_photon_density_profile(gamma, e_phot_in, e_phot_out, bg_phot_density,
                                                    if_norm: bool = False, mass=CST_m_e):
    """
    Calculate the Klein-Nishina scattering on a given photon density profile.
    Based on Blumenthal et al. (1970).
    
    Parameters:
    -----------
        gamma: Lorentz factor of the incident electron.
        e_phot_in: Incoming photon energies (array-like, in eV).
        e_phot_out: Outgoing photon energies (array-like, in eV).
        bg_phot_density: Background photon density as a function of energy (array-like, in cm^-3 eV^-1).
        if_norm: If True, normalizes the result to the total scattering rate.
        mass: Mass of the electron (default is CST_m_e).
    
    Returns:
    --------
        Scattering rate dN/dt/de1 in eV-1 s-1.
    """
    e12 = e_phot_out[:, None]
    e21 = e_phot_in[None, :]
    
    g_e21 = 4 * e21 * gamma / mass
    E12 = e12 / (gamma * mass)
    x_12 = E12 * (1 + g_e21) / g_e21  # x = E1/E1(max)
    
    F1 = klein_nishina_profile_function(x_12, g_e21)  # [DL]
    
    result = np.trapezoid(bg_phot_density[None, :] * F1, np.log(e21.to_value(u.eV)), axis=1)  # [cm-3 eV-1]

    if if_norm:
        norm = np.trapezoid(result, e_phot_out, axis=0)
        return result / norm
    
    dim_factor = KN_CST / gamma**2  # [cm 3 s-1]

    return (dim_factor * result).to(1/(u.eV * u.s))


def klein_nishina_on_CMB(gamma, e_phot_out, e_phot_in=None, if_norm: bool = False):
    """
    Calculate the Klein-Nishina scattering on the Cosmic Microwave Background (CMB).
    
    Parameters:
    -----------
        gamma: Lorentz factor of the incident electron.
        e_phot_out: Outgoing photon energies (array-like, in eV).
        e_phot_in: Incoming photon energies (array-like, in eV) (to integrate).
        if_norm: If True, normalizes result to total scattering rate.
        
    Returns:
    --------
        Scattering rate dN/dt/de1 in eV-1 s-1.
    """

    if e_phot_in is None:
        e_phot_in = np.logspace(-9, -1, 1000) * u.eV

    n_CMB = bbr_density(e_phot_in, T_CMB)
    
    return klein_nishina_on_a_given_photon_density_profile(gamma, e_phot_in, e_phot_out, n_CMB, if_norm=if_norm)


if __name__ == '__main__':
    print("Not for direct use.")

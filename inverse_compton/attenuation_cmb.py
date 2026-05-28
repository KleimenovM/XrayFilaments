import numpy as np

import astropy.units as u

from inverse_compton.cross_section import total_cross_section
from src.black_body_radiation import bbr_density
from config.constants import T_CMB


def get_extinction_at_fixed_energy(photon_energy, distance,
                                   cmb_energies, cmb_density):
    """
    Calculates the extinction (attenuation factor) for a photon at a fixed energy due
    to scattering and absorption by the CMB photons
    Returns the resulting exponential attenuation factor e^{optical_depth}

    :param photon_energy: [eV] Photon energy for which the extinction needs to be calculated.
    :param distance: [kpc], Distance over which the photon propagates.
    :param cmb_energies: [eV], Energies of the CMB photons.
    :param cmb_density: [eV], Number densities corresponding to the CMB photon energies.
    :return: [DL] Exponential attenuation factor.
    """

    tcs = total_cross_section(e0=photon_energy, e=cmb_energies, z=0).to(u.cm ** 2)
    tau_cmb = np.trapezoid(tcs * cmb_density * distance, cmb_energies)

    return np.exp(tau_cmb)


def get_extinction_by_CMB(photon_energy, distance,
                          cmb_energies=None):
    if cmb_energies is None:
        cmb_energies = np.logspace(-9, -1, 1000) * u.eV
        
    cmb_density = bbr_density(cmb_energies, T_CMB)
    
    abs_exp = get_extinction_at_fixed_energy(photon_energy, distance,
                                             cmb_energies, cmb_density)
    return abs_exp


if __name__ == "__main__":
    print("Not for direct use!")

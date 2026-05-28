import numpy as np
from astropy.constants import codata2010 as cst
import astropy.units as u


CONST_BBR = 8 * np.pi / (cst.h * cst.c)**3


def bbr_density(energy, temperature):
    """
    Get the !isotropic! BBR density given its temperature
    :param energy: [eV], numpy array / float, energy range
    :param temperature: [K], temperature
    :return: dN/dE
    """

    theta = (energy / (temperature * cst.k_B)).to_value('')
    theta = np.clip(theta, 1e-12, 700)

    # stable computation
    denom = np.expm1(theta)
    spec = 1.0 / denom

    spec = np.where(spec < 1e-20, 0.0, spec)

    return (CONST_BBR * spec * energy**2).to(1/(u.cm**3 * u.eV))


if __name__ == "__main__":
    print("Not for direct use.")

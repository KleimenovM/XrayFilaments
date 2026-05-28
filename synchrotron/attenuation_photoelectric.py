import numpy as np
import astropy.units as u
from astropy.constants import codata2010 as cst


def photoelectric_optical_depth(photon_energy, hI_column_density):
    # see eq. (9.3) in p.230 in [Longair, 2011]
    return np.exp(2e-26 * photon_energy.to_value(u.keV)**(-8/3) * hI_column_density.to_value(u.m**(-2)))

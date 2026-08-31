import numpy as np
import astropy.units as u
from astropy.constants import codata2010 as cst
from config.constants import CST_m_e, T_CMB, CST_r0

from inverse_compton.klein_nishina import klein_nishina_on_CMB
from src.black_body_radiation import bbr_density

from config.settings import IC_DIR

from scipy.interpolate import interp1d


cst_sum = 5/6 + 0.5772 + 0.5700

def extreme_klein_nishina_timescale(E):
    prefactor = (6 * cst.hbar**3 * E / (np.pi * CST_r0**2 * (CST_m_e/cst.c * cst.k_B * T_CMB)**2)).to(u.kyr)
    log_scaling = np.log((4 * E * cst.k_B * T_CMB)/(CST_m_e)**2) - cst_sum
    return prefactor / log_scaling


def ic_cooling_time(E):
    data = np.load(IC_DIR / "ic_time.npz")
    energy = data["e"]  # [eV]
    times = data["t"]  # [yr]

    data_interpolator = interp1d(np.log10(energy), np.log10(times), bounds_error=False, fill_value='extrapolate')
    return 10**data_interpolator(np.log10(E.to_value(u.eV))) * u.yr


if __name__ == "__main__":
    print("Not for direct use")

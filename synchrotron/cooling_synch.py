import numpy as np
import astropy.units as u

import astropy.constants.codata2010 as cst
from config.constants import CST_m_e, CST_e


def synchrotron_timescale(energy, bfield, alpha):
    sin_alpha = np.sin(alpha)
    gamma = (energy / CST_m_e).to('')
    P_syn = 2 * CST_e ** 4 / (3 * CST_m_e ** 2 / cst.c) * bfield ** 2 * gamma ** 2 * sin_alpha**2
    return (energy / P_syn).to(u.yr)


def synchrotron_timescale_avg(energy, bfield):
    sin_avg = 2 / 3
    gamma = (energy / CST_m_e).to('')
    P_syn = 2 * CST_e ** 4 / (3 * CST_m_e ** 2 / cst.c) * bfield ** 2 * gamma ** 2 * sin_avg
    return (energy / P_syn).to(u.yr)

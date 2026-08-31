import numpy as np

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.differentiate import derivative
from scipy.interpolate import interp1d

from config.settings import IC_DIR
from config.units import uGauss
from config.constants import CST_m_e, CST_e


max_value = 80


class Cooling:
    """
    Represents a system or process related to cooling.

    This class provides mechanisms to calculate power and its derivative
    for a cooling process over a given time and energy input. It serves as 
    a base or generic model for systems related to cooling calculations.

    """
    def __init__(self):
        self.magnetic_field = 0  # [uGauss]
        self.min_energy = 1e9    # [eV]
        self.max_energy = 1e22   # [eV]
        pass

    def clip_energy(self, energy_eV):
        """
        Clips the given energy value to ensure it stays within the predefined minimum
        and maximum energy bounds.

        :param energy_value: [eV], The energy value to be clipped.
        :return: [eV], The energy value
        """
        return np.clip(energy_eV, a_min=self.min_energy, a_max=self.max_energy)

    def power(self, time, energy):
        return

    def power_derivative(self, time, energy):
        return


class SynchrotronCooling(Cooling):
    """
    This class initializes and calculates the total synchrotron radiation power in
    a given magnetic field.

    :param magnetic_field: [uGauss], Strength of the magnetic field
    """
    def __init__(self, magnetic_field, angle, if_avg = False):
        super().__init__()
        sin_alpha2 = 2 / 3 if if_avg else np.sin(angle)**2
        sharp_units = u.eV**(-1) * u.yr**(-1)
        self.sharp = (2 * CST_e**4 / (3 * CST_m_e**4 / cst.c) * magnetic_field**2 * sin_alpha2).to_value(sharp_units) # [1/(eV yr)]

    def power(self, time_yr, energy_eV):
        """
        Synchrotron radiation power.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: Synchrotron radiation power [eV/yr]
        """
        energy_eV = self.clip_energy(energy_eV)
        return self.sharp * energy_eV**2  # [eV/yr]

    def power_derivative(self, time, energy):
        """
        Derivative of synchrotron radiation power function with respect to energy.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: Synchrotron radiation power derivative [1/yr]
        """
        energy_eV = self.clip_energy(energy)
        return -2 * self.sharp * energy_eV


class InverseComptonCooling(Cooling):
    """
    This class initializes and calculates the total inverse compton radiation power
    """
    def __init__(self):
        super().__init__()
        self.min_energy = 1e9   # [eV]
        self.max_energy = 1e20  # [eV]
        self.ic_time_f = self.__set_ic_time_approximation()

    @staticmethod
    def __set_ic_time_approximation():
        """
        Load and process inverse Compton (IC) time data and create a spline representation for IC time 
        approximation. This method reads precomputed data from a file, processes the energy and IC 
        time arrays, and generates a spline for efficient interpolation.

        :return: A cubic spline for IC time as a function of log-scaled energy
        """
        data = np.load(IC_DIR / "ic_time.npz")
        energies = data["e"]  # [eV]
        times = data["t"]     # [yr]
        ic_time_f = interp1d(np.log10(energies), np.log10(times), bounds_error=False, fill_value="extrapolate")
        return ic_time_f

    def power(self, time, energy):
        """
        Inverse Compton radiation power.

        :param time: [yr] Time value (no dependence).
        :param energy: [eV] Energy value.
        :return: inverse comptopn radiation power [eV/yr]
        """
        energy_eV = self.clip_energy(energy)
        return energy_eV / (10**self.ic_time_f(np.log10(energy_eV)))  # [eV/yr]
    
    def __loglog_power(self, lg_energy_eV):
        """
        Computes the logarithmic value of a power function using base-10 logarithms.
        
        :param lg_energy_eV: The logarithmic energy in eV.
        :return: The base-10 logarithm of the computed power.
        """
        return np.log10(self.power(0.0, 10**lg_energy_eV))  # [DL]

    def power_derivative(self, time_yr, energy_eV):
        """
        Derivative of Inverse Compton radiation power function with respect to energy.
        This function calculates double-log derivative of the power function.

        :param time_yr: [yr] Time value (no dependece).
        :param energy_eV: [eV] Energy value.
        :return: Inverse Compton radiation power derivative [1/yr]
        """
        energy_eV = self.clip_energy(energy_eV)
        lg_energy_value = np.log10(energy_eV)
        power = self.power(time_yr, energy_eV)
        return -power / energy_eV * derivative(self.__loglog_power, lg_energy_value).df  # [1/yr]


class JointCooling(Cooling):
    """
    Both synchrotron and inverse Compton cooling taken into account
    """
    def __init__(self, magnetic_field=1*uGauss, alpha=90*u.deg, if_avg=False):
        super().__init__()
        self.magnetic_field = magnetic_field
        self.synch = SynchrotronCooling(self.magnetic_field, alpha, if_avg=if_avg)
        self.ic = InverseComptonCooling()

    def power(self, time_yr, energy_eV):
        return self.synch.power(time_yr, energy_eV) + self.ic.power(time_yr, energy_eV)
    
    def power_for_log(self, time_yr, x):
        energy_eV = np.exp(np.clip(x, a_min=0, a_max=max_value))
        return self.power(time_yr, energy_eV) / energy_eV
    
    def power_derivative(self, time_yr, energy_eV):
        return self.synch.power_derivative(time_yr, energy_eV) + self.ic.power_derivative(time_yr, energy_eV)


if __name__ == '__main__':
    print("Not for direct use")

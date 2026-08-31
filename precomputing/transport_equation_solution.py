import numpy as np

import astropy.units as u
from scipy.integrate import solve_ivp, cumulative_trapezoid

from cooling.cooling import Cooling
from synchrotron.cooling_synch import synchrotron_timescale

from config.units import uGauss


def energy_backpropagation(cooling: Cooling, magnetic_field, alpha=90*u.deg,
                           N_time: int = 2000, N_energy: int = 400,
                           if_silent: bool = False):
    """
    Calculates energy grid backpropagation over time with the provided cooling model.

    The function implements energy loss calculations using a provided
    cooling model and integrates the temporal dynamics of particle
    energies using an ODE solver. It computes results over logarithmic
    time and energy grids defined by the inputs and outputs the grid
    values along with the computed solution.

    :param magnetic_field: [uGauss], Magnetic field value
    :param if_silent: (bool), verbosity level
    :param cooling: An instance of a Cooling class, which defines the cooling mechanisms and power loss function.
    :param N_time: The number of grid points over the time domain. Default is 10,000.
    :param N_energy: The number of grid points over the energy domain. Default is 1,000.
    :return: A tuple containing the time array and the computed energy solutions across time.
    """

    # the grid is defined by synchrotron cooling timescales as the fastest ones!
    t_min = np.minimum(synchrotron_timescale(1e18 * u.eV, magnetic_field + 3 * uGauss, alpha).to_value(u.yr), 1)  # [yr] cooling timescale for 1000 PeV is the smallest value
    if not if_silent:
        print(f"min time {t_min:.0g} yr")
    times = np.zeros(N_time+1)  # [yr], time grid
    times[1:] = t_min * np.logspace(0, 7, N_time)  # [yr], log time grid

    # energy grid
    energies = np.logspace(9, 19, N_energy)  # [eV]
    if not if_silent:
        print(f"min energy {energies[0]:.0g} eV, max energy {energies[-1]:.0g} eV")

    # losses
    sol = solve_ivp(cooling.power_for_log,
                    t_span=(0, times[-1]),  # [yr]
                    y0=np.log(energies),    # [eV]
                    first_step=t_min,       # [yr]
                    t_eval=times,           # [yr]
                    method='DOP853',
                    dense_output=True)

    if not if_silent:
        print(f"SOLUTION EXISTS: {sol}")

    return times, np.exp(np.clip(sol.y, a_min=0, a_max=80))  # [yr], [eV]

def get_the_modulation_coefficient(cooling: Cooling, time_sol_yr, energy_sol_eV):
    """
    Calculates the modulation coefficient from the method of characteristics.

    :param cooling: An instance of a Cooling class, which defines the cooling mechanisms and power loss function.
    :param time_sol: The computed time solution from the energy loss calculation.
    :param energy_sol: The computed energy solution from the energy loss calculation.
    """
    loss_derivative = cooling.power_derivative(time_sol_yr, energy_sol_eV)
    loss_derivative_integral = np.clip(cumulative_trapezoid(loss_derivative, time_sol_yr, initial=0), a_min=-10, a_max=20)
    return np.exp(-loss_derivative_integral)


if __name__ == '__main__':
    print("Not for direct use")


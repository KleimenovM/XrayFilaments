import time

import numpy as np

import astropy.units as u

from precomputing.transport_equation_solution import energy_backpropagation, get_the_modulation_coefficient
from config.settings import PREC_DIR
from config.units import uGauss
from cooling.cooling import JointCooling


def joint_cooling_precomputing(magnetic_field, alpha=90*u.deg, if_avg=False):
    """
    Performs energy backpropagation and modulation coefficient calculation for a given magnetic field.
    :param bvalue: Magnetic field value in Gauss
    :return: times, energies, modulation_coefficient
    """
    joint_cooling = JointCooling(magnetic_field, alpha=alpha, if_avg=if_avg)  # joing cooling set
    t0 = time.time()
    print("energy backpropagation calculation started")
    times_yr, energies_eV = energy_backpropagation(cooling=joint_cooling, magnetic_field=magnetic_field)
    t1 = time.time()
    print(f"energy backpropagation calculation finished in {t1 - t0:.0f} s")

    # modulation coefficient calculation
    t2 = time.time()
    print("modulation coefficient calculation started")
    modulation_coefficient = get_the_modulation_coefficient(cooling=joint_cooling, time_sol_yr=times_yr, energy_sol_eV=energies_eV)
    t3 = time.time()
    print(f"modulation coefficient calculation finished in {t3 - t2:.0f} s")
    return times_yr, energies_eV, modulation_coefficient


def save_cooling_precomputed(bvalue, times, energies, modulation_coefficient):
    # times in yr
    # energies in eV
    np.savez(PREC_DIR/f"joint_cooling_{bvalue.to_value(uGauss):.2f}",
             t=times, e=energies, m=modulation_coefficient)
    return


if __name__ == '__main__':
    bfs = np.loadtxt(f"{PREC_DIR}/fields.txt") * uGauss
    for bf in bfs:
        print(f"MF = {bf.to_value(uGauss):.2f} uG")
        t, e, mc = joint_cooling_precomputing(bf)
        save_cooling_precomputed(bf, t, e, mc)



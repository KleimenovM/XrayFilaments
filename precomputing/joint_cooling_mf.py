import numpy as np

import astropy.units as u
import astropy.constants.codata2010 as cst

from scipy.interpolate import RegularGridInterpolator

from config.constants import T_CMB
from config.settings import PREC_DIR
from config.units import uGauss
from cooling.cooling import SynchrotronCooling


cmb_related_mf = np.sqrt((4 * cst.sigma_sb / cst.c * T_CMB**4) * (8*np.pi)).to_value(uGauss)
cmb_effective_mf = np.sqrt(2/3) * cmb_related_mf
print(f"Effective field related to CMB: {cmb_related_mf:.2f} uG")
print(f"But effective MF is decreased by a factor of sqrt(2/3) => {cmb_effective_mf:.2f} uG")


def fix_broken(matrix):
    one_px_delta = matrix[:, 1:] - matrix[:, :-1]  # must increase in backpropagation
    broken_px = one_px_delta < 0
    
    if_broken = np.sum(broken_px)   # if there exist negative values
    
    print(if_broken)
        
    if if_broken < 1:
        return matrix
    
    matrix[:, 1:][broken_px] = matrix[:, :-1][broken_px]  # change to the previous value
    
    matrix = fix_broken(matrix)
    return matrix


def joint_cooling_interpolator():
    fields = np.loadtxt(f"{PREC_DIR}/fields.txt")  # [uGauss]
    # load precomputed data
    energies = []
    times = []
    modulation_coefficient = []

    for field in fields:
        data = np.load(f"{PREC_DIR}/joint_cooling_{field:.2f}.npz")
        energies.append(data["e"])
        times.append(data["t"])
        modulation_coefficient.append(data["m"])
        
    energies = np.array(energies)
    energies0 = energies[0, :, 0]  # starting energies are the same for all the fields, so we can take the first one
    
    modulation_coefficient = np.array(modulation_coefficient)

    # print(energies0)

    times = np.array(times)
    tmin = times[times > 0].min()        # [yr]
    tmax = np.maximum(times.max(), 1e7)  # [yr]

    times0 = np.zeros(251)  # [yr], time grid
    times0[1:] = tmin * np.logspace(0, np.log10(tmax/tmin), 250)  # [yr], log time grid
    modulation_coefficient = np.array(modulation_coefficient)
    
    new_table_e = np.zeros((len(fields), len(energies0), len(times0)))
    new_table_m = np.zeros((len(fields), len(energies0), len(times0)))

    for i, field in enumerate(fields):
        sc = SynchrotronCooling(field * uGauss, 90*u.deg)
        sc2 = SynchrotronCooling(np.sqrt(field**2 + cmb_effective_mf**2) * uGauss, 90*u.deg) # Synchrotron + Thomson
            
        # 1A. energy interpolation
        f_interp_e = RegularGridInterpolator(
            (np.log10(energies0), np.log10(times[i])),
            np.log10(energies[i]),
            bounds_error=False,
            fill_value=-32)

        lg_new_table_e_i = f_interp_e((np.log10(energies0)[:, None], np.log10(times0[None, :])))

        # 1B. modulation interpolation
        f_interp_m = RegularGridInterpolator(
            (np.log10(energies0), np.log10(times[i])),
            np.log10(modulation_coefficient[i]),
            bounds_error=False,
            fill_value=-32)

        lg_new_table_m_i = f_interp_m((np.log10(energies0)[:, None], np.log10(times0[None, :])))

        # out of bounds values
        out_values = (lg_new_table_e_i == -32)
        
        # 2. nan values
        # 2A
        nan_values_e = np.isnan(lg_new_table_e_i)
        lg_new_table_e_i[nan_values_e] = np.log10((energies0[:, None] / (1 - sc.sharp * energies0[:, None] * times0[None, :]))[nan_values_e])
        new_nan = np.isnan(lg_new_table_e_i)

        # 2B
        nan_values_m = np.isnan(lg_new_table_m_i)
        lg_new_table_m_i[nan_values_m] = np.log10((1 / (1 - sc.sharp * energies0[:, None] * times0[None, :])**2)[nan_values_m])
        lg_new_table_m_i[new_nan] = np.nan

        if_out = np.sum(out_values)
        if if_out > 0:
            print("OUT VALUES EXIST!")

        first_row_nan = np.argmax(np.abs(out_values[0, :-1] ^ out_values[0, 1:])) + 1
        first_line_invalid = np.argmin(np.abs(lg_new_table_e_i[:, first_row_nan-1] - 20))
                                
        if if_out: # if at longer timescales there is no data...
            t_start = times0[first_row_nan-1]
            t_valid = times0[first_row_nan:]

            e_start = 10**lg_new_table_e_i[:first_line_invalid, first_row_nan-1]

            ans_e = e_start[:, None] / (1 - sc2.sharp * e_start[:, None] * (t_valid - t_start)[None, :])
            ans_m = 1 / (1 - sc2.sharp * e_start[:, None] * (t_valid - t_start)[None, :])**2
            
            ans_good = ans_e > 0

            lg_new_table_e_i[:first_line_invalid, first_row_nan:][ans_good] = np.log10(ans_e[ans_good])
            lg_new_table_m_i[:first_line_invalid, first_row_nan:][ans_good] = np.log10(10**lg_new_table_m_i[:first_line_invalid, first_row_nan-1][:, None] * ans_m)[ans_good]
            
            nan_final = lg_new_table_e_i == -32
            lg_new_table_e_i[nan_final] = np.nan
            lg_new_table_m_i[nan_final] = np.nan
            
        # Finally, check if the energy increases to exclude possible broken bins
        
            
        lg_new_table_e_i = np.nan_to_num(lg_new_table_e_i, nan=32)
        lg_new_table_m_i = np.nan_to_num(lg_new_table_m_i, nan=5)
        
        lg_new_table_e_i = np.clip(lg_new_table_e_i, a_min=0, a_max=25)
        lg_new_table_m_i = np.clip(lg_new_table_m_i, a_min=-10, a_max=4)
        
        lg_new_table_e_i = fix_broken(lg_new_table_e_i)
        lg_new_table_m_i = fix_broken(lg_new_table_m_i)
        
        new_table_e[i, :, :] = 10**lg_new_table_e_i
        new_table_m[i, :, :] = 10**lg_new_table_m_i
    
    np.savez(PREC_DIR / "table_e", b=fields, e0=energies0, t0=times0, table=new_table_e)
    np.savez(PREC_DIR / "table_m", b=fields, e0=energies0, t0=times0, table=new_table_m)
        
    return


if __name__ == '__main__':
    joint_cooling_interpolator()
    
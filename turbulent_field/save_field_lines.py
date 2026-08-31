import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

from field_lines import PeriodicInterpolator, integrate_lines, unwrap_lines, plot_field_lines
from generate_goldreich_sridhar import get_aniso_cube
from cubes import set_r_cube, set_k_cube


ROOT_DIR = Path(__file__).resolve().parent.parent


def get_the_lines(L, n_side, eta, M):
    dx = 2*L/n_side # [pc], cell size
    k_min = 2*np.pi / (2*L)  # [pc-1], injection scale
    print(f"Each cell is a cube with size dx = {dx:.2f} pc")
    print(f"Injection scale (2pi/k_min) is {2*np.pi/k_min:.2f} pc")

    r_cube = set_r_cube(n_side, L)
    b_cube = get_aniso_cube(n_side, L, k_min, model='YL02')
    
    # add the regular component
    b0 = 1 / np.sqrt(1 + eta**2)
    
    b_cube = eta / np.sqrt(1 + eta**2) * b_cube
    b_cube[..., 2] += b0

    periodic_interpolator = PeriodicInterpolator(r_cube, b_cube)
    
    # INTEGRATE THE LINES
    ds = dx / 5
    n_steps = int(2 * L / ds)  # n_steps

    # calculate field lines
    lines = integrate_lines(periodic_interpolator, M, n_steps, ds)

    # calculate magnetic fields along the line
    fields = np.zeros([2*n_steps + 1, M, 3])
    for i in range(M):
        fields[:, i, :] = periodic_interpolator(lines[:, i])

    # now shift the lines to 0
    lines_from_zero = unwrap_lines(lines - lines[n_steps], 2 * L)
    
    # finally, the s array
    s = ds * np.arange(-n_steps, n_steps + 1)
    
    return s, lines_from_zero, fields


def save_field_lines(N, eta):
    
    L = 300  # [pc]
    nside = 256
    M = 1000
    
    location = ROOT_DIR / "data" / "field_lines" / "gs" / f"eta{eta:.1f}"
    os.makedirs(location, exist_ok=True)
    
    print(location)
    
    for i in range(N):
        print(f"Cube generation & MF line computation #{i+1}...")
        
        s, lines_from_zero, fields = get_the_lines(L, nside, eta, M)
        
        filename = location / f"lines_eta{eta:.1f}_s{i+1}"
        print(f"Saving to {filename}...")
        np.savez_compressed(
            filename,
            s=s,
            lines=lines_from_zero,
            fields=fields
            )
        print("Saved!")
    
    print("Job done!")
    return



if __name__ == "__main__":
    for e in [0.1, 0.2, 0.5, 1.0, 2.0]:
        save_field_lines(N=20, eta=e)
    
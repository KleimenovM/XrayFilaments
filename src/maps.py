import numpy as np
import matplotlib.pyplot as plt

# from source_extended import ExtendedSource, galactic_center
from src.tools import rad_to_hours, hours_to_rad, galactic2equatorial


def coordinates_conversion_Hammer_Aitoff(ra, dec):
    """
    Convert 2nd equatorial coordinates to Hammer-Aitoff projection of the whole sky
    :param ra: (float or np.ndarray) right ascension, rad
    :param dec: (float or np.ndarray) declination, rad
    :return: (float or np.ndarray) x, (float or np.ndarray) y
    """
    # See https://en.wikipedia.org/wiki/Hammer_projection
    ra = np.deg2rad(180 - ra)
    dec = np.deg2rad(dec)
    denominator = np.sqrt(1 + np.cos(dec) * np.cos(ra / 2))
    x = 2 ** 3 / 2 * np.cos(dec) * np.sin(ra / 2) / denominator
    y = 2 ** 0.5 * np.sin(dec) / denominator
    return x, y


def coordinates_conversion_mercator(ra, dec):
    """
    Convert 2nd equatorial coordinates to Mercator projection of the whole sky
    :param ra: (float or np.ndarray) right ascension, rad
    :param dec: (float or np.ndarray) declination, rad
    :return: (float or np.ndarray) x, (float or np.ndarray) y
    """
    # See https://en.wikipedia.org/wiki/Mercator_projection
    return -ra * np.ones_like(dec), np.log(np.tan(np.deg2rad(dec/2) + np.pi/4.01)) * np.ones_like(ra)


def coordinates_conversion_equirectancular(ra, dec):
    """
    Convert 2nd equatorial coordinates to equirectangular projection of the whole sky
    :param ra: (float or np.ndarray) right ascension, rad
    :param dec: (float or np.ndarray) declination, rad
    :return: (float or np.ndarray) x, (float or np.ndarray) y
    """
    # See https://en.wikipedia.org/wiki/Equirectangular_projection
    return -ra * np.ones_like(dec), dec * np.ones_like(ra)


def coordinates_conversion_equirectancular_m180(ra, dec):
    """
    Convert 2nd equatorial coordinates to equirectangular projection of the whole sky
    :param ra: (float or np.ndarray) right ascension, rad
    :param dec: (float or np.ndarray) declination, rad
    :return: (float or np.ndarray) x, (float or np.ndarray) y
    """
    # See https://en.wikipedia.org/wiki/Equirectangular_projection
    return -ra * np.ones_like(dec) - 360 * (ra <= 180), dec * np.ones_like(ra)


def plot_grid(delta_step=30, phi_step=60,
              linewidth=1,
              coordinates_conversion=coordinates_conversion_Hammer_Aitoff,
              lambda_brd=None):
    """
    Draw the coordinate grid on the canvas
    :param delta_step: (float) distance between declination values, deg
    :param phi_step: (float) distance between right ascension values, hours
    :param linewidth: (float) width of the line
    :param coordinates_conversion: (function) coordinates conversion function
    :param lambda_brd: (list) map edges
    :return:
    """
    phi_brd = [-90, 90]
    if lambda_brd is None:
        lambda_brd = [0, 360]

    # draw parallels
    d = phi_brd[0]
    ph = np.linspace(lambda_brd[0], lambda_brd[1], 400)
    while d <= phi_brd[1]:
        x, y = coordinates_conversion(ph, d)
        plt.plot(x, y, color='gray', alpha=.5, linewidth=linewidth)

        # text
        x_t, y_t = coordinates_conversion(0, d)
        if d < 0:
            y_t -= .14
            x_t -= .2
        if d == 0:
            y_t -= .07
            x_t -= .1
        # plt.text(x_t - 0.2, y_t + .02, str(d) + r'$^\circ$')

        d += delta_step

    # draw meridians
    d = lambda_brd[0]
    phi = np.linspace(phi_brd[0], phi_brd[1], 400)
    while d <= lambda_brd[1]:
        x, y = coordinates_conversion(d, phi)
        plt.plot(x, y, color='gray', alpha=.5, linewidth=linewidth)

        # text
        x_t, y_t = coordinates_conversion(d, 0)
        # plt.text(x_t + .03, y_t - .1, f"{d:.0f}"r'$\!{}^\circ$')

        d += phi_step

    x1, y1 = coordinates_conversion(lambda_brd[0], phi)
    x2, y2 = coordinates_conversion(lambda_brd[1], phi)
    plt.fill_betweenx(y2, x1, x2, color='#ddf', alpha=1)
    return


def plot_galactic_plane(legend: bool = False,
                        coordinates_conversion=coordinates_conversion_Hammer_Aitoff):
    """
    Draw the galactic plane line in the Hammer-Aitoff projection
    :param legend: (bool) if true, appears in the final legend
    :param coordinates_conversion: (function) coordinates conversion function
    :return:
    """
    # Galactic plane
    m = 1000
    points_ll, points_b = np.zeros([m, 1]), np.zeros([m, 1])
    points_ll[:, 0] = np.linspace(0, 2 * np.pi, m)

    d, ra = galactic2equatorial(points_ll, points_b)
    x, y = coordinates_conversion(np.rad2deg(ra), np.rad2deg(d))

    x, y = x.ravel(), y.ravel()

    # Compute the difference between successive t2 values
    diffs = np.append(np.diff(x), 0)

    # Find the differences that are greater than pi
    discont_indices = np.abs(diffs) > np.pi

    # Set those t2 values to NaN
    x[discont_indices] = np.nan

    if legend:
        plt.plot(x, y, color='black', alpha=.8, label='Galactic plane')
    else:
        plt.plot(x, y, color='black', alpha=.8)

    return


if __name__ == '__main__':
    print("Not for direct use")

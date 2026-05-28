# Neutrino through Earth propagation
# Tools description

import numpy as np
from scipy.interpolate import interp1d

Ag = np.array([[-0.0548755601367195, -0.8734370902532698, -0.4838350155472244],
               [+0.4941094280132430, -0.4448296298016944, +0.7469822445004389],
               [-0.8676661489582886, -0.1980763737056720, +0.4559837761713720]])


def deg_to_rad(deg: list):
    """
    Translates [degrees, minutes, seconds] to radians
    :param deg: (list) [deg, min, sec]
    :return: (float) angle, rad
    """
    result = .0
    for i in range(len(deg)):
        result += deg[i] / (60 ** i) / 180 * np.pi
    return result


def hours_to_rad(h):
    """
    Translate hours to radians
    :param h: (float or np.ndarray) hours (0.0 < h < 24.0)
    :return: (float or np.ndarray) radians
    """
    return h / 12 * np.pi


def rad_to_hours(r):
    """
    Translate radians to hours
    :param r: (float or np.ndarray) radians (0.0 < r < 2 * np.pi)
    :return: (float or np.ndarray) hours
    """
    return r / np.pi * 12


def rot_matrix(rotation_angle: float):
    """
    Set a rotation matrix around the X-axis on the angle (90° - rotation_angle)
    :param rotation_angle: (float) rotation angle, rad
    :return: (np.ndarray) 3x3 rotation matrix
    """
    cp, sp = np.cos(rotation_angle), np.sin(rotation_angle)
    return np.array([[1, 0, 0],
                     [0, sp, -cp],
                     [0, cp, sp]])


def sph_coord(r, theta, phi) -> np.ndarray:
    """
    Set a vector on a sphere by its coordinates
    :param r: (float) sphere radius
    :param theta: (float or np.ndarray) distance from the pole to the point
    :param phi: (float or np.ndarray) distance from the 0° meridian to the point
    :return: (np.ndarray) 3D-vector(s)
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.ones(phi.shape)
    return np.array([x, y, z])


def simple_gal2eq(ll, b):
    """
    Translate single point coordinates (longitude & latitude) to the 2nd equatorial system (declination, right ascension)
    :param ll: (float) galactic longitude, rad
    :param b: (float) galactic latitude, rad
    :return: (float) declination, (float) right ascension
    """
    r_gal = sph_coord(1, np.array([b]), np.array([ll]))
    r_eq = np.dot(Ag.T, r_gal)
    delta = np.arcsin(r_eq[2])
    c_delta = np.cos(delta)
    c_alpha = r_eq[0] / c_delta
    s_alpha = r_eq[1] / c_delta
    # if sin(alpha) > 0, alpha = arccos(c_alpha)
    # else, alpha = 2 pi - arccos(c_alpha)
    k = (s_alpha > 0)  # == 1 if sin(a) > 0, == 0 else
    alpha = (2 * k - 1) * np.arccos(c_alpha) + (1 - k) * 2 * np.pi
    return delta, alpha


def galactic2equatorial(ll, b):
    """
    Translate the coordinates (longitude & latitude) to the 2nd equatorial system (declination, right ascension)
    :param ll: (np.ndarray) galactic longitude, rad
    :param b: (np.ndarray) galactic latitude, rad
    :return: (np.ndarray) declination, (np.ndarray) right ascension
    """
    r_gal = sph_coord(1, b, ll)
    r_eq = np.einsum('ij,ikl', Ag, r_gal)
    delta = np.arcsin(r_eq[2])
    c_delta = np.cos(delta)
    c_alpha = r_eq[0] / c_delta
    s_alpha = r_eq[1] / c_delta
    # if sin(alpha) > 0, alpha = arccos(c_alpha)
    # else, alpha = 2 pi - arccos(c_alpha)
    k = (s_alpha > 0)  # == 1 if sin(a) > 0, == 0 else
    alpha = (2 * k - 1) * np.arccos(c_alpha) + (1 - k) * 2 * np.pi
    return delta, alpha


def vec_product(v1, v2):
    """
    Calculate the vector product of the two 3D vectors
    :param v1: (np.ndarray) vector one
    :param v2: (np.ndarray) vector two
    :return: (np.ndarray) vector one X vector two
    """
    return np.array([v1[1] * v2[2] - v1[2] * v2[1],
                     v2[0] * v1[2] - v1[0] * v2[2],
                     v1[0] * v2[1] - v1[1] * v2[0]])


def smart_division(a: np.ndarray, b: np.ndarray):
    """
    Divide a on b. If a == b == 0, get 0
    :param a: (np.ndarray) numerator
    :param b: (np.ndarray) denominator
    :return: a / b (if a == b == 0, return 0.0)
    """
    good_indices = (b != 0.)
    result = np.zeros(b.size)
    result[good_indices] = a[good_indices] / b[good_indices]
    return result


def extrapolating_spline(x, x0, y0, if_delta=False):
    if if_delta:
        spline = interp1d(np.log10(x0), y0, kind='linear', fill_value='extrapolate')
        return spline(np.log10(x))

    spline = interp1d(np.log10(x0), np.log(y0), kind='linear', fill_value='extrapolate')
    y_p = spline(np.log10(x))

    return np.exp(y_p)


if __name__ == '__main__':
    print("Not for direct use")
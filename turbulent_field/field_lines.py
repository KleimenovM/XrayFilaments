import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class PeriodicInterpolator:
    def __init__(self, r_cube, B_cube):
        self.origin = r_cube[:, 0, 0, 0]
        self.dx = r_cube[0, 1, 0, 0] - r_cube[0, 0, 0, 0]

        self.N = r_cube.shape[1]
        self.L = self.N * self.dx

        grid = np.arange(self.N) * self.dx + self.origin[0]

        # interpolated cube
        self.interp = [
            RegularGridInterpolator(
                (grid, grid, grid),
                B_cube[i],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            for i in range(3)
        ]

    def __call__(self, x):
        """
        x : (M, 3)
        returns : (M, 3)
        """

        x = self.origin + np.mod(x - self.origin, self.L)

        B = np.empty_like(x)

        for i in range(3):
            B[:, i] = self.interp[i](x)

        return B
    
    
def rk4_step(x, ds, interp):
    """
    Vectorized realization of the classic Runge-Kutta (RK4) step
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
    
    :param x: (nvec, shape) current integration step
    :param ds: (float)
    :param interp: (callable) interpolated cube
    :return: (nvec, shape) next integration step
    """
    
    def norm(t):
        return np.linalg.norm(t, axis=1, keepdims=True)
    
    k1 = interp(x)
    k1 /= norm(k1)

    k2 = interp(x + 0.5 * ds * k1)
    k2 /= norm(k2)

    k3 = interp(x + 0.5 * ds * k2)
    k3 /= norm(k3)

    k4 = interp(x + ds * k3)
    k4 /= norm(k4)

    return x + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lines(interp, M, n_steps, ds):
    """
    Integrate the lines from -n_steps * ds to +n_steps * ds
    """
    rng = np.random.default_rng()

    x = rng.uniform(
        interp.origin,
        interp.origin + interp.L,
        size=(M, 3),
    )

    lines = np.empty((2 * n_steps + 1, M, 3))

    lines[n_steps] = x

    for i in range(n_steps, 2 * n_steps):
        x = rk4_step(x, ds, interp)
        lines[i + 1] = x
    
    x = lines[n_steps]
    for i in range(n_steps, 0, -1):
        x = rk4_step(x, -ds, interp)
        lines[i - 1] = x

    return lines


def unwrap_lines(lines, box_size):
    unwrapped = lines.copy()

    for i in range(1, len(lines)):
        delta = unwrapped[i] - unwrapped[i-1]

        delta[delta >  box_size/2] -= box_size
        delta[delta < -box_size/2] += box_size

        unwrapped[i] = unwrapped[i-1] + delta

    return unwrapped


def plot_field_lines(lines, n_lines=20, box_size=None):
    """
    Plot magnetic field lines.

    Parameters
    ----------
    lines : ndarray
        Shape (n_steps, M, 3)
    n_lines : int
        Number of lines to plot
    box_size : float, optional
        Size of periodic box
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    M = lines.shape[1]

    indices = np.random.choice(M, size=min(n_lines, M), replace=False)
    
    n_center = lines.shape[0] // 2

    for i in indices:
        x = lines[:, i, 0]
        y = lines[:, i, 1]
        z = lines[:, i, 2]

        ax.plot(x, y, z, linewidth=1)
        
        # Starting point
        ax.scatter(
            x[n_center],
            y[n_center],
            z[n_center],
            s=20
        )

    if box_size is not None:
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)
        ax.set_zlim(-box_size, box_size)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect([1, 1, 1])
    
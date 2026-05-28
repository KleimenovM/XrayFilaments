import numpy as np
import astropy.units as u


def butterfly_data(energy_range,
                   normalization, normalization_error,
                   powerlaw_index, powerlaw_index_error,
                   reference_energy=1 * u.keV):
    """
    Construct the butterfly experimental result
    :param energy_range: [u.keV or other energy], (np.ndarray) energy range
    :param normalization: [flux unit], (float) normalization value
    :param normalization_error: [flux unit], (float or list) normalization error or [err_min, err_max]
    :param powerlaw_index: [DL], (float) power law index
    :param powerlaw_index_error: [DL], (float or list) power law index error or [err_min, err_max]
    :param reference_energy: [u.keV or other energy], (float) reference energy
    :return: model values [flux unit], minimal line, maximal line [flux unit]
    """
    norm_min = normalization - normalization_error
    norm_max = normalization + normalization_error
    index_min = powerlaw_index - powerlaw_index_error
    index_max = powerlaw_index + powerlaw_index_error

    flux = normalization * (energy_range / reference_energy) ** powerlaw_index
    f1 = norm_min * (energy_range / reference_energy) ** index_min
    f2 = norm_min * (energy_range / reference_energy) ** index_max
    f3 = norm_max * (energy_range / reference_energy) ** index_min
    f4 = norm_max * (energy_range / reference_energy) ** index_max

    flux_array = np.array([f1.value, f2.value, f3.value, f4.value])
    flux_unit = flux.unit

    butterfly_min = np.min(flux_array, axis=0)
    butterfly_max = np.max(flux_array, axis=0)

    return flux, butterfly_min * flux_unit, butterfly_max * flux_unit


if __name__ == '__main__':
    print("Not for direct use.")

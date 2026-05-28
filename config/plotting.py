import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('default')

from config.settings import PNG_PICS_DIR, PDF_PICS_DIR

Linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

Tab10 = mpl.color_sequences["tab10"]
Tab20 = mpl.color_sequences["tab20"]

royalblue_palette = [
    '#6A89FF',  # Bright blue — noticeable, but not too light
    '#4169E1',  # Base RoyalBlue
    '#1F3A8A'   # Deep blue — high contrast, still blue (not blackish)
]

seagreen_palette = [
    '#97c4ab',  # Bright-green
    '#2e8b57',  # Base Seagreen
    '#1a5c43',  # Dark-green
]

orangered_palette = [
    '#FFA07A',  # LightSalmon (brighter variant)
    '#FF4500',  # Orangered (base color)
    '#CC3700',  # Darker Orangered
    '#992800'   # Even darker shade
]

mediumorchid_palette = [
    "#4b2354",  # très foncé, bonne lisibilité sur fond clair
    "#8941a3",  # foncé, atténué mais distinct
    "#ba55d3",  # mediumorchid original
    "#e7b6f7"   # très clair, pour contraste sur papier blanc
]


def set_plotting_defaults():
    # default figure size
    mpl.rcParams["figure.figsize"] = [7, 5]

    # Set default font size
    mpl.rcParams['font.size'] = 12

    # Grid parameters
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.color'] = 'lightgray'

    return


def set_plotting_serif():
    set_plotting_defaults()
    # Enable TeX rendering
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",  # Use serif fonts
        "axes.labelsize": 20,  # Axis labels font size
        "axes.titlesize": 20,  # Title font size
        "legend.fontsize": 18,  # Legend font size
        "xtick.labelsize": 14,  # X-axis tick font size
        "ytick.labelsize": 14,  # Y-axis tick font size
        "font.serif": ["Computer Modern"],  # LaTeX default font
    })
    return


def set_inverted_defaults():
    # default figure size
    mpl.rcParams["figure.figsize"] = [7, 5]

    # Set default font size
    mpl.rcParams['font.size'] = 12

    # Grid parameters
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.color'] = 'lightgray'

    mpl.rcParams['xtick.color'] = 'white'
    mpl.rcParams['ytick.color'] = 'white'
    mpl.rcParams['axes.edgecolor'] = 'white'
    mpl.rcParams['axes.labelcolor'] = 'white'
    return


def save_figure(title, dpi=600):
    plt.savefig(os.path.join(PNG_PICS_DIR, f"{title}.png"), dpi=dpi, transparent=True)
    plt.savefig(os.path.join(PDF_PICS_DIR, f"{title}.pdf"))
    return


if __name__ == '__main__':
    print('Not for direct use')
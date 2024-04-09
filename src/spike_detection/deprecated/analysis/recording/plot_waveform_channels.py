from src.plot import set_dpi
from src.utils import FACTOR_UV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def plot_waveform_channels(kilosort_npz_path, thresh_amp, thresh_std,
                           scale_v, scale_h, xlim=None, ylim=None,
                           curation_colors=("red", "black")):
    """
    Plot the mean waveforms (which are defined on different channels) of a unit at their spatial locations
    and color code them based on whether they are curated
    #
    :param kilosort_npz_path:
        spikesort_matlab4.py --> sorted.npz
    :param thresh_amp:
        Minimum amplitude (in arbitrary scaled down units)
    :param thresh_std:
        Minimum standard deviation in amplitude divided by amplitude
    :param scale_v:
        Vertical stretch factor of wf
    :param scale_h:
        Horizontal stretch factor of wf
    :param xlim:
        of plot
        None --> use auto-generated lim
    :param ylim:
        of plot
        None --> use auto-generated lim
    :param curation_colors:
        (failed color, passed color)
    """

    npz = np.load(kilosort_npz_path, allow_pickle=True)
    locations = npz["locations"]
    # Iterate through units
    for i_u, unit in enumerate(npz["units"][10:11]):
        # Set up plot
        fig, ax = plt.subplots(1)  # type: Figure, Axes
        ax.set_title(i_u)
        ax.set_aspect("equal")

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Get waveform data
        templates = unit["template"].T   # (n_channels, n_samples)
        center_i = templates.shape[1] // 2
        is_curated = (unit["amplitudes"] >= thresh_amp) * (unit["std_norms"] <= thresh_std)  # type: np.ndarray

        # Plot waveforms
        for i, wf in enumerate(templates):
            color = curation_colors[int(is_curated[i])]

            wf *= scale_v
            loc = locations[i]

            x = np.arange(wf.size, dtype=float) - center_i
            x *= scale_h

            x += loc[0]
            wf += loc[1]
            ax.plot(x, wf, color=color)

        # Add scalebar
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax.transData,
                                   FACTOR_UV * thresh_amp, '20 m', 'lower right',
                                   pad=0.1,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)

        plt.tight_layout()
        plt.show()


def main():
    # spikesort_matlab4.py -> sorted.npz
    KILOSORT_NPZ_PATH = r"C:/KosikLab/2953/sorted.npz"

    # Minimum amplitude
    THRESH_AMP = 3
    # Minimum standard deviation in amplitude divided by amplitude
    THRESH_STD = 0.6

    # Vertical stretch factor of wf
    SCALE_V = 3
    # Horizontal stretch factor of
    SCALE_H = 0.5

    XLIM = (200, 600)  # (-50, 3850+50)
    YLIM = (1400, 1700)  # (-50, 2100+50)

    # First element is color of waveforms that fail at least one threshold. Second element is color of waveforms that are curated
    CURATED_COLORS = ["red", "black"]

    set_dpi(400)

    npz = np.load(KILOSORT_NPZ_PATH, allow_pickle=True)
    locations = npz["locations"]
    # Iterate through units
    for i_u, unit in enumerate(npz["units"][10:11]):
        # Set up plot
        fig, ax = plt.subplots(1)  # type: Figure, Axes
        ax.set_title(i_u)
        ax.set_aspect("equal")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

        # Get waveform data
        templates = unit["template"].T  # (n_channels, n_samples)
        center_i = templates.shape[1] // 2
        is_curated = (unit["amplitudes"] >= THRESH_AMP) * (unit["std_norms"] <= THRESH_STD)  # type: np.ndarray

        # Plot waveforms
        for i, wf in enumerate(templates):
            color = CURATED_COLORS[int(is_curated[i])]

            wf *= SCALE_V
            loc = locations[i]

            x = np.arange(wf.size, dtype=float) - center_i
            x *= SCALE_H

            x += loc[0]
            wf += loc[1]
            ax.plot(x, wf, color=color)

        # Add scalebar
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax.transData,
                                   FACTOR_UV * THRESH_AMP, '20 m', 'lower right',
                                   pad=0.1,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

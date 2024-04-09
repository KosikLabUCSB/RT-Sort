import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from scipy.io import loadmat
import numpy as np


def plot_propagation_signal(list_of_propagation, channel_map,
                            axis=None,
                            kilosort_npz_path=None,
                            prop_scale=1, prop_color_a=.2, prop_color_b=0.6, prop_alpha=0.8,
                            electrode_color="black", electrode_size=1,
                            x_label="x (µm)", y_label="y (µm)"):
    """
    This function plots the propagation signals found in list_of_propagation

    If kilosort_mat_path (.mat file) is not None, the kilosort's detected units will be plotted
    Different propagation signals will be plotted in different shades of blue
    Different units detected by kilosort will be plotted in different shades of red

    The propagation signals will be plotted in different shades of blue
    If kilosort's units are plotted as well, they will be plotted in different shades of red

    Inputs:
        list_of_propagation: list
            First output of automated_detection_propagation
            Contains P elements, each a pandas.DataFrames (each with 4 columns) of electrode cohorts for
            each propagation (p) in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
        channel_map: list
            Second output of get_inputs_from_maxwell
            Same length as spike_times. Contains a mapping from index in spike_times to
            (channel_id, electrode_id, x, y). I.e. the ith electrode in spike_times corresponds
            to the ith data in channel_map.
        axis: Axes
            Axis to plot on
            If None, a new figure will be created and automatically shown
        kilosort_npz_path: None or str
            The path to the .npz file containing kilosort's detected units. These units will be plotted
            on the same plot as the propagation signals with the same size and color parameters
            If None, this will not be plotted
        prop_scale: int
            Value to scale the size of electrodes that belong to a signal propagation
            The size of electrodes belonging to a propagation signal is prop_scale * electrode_size
        prop_color_a: float
            Must be within [0, 1)
            Each value in the RGB triplet for each propagation signal will be between [a, b)
        prop_color_b: float
            Must be within [0, 1) and greater than prop_color_a
            Each value in the RGB triplet for each propagation signal will be between [a, b)
        prop_alpha: float
            The alpha for each propagation signal
            1 = fully opaque
            0 = fully transparent
        electrode_color: str
            Color of each electrode
            Name of color (such as black) or hex value (such as #000000)
        electrode_size: int
            Size of each electrode
        x_label: str
            Label of x-axis
        y_label: str
            Label of y-axis
    """
    assert prop_color_b > prop_color_a, "'prop_color_b' must be greater than prop_color_a"
    assert 0 <= prop_color_a <= 1, "'prop_color_a' must be within [0, 1]"
    assert 0 <= prop_color_b <= 1, "'prop_color_b' must be within [0, 1]"

    if axis is None:
        axis = plt.subplot()  # type: Axes
        show = True
    else:
        show = False

    # Plot all electrodes
    for (channel, electrode, x, y) in channel_map:
        axis.scatter(x, y, color=electrode_color, s=electrode_size)

    # Plot propagation signals
    prop_points = set()
    for prop in list_of_propagation:
        color = np.random.random(3) * (prop_color_b - prop_color_a) + prop_color_a
        if kilosort_npz_path is not None:
            color[2] = 1
        size = electrode_size * prop_scale
        x_locs = []
        y_locs = []
        for id in prop.loc[:, "ID"]:
            mapping = channel_map[int(id)]
            x, y = mapping[2], mapping[3]
            x_locs.append(x)
            y_locs.append(y)
            prop_points.add((x, y))
        axis.scatter(x_locs, y_locs, color=color, s=size, alpha=prop_alpha)
        for i in range(len(x_locs)-1):
            x, y = x_locs[i], y_locs[i]
            axis.arrow(x, y, x_locs[i+1]-x, y_locs[i+1]-y, color=color, linewidth=1.5*size, alpha=prop_alpha)
        # axis.plot(x_locs, y_locs, color=color, linewidth=1.5*size, alpha=prop_alpha)

    # Plot kilosort's units
    if kilosort_npz_path is not None:
        kilosort = np.load(kilosort_npz_path, allow_pickle=True)
        for unit in kilosort["units"]:
            x = unit["x_max"]
            y = unit["y_max"]
            color = np.random.random(3) * (prop_color_b - prop_color_a) + prop_color_a
            if (x, y) in prop_points:
                color[1] = 1
                alpha = 1
                zorder = 10
            else:
                color[0] = 1
                alpha = prop_alpha
                zorder = 1
            axis.scatter(x, y, color=color, s=electrode_size*prop_scale, alpha=alpha, zorder=zorder)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_aspect("equal")

    if show:
        plt.show()


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 600

    recording = "data/220705/16460/000439"

    list_of_propagation = np.load(f"{recording}/list_of_propagation.npy", allow_pickle=True).tolist()
    channel_map = np.load(f"{recording}/channel_map.npy", allow_pickle=True).tolist()
    list_of_propagation = [pd.DataFrame(prop, columns=["ID", "latency", "small_window_cooccurrences", "n1_n2_ratio"])
                           for prop in list_of_propagation]

    fig, a0 = plt.subplots()
    plot_propagation_signal(list_of_propagation, channel_map, a0, f"{recording}/sorted.npz")
    a0.set_title(recording)
    plt.show()

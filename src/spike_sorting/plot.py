"""
Default figure size: 6.4, 4.8
    For plotting neuropixels: 
        With color bar: figsize=(3.6, 4.8)
        Without: figsize=(2.8, 4.8)

"""

import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from spikeinterface.extractors import MaxwellRecordingExtractor


TRACE_X_LABEL = "Time (ms)"


def set_dpi(dpi):
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = dpi


def get_yticks_lim(trace, anchor=0, increment=5,
                   buffer_min=5, buffer_max=3):
    """
    Get lim and ticks for y-axis when trace is plotted

    :param trace: np.array
        Trace that will be plotted using the returned lim and ticks
    :param anchor: int or float
        The ticks will show anchor
    :param increment: int or float
        Increment between ticks
    :param buffer_min:
        Ticks will be within [min(trace) - buffer_min, max(trace) + buffer_max)]
    :param buffer_max:
        [min(trace) - buffer_min, max(trace) + buffer_max)]
    """
    trace_min = min(trace) - buffer_min
    trace_max = max(trace) + buffer_max

    ylim = (trace_min, trace_max)
    yticks = np.arange(
                anchor + np.floor(min(trace) / increment) * increment,
                anchor + np.ceil(max(trace) / increment) * increment + 1,
                increment
            )
    return yticks, ylim


def set_ticks(subplots: tuple, trace: np.array,
              sampling_frequency=30, center_xticks=False):
    """
    Set x and y ticks for subplots

    :param subplots
        Each element is a subplot
    :param trace
        The trace to calculate the appropatiate ticks for
    :param sampling_frequency
        Sampling frequency of the recording
    :param center_xticks
        Whether to set center of xticks to 0 (left is negative time and right is positive)
    """
    yticks, ylim = get_yticks_lim(trace, 0, 5)

    sample_size = len(trace.flatten())
    xlim = (0, sample_size)
    xtick_locs = np.arange(0, sample_size + 1, 20)
    xtick_labels = xtick_locs / sampling_frequency
    if center_xticks:
        xtick_labels -= (xtick_labels[-1] - xtick_labels[0]) / 2
    xtick_labels = xtick_labels.astype(int)
    for sub in subplots:
        sub.set_yticks(yticks)
        sub.set_ylim(ylim)

        sub.set_xticks(xtick_locs)  # , xtick_labels
        sub.set_xlim(xlim)

        sub.set_xlabel(TRACE_X_LABEL)


def plot_hist_percents(data, decimals=None, labels=False, **hist_kwargs):
    """
    # Plot a histogram with percents as y-axis
    # https://www.geeksforgeeks.org/matplotlib-ticker-percentformatter-class-in-python/

    :param data:
    :param decimals:
    :param labels:
        If True, add the percent above each bin
    :param hist_kwargs:
    :return:
    """

    counts, edges, patches = plt.hist(data, weights=np.ones(len(data)) / len(data), **hist_kwargs)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=decimals))
    plt.ylabel("Frequency")

    if labels:
        dx = (edges[1] - edges[0]) / 2
        for i, count in enumerate(counts):
            plt.text(edges[i] + dx, count, f"{count*100:.1f}%", ha="center", va="bottom")


def hist(data, axis=None, labels=False, xlim=None, **hist_kwargs):
    # Plot a histogram with optional labels on each bar showing the count
    # axis is axis to plot on
    if xlim is not None:
        hist_kwargs["range"] = xlim
    
    if axis is not None:
        plt.sca(axis)
        
    counts, edges, patches = plt.hist(data, **hist_kwargs)
    
    # plt.ylabel("Frequency")
    if hist_kwargs.get("range", None) is not None:
        plt.xlim(hist_kwargs["range"])

    if labels:
        dx = (edges[1] - edges[0]) / 2
        for i, count in enumerate(counts):
            plt.text(edges[i] + dx, count, int(count), ha="center", va="bottom")

def bar(data, xmax=None, fill_empty=True, **bar_kwargs):
    """
    x_max: If not None, include bars with 0 height up to x_max
    fill_empty: If True, include bars between values
    
    **bar_kwargs
        width: width of bar
    """
    if "width" not in bar_kwargs:
        bar_kwargs["width"] = 0.7

    bars_og, counts_og = np.unique(data, return_counts=True)
    bars_og = bars_og.tolist()
    counts_og = counts_og.tolist()
    # Fill empty values between bars
    if fill_empty and not isinstance(bars_og[0], float):
        bars = []
        counts = []
        for b in range(bars_og[0], bars_og[-1]+1):
            bars.append(b)
            if b not in bars_og:
                counts.append(0)
            else:
                counts.append(counts_og[bars_og.index(b)])
    else:
        bars, counts = bars_og, counts_og
        
    # Extend bars
    if xmax is not None and xmax > bars[-1]:
        counts.extend([0] * (xmax-bars[-1]))
        bars.extend(range(bars[-1]+1, xmax+1))

    # Plot
    df = DataFrame(data=counts, index=bars)
    df.plot.bar(rot=0, legend=False, **bar_kwargs)
    
    
def plot_wf(wf, samp_freq=30):
    x_values = np.arange(wf.shape[1])
    x_values -= x_values.size // 2
    x_values = x_values / samp_freq
    for w in wf:
        plt.plot(x_values, w)
    plt.xlabel("Time rel. trough (ms)")
    plt.ylabel("Microvolts")

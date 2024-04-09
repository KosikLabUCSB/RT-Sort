import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from spikeinterface.extractors import NwbRecordingExtractor, MaxwellRecordingExtractor
from spikeinterface.preprocessing import bandpass_filter, scale
import si


def plot_traces(rec_path, start_frame, end_frame, channel_ids, channel_spacer=150, line_times=None, line_colors=None):
    rec = NwbRecordingExtractor(rec_path)
    rec = scale(rec, gain=rec.get_channel_gains(), offset=rec.get_channel_offsets(), dtype="float32")
    filt = bandpass_filter(rec, freq_min=300, freq_max=3000, dtype="float32")

    fig, (a0, a1) = plt.subplots(2, tight_layout=True)
    a0.set_title("Raw")
    a1.set_title("Filtered")

    x = np.arange(start_frame, end_frame, dtype=float) / rec.get_sampling_frequency()
    a0.set_xlim(min(x), max(x))
    a1.set_xlim(min(x), max(x))

    traces_raw = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_scaled=False)
    traces_filt = filt.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_scaled=False)

    # Vertically split up traces plotted from different channels
    traces_raw -= np.median(traces_raw, axis=0)

    if traces_raw.ndim == 2:
        spacer = np.arange(traces_raw.shape[1])
        traces_raw += spacer * channel_spacer
        traces_filt += spacer * channel_spacer

    a0.plot(x, traces_raw)
    a1.plot(x, traces_filt)

    if line_times is not None:
        for i in range(len(line_times)):
            time = line_times[i]
            time /= rec.get_sampling_frequency()
            if line_colors is not None:
                color = line_colors[i]
            else:
                color = "black"
            a0.axvline(time, color=color, linestyle="dashed", label="Spike")
            a1.axvline(time, color=color, linestyle="dashed", label="Spike")

            # a0.legend()
            # a1.legend()

    a1.set_xlabel("Time (s)")

    plt.show()


def plot_propagations(list_of_propagation: list, channel_map: dict,
                      xlim_buffer=30, ylim_buffer=120,
                      size_scale=600, elec_color="black",
                      xlabel="x (µm)", ylabel="y (µm)"):
    """
    Plot propagations

    :param list_of_propagation:
    :param channel_map:
        Dict of {channel_id: (x, y)}
    :param xlim_buffer:
        lims will be [min-lim_buffer, max+lim_buffer]
    :param ylim_buffer:
        lims will be [min-lim_buffer, max+lim_buffer]
    :param size_scale:
        Size of dot = num_cooccurrences * size_scale
    :param elec_color:
        Color of electrodes
    :param xlabel:
        x-label of plot
    :param ylabel:
        y-label of plot
        y (µm)
    """

    # Plot each unit sequence on a separate plot
    for prop in list_of_propagation:
        x_values = []
        y_values = []
        latencies = []
        sizes = []
        for i in range(len(prop)):
            elec = prop.iloc[i]
            x, y = channel_map[int(elec.ID)]
            x_values.append(x)
            y_values.append(y)
            latencies.append(elec.latency)
            sizes.append(elec.small_window_cooccurrences)
        plt.scatter(x_values, y_values, s=np.asarray(sizes) / max(sizes) * size_scale, c=latencies, cmap="viridis")
        plt.colorbar(ticks=np.unique(latencies), label="latency (ms)")

        for x, y in channel_map.values():
            plt.scatter(x, y, s=100, c=elec_color, marker="s", zorder=-100)

        plt.xlim(
            min(x_values)-xlim_buffer,
            max(x_values)+xlim_buffer
        )
        plt.ylim(
            max(0, min(y_values)-ylim_buffer),
            max(y_values)+ylim_buffer
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_aspect("equal")
        plt.show()

    plt.show()


def plot_templates_units(npz_path: str, unit_idx=None):
    """
    Plot units' templates

    :param npz_path:
    :param unit_idx:
        If not None, only plot this unit
    """

    npz = np.load(npz_path, allow_pickle=True)
    units = npz["units"]

    if unit_idx is not None:
        units = [units[unit_idx]]

    count = 0
    count_200 = 0
    for i, unit in enumerate(units):
        template = unit["template"]
        chan_max = np.min(template, axis=0).argmin()

        # if len(unit["spike_train"]) < 30:
        #     continue

        # fig, ax = plt.subplots(1)
        # x = np.arange(template.shape[0]) / npz["fs"] * 1000
        # ax.plot(x, template[:, chan_max])
        # ax.set_title(f"Unit Index: {i}")
        # ax.set_ylabel("Microvolts")
        # ax.set_xlabel("Time (ms)")
        # plt.show()

        if np.abs(np.min(template)) > 200:
            count_200 += 1
        count += 1
    print(count_200, count)


def plot_templates_space(npz_path: str, unit_idx: int, ylim: tuple):
    """
    Plot np.load(npz_path)["units"][unit_idx]["templates"] in space

    :param npz_path:
    :param unit_idx:
        Index of unit
    :param ylim:
    """

    npz = np.load(npz_path, allow_pickle=True)
    locations = npz["locations"]
    unit = npz["units"][unit_idx]

    fig, ax = plt.subplots(1)  # type: None, Axes

    templates = unit["template"]
    center_idx = templates.shape[0]
    for i in range(templates.shape[1]):
        loc = locations[i]
        temp = templates[:, i]

        x = np.arange(temp.size, dtype=float) - center_idx
        x += loc[0] * 10
        temp += loc[1] * 10
        ax.plot(x, temp)

    ax.set_ylim(*ylim)
    plt.show()


def plot_kilosort_spikes(nwb_path: str, npz_path: str, unit_idx: int,
                         spike_buffer=100, num_spikes=10):
    """
    Plot raw traces where np.load(npz_path)[unit_idx] detects spike
    [spike_time - spike_buffer, spike_time + spike_buffer)

    :param nwb_path:
        Path to .nwb path containing recording
    :param npz_path:
        Path to .npz path containing compiled results frmo spikesort_matlab4.py
    :param unit_idx:
    :param spike_buffer:
    :param num_spikes:
    """

    rec = NwbRecordingExtractor(nwb_path)
    # rec = bandpass_filter(rec, freq_min=300, freq_max=6000)

    unit = np.load(npz_path, allow_pickle=True)["units"][unit_idx]
    template = unit["template"]
    chan_max = np.argmin(np.min(template, axis=0))
    chan_id = rec.get_channel_ids()[chan_max]

    for st in np.random.choice(unit["spike_train"], num_spikes):
        print(st, chan_id)
        trace = rec.get_traces(start_frame=st-spike_buffer, end_frame=st+spike_buffer, channel_ids=[chan_id], return_scaled=True)
        plt.plot(trace)
        plt.show()


def plot_spikes(rec_path: str, unit_sequences: list, spike_times: list, buffer=200):
    idx = 10
    spike_times = spike_times[idx]
    sequence = unit_sequences[idx]

    if rec_path.endswith("nwb"):
        rec = NwbRecordingExtractor(rec_path)
    else:
        rec = MaxwellRecordingExtractor(rec_path)
    sampling_freq = rec.get_sampling_frequency() / 1000  # /1000 to convert Hz to kHz

    exclude_st = set(np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/ensemble_sorting_of_a_neuropixels_recording/working_15min/rec/kilosort2/spike_times.npy").flatten())
    for st in spike_times:
        st = int(st * sampling_freq)  # ms to samples
        end = False
        frames_buffer = int(np.ceil(sampling_freq * 0.4))
        for x in range(st-frames_buffer, st+frames_buffer):
            if x in exclude_st:
                end = True
                break
        if end:
            continue

        print(st)

        plot_traces(rec_path, start_frame=st-buffer, end_frame=st+buffer, channel_ids=sequence, line_times=[st])


def plot_unit_spike_counts(spike_counts: list, hist_range=(0, 14000), hist_bins=15):
    """
    Plot a histogram of the number of spikes from each unit

    :param spike_counts:
        Each element is the number of spikes from a unit
    :param hist_range:
        Range for plt.hist
    :param hist_bins:
        Number of bins for plt.hist
    """

    plt.hist(spike_counts, range=hist_range, bins=hist_bins)
    plt.title("Distribution of spike counts")
    plt.ylabel("Count")
    plt.xlabel("Number of spikes")
    plt.show()

    print(f"Mean number of spikes: {np.mean(spike_counts)}")
    print(f"STD number of spikes: {np.std(spike_counts)}")

    # SAMPLING_FREQUENCY = 30  # in kHz
    # KS_PATH = Path("/data/MEAprojects/dandi/000034/sub-mouse412804/ensemble_sorting_of_a_neuropixels_recording/working_15min/rec/kilosort2")
    # spike_times = np.load(KS_PATH / "spike_times.npy").flatten()
    # spike_clusters = np.load(KS_PATH / "spike_clusters.npy").flatten()
    # spike_trains_ks = standardize_kilosort_units(spike_times, spike_clusters, sampling_frequency=SAMPLING_FREQUENCY)
    #
    # num_spikes_ks = [len(st) for st in spike_trains_ks]
    # print(np.mean(num_spikes_ks))
    # print(np.std(num_spikes_ks))
    # plt.hist([len(st) for st in spike_trains_ks], range=(0, 14000), bins=15)
    # plt.title("Distribution of kilosort's units' number of spikes")
    # plt.ylabel("Count")
    # plt.xlabel("Number of spikes")
    # plt.show()
    #
    # # Get propagation algorithm spike trains
    # propagating_times = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/propagating_times_900s.npy", allow_pickle=True)
    # spike_trains_alg = [
    #     times[-1]
    #     for times in propagating_times
    # ]
    #
    # num_spikes_alg = [len(st) for st in spike_trains_alg]
    # print(np.mean(num_spikes_alg))
    # print(np.std(num_spikes_alg))
    # plt.hist(num_spikes_alg, range=(0, 14000), bins=15)
    # plt.title("Distribution of algorithm's units' number of spikes")
    # plt.ylabel("Count")
    # plt.xlabel("Number of spikes")
    # plt.show()


def plot_isis(spike_times: np.ndarray, isi_viol=1.5, **hist_kwargs):
    """
    Plot interspike-interval distribution

    :param spike_times:
        Spike times in standardized format (each element is a list containing a unit's spike times)
    :param isi_viol:
        Any consecutive spikes within isi_viol (ms) is considered an ISI violation
    :param hist_kwargs:
        kwargs for plt.hist
    """

    violations = []
    for unit in spike_times:
        num_spikes = len(unit)
        isis = np.diff(unit)

        violation_num = np.sum(isis < isi_viol)
        # violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        #
        # total_rate = num_spikes / total_duration
        # violation_rate = violation_num / violation_time
        # violation_rate_ratio = violation_rate / total_rate

        violation_percent = (violation_num / num_spikes) * 100
        violations.append(violation_percent)

    plt.hist(violations, **hist_kwargs)
    plt.title("ISI Violation Distribution")
    plt.xlabel("Percent ISI violations")
    plt.ylabel("Count")
    plt.show()


def plot_latencies_bar(list_of_propagation: list):
    """
    Plot a bar plot of the latencies of each electrode in each propagation relative to the first electrode.
    The first electrode is omitted from the histogram because its latency is always 0

    :param list_of_propagation:
    """
    bars = {}
    for prop in list_of_propagation:
        for lat in prop["latency"].values[1:]:  # [1:] to not include first electrode
            if lat not in bars:
                bars[lat] = 1
            else:
                bars[lat] += 1

    tick_label = [f"{l:.3}" for l in bars.keys()]
    plt.bar(list(bars.keys()), list(bars.values()), width=0.033333*0.8, tick_label=tick_label)
    plt.ylabel("Count")
    plt.xlabel("Latency with first electrode in propagation (ms)")
    plt.show()


def plot_num_electrodes_bar(list_of_propagation: list):
    """
    Plot a bar plot of the number of electrodes in each propagation

    :param list_of_propagation:
    """

    bars = {}
    for prop in list_of_propagation:
        num = len(prop)
        if num not in bars:
            bars[num] = 1
        else:
            bars[num] += 1
    plt.bar(list(bars.keys()), list(bars.values()))
    plt.ylabel("Count")
    plt.xlabel("Number of electrodes in propagation")
    plt.show()


def main_allen_inst():
    SESSION_ID = 757216464
    PROBE_ID = 769322753
    UNIT_SEQUENCES_ALG_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_unit_sequences_alg.npy"
    UNIT_SEQUENCES_ALLEN_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_unit_sequences_allen.npy"
    CHANNEL_MAP_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_channel_map.npy"

    # plot_unit_sequences(
    #     np.load(UNIT_SEQUENCES_ALG_PATH, allow_pickle=True),
    #     np.load(UNIT_SEQUENCES_ALLEN_PATH, allow_pickle=True),
    #     np.load(CHANNEL_MAP_PATH, allow_pickle=True)[()]
    # )


def main():
    # plot_traces("/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb", start_frame=0, end_frame=int(1e4), channel_ids=list(range(2)))
    # exit()

    from run_alg import standardize_alg_outputs
    rec_path = "/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb"
    channel_map = si.get_channel_map(rec_path)

    list_of_propagation = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/list_of_propagation_final.npy", allow_pickle=True)
    propagating_times = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/propagating_times_final.npy", allow_pickle=True)
    # list_of_propagation = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/list_of_propagation_neg_only_ccg_91.npy", allow_pickle=True)
    # propagating_times = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/propagating_times_neg_only_ccg_91.npy", allow_pickle=True)
    # list_of_propagation = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/list_of_propagation_900s.npy", allow_pickle=True)
    # propagating_times = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/propagating_times_900s.npy", allow_pickle=True)

    unit_sequences, unit_spike_times = standardize_alg_outputs(list_of_propagation, propagating_times, rec_path, max_electrodes=None)
    spike_trains_alg = [len(st) for st in unit_spike_times]

    # from pathlib import Path
    # SAMPLING_FREQUENCY = 30  # in kHz
    # KS_PATH = Path("/data/MEAprojects/dandi/000034/sub-mouse412804/ensemble_sorting_of_a_neuropixels_recording/working_15min/rec/kilosort2")
    # spike_times = np.load(KS_PATH / "spike_times.npy").flatten()
    # spike_clusters = np.load(KS_PATH / "spike_clusters.npy").flatten()
    # from compare_sortings import standardize_kilosort_units
    # spike_trains_ks = standardize_kilosort_units(spike_times, spike_clusters, sampling_frequency=SAMPLING_FREQUENCY)

    # plot_latencies_bar(list_of_propagation)
    # plot_num_electrodes_bar(list_of_propagation)
    # plot_unit_spike_counts([len(st) for st in unit_spike_times])
    # plot_isis(unit_spike_times, bins=20)
    plot_propagations(list_of_propagation, channel_map)
    # plot_spikes(rec_path, unit_sequences, unit_spike_times)


if __name__ == "__main__":
    main()

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

from src.sorters.prop_signal import PropSignal
from src.recording import Recording
from src import utils

RECORDING = Recording(utils.PATH_REC_SI)
PROP_SIGNAL = PropSignal("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/230423", RECORDING)
THRESH_CROSSINGS = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/prop_signal/thresh_5_full/crossings.npy", allow_pickle=True)

# In ms
MAX_DURATION = 0.5
ISI_VIOL = 1.5

##

thresh_crossings = {}
for elec, crossings in enumerate(THRESH_CROSSINGS):
    for cross in crossings:
        if cross not in thresh_crossings:
            thresh_crossings[cross] = [elec]
        else:
            thresh_crossings[cross].append(elec)
thresh_crossing_times = sorted(list(thresh_crossings.keys()))


# Keep track of latest threshold crossing on first electrode of each propagation
propagations = {elec: -np.inf for elec in range(RECORDING.get_num_channels())}  # Faster to have one for every electrode (more than number of props)

# Keep track of last detected spike for prop (prevents ISI violations of merged propagations)
last_detected_spikes = {}
for prop in PROP_SIGNAL:
    last = [-np.inf]  # Use array since passed by reference
    for df in prop.df:
        last_detected_spikes[df.ID[0]] = last

# Go from an electrode with thresh crossing to props with that electrode
elec_to_prop = {elec: [] for elec in range(RECORDING.get_num_channels())}  # Go from an electrode with thresh crossing to props with that electrode
for prop in PROP_SIGNAL:
    for df in prop.df:
        first_elec = df.ID[0]
        for elec in df.ID[1:]:
            elec_to_prop[elec].append(first_elec)

detected_spikes = []

processing_time = thresh_crossing_times[0]
for time in thresh_crossing_times:
    electrode_crossings = thresh_crossings[time]
    processing_end = None
    if time < processing_time:
        add_time = processing_time - time
    else:
        add_time = 0
        processing_time = time
    processing_start = perf_counter()
    for elec in electrode_crossings:
        propagations[elec] = time  # Faster to just set value than use an if-statement

    for elec in electrode_crossings:
        first_elecs = elec_to_prop[elec]
        for first_elec in first_elecs:
            first_time = propagations[first_elec]
            if time - first_time <= MAX_DURATION and first_time - last_detected_spikes[first_elec][0] > ISI_VIOL:
                propagations[first_elec] = -np.inf
                last_detected_spikes[first_elec][0] = first_time
                processing_end = perf_counter()

                add_start = perf_counter()
                detected_spikes.append((
                    time, first_time,
                    (processing_end - processing_start)*1000 + (time - first_time) + add_time
                ))
                add_end = perf_counter()
                processing_start += add_end - add_start

    processing_end_2 = perf_counter()
    if processing_end is None:
        processing_end = processing_end_2

    processing_time += (processing_end - processing_start) * 1000

from src import plot
plot.set_dpi(400)

detection_delays = [ds[2] for ds in detected_spikes]
detection_delays_inst = [ds[2] - (ds[0] - ds[1]) for ds in detected_spikes]
prop_delays = [ds[0] - ds[1] for ds in detected_spikes]

plot.plot_hist_percents(detection_delays, range=(0, MAX_DURATION), bins=20)
plt.xlim(0, MAX_DURATION)
plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
plt.xlabel("Time delay (ms)")
plt.title("Time delay from spike to detection.")
plt.tight_layout()
plt.show()

plot.plot_hist_percents(detection_delays_inst, range=(0, MAX_DURATION), bins=20)
plt.xlim(0, MAX_DURATION)
plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
plt.xlabel("Time delay (ms)")
plt.title("Time delay caused by integrating prop.")
plt.tight_layout()
plt.show()

plot.plot_hist_percents(detection_delays_inst, bins=50, range=(min(detection_delays_inst), 0.01))
plt.xlim(min(detection_delays_inst), 0.01)
plt.xlabel("Detection delay (ms)")
plt.title("Time delay caused by integrating prop.")
plt.tight_layout()
plt.show()
plt.show()

plot.plot_hist_percents(prop_delays, bins=20)
plt.xlim(0, MAX_DURATION)
plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
plt.xlabel("Time delay (ms)")
plt.title("Time delay caused by spike propagating across electrodes.")
plt.tight_layout()
plt.show()

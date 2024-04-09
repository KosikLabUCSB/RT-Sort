import h5py as h5
import numpy as np
from tqdm import tqdm

# Get spike times from maxwell recording file
maxwell = h5.File("/data/MEAprojects/organoid/intrinsic/200123/2953/network/data.raw.h5")
SAMPLING_FREQ = 20 # in kHz
spike_times_maxwell = {}
for st in maxwell["proc0/spikeTimes"]:
    frame, channel, amp = st
    frame /= 20  # Frames to ms
    if channel not in spike_times_maxwell:
        spike_times_maxwell[channel] = [frame]
    else:
        spike_times_maxwell[channel].append(frame)

spike_times = [spike_times_maxwell[chan] for chan in spike_times_maxwell]


spike_times_mine = np.load("/data/MEAprojects/organoid/intrinsic/200123/2953/network/thresh_crossings_5.npy", allow_pickle=True)

# maxwell = spike_times_maxwell[0]
# mine = spike_times_mine[0]
#
# import matplotlib.pyplot as plt
# st = int(maxwell[0] * 20)
# from spikeinterface.extractors import MaxwellRecordingExtractor
# from spikeinterface.preprocessing import bandpass_filter, scale
# rec = MaxwellRecordingExtractor("/data/MEAprojects/organoid/intrinsic/200123/2953/network/data.raw.h5")
# rec = scale(rec, rec.get_channel_gains(), rec.get_channel_offsets())
# rec = bandpass_filter(rec, freq_min=300, freq_max=3000, dtype="float32")  # type: MaxwellRecordingExtractor
#
# trace = rec.get_traces(start_frame=st-200, end_frame=st+200, channel_ids=rec.get_channel_ids()[0])
# plt.plot(trace)
# noise = np.sqrt(np.mean(trace ** 2))
# plt.axhline(noise * 5)
# plt.axhline(-noise * 5)
# plt.show()
# exit()



match = 0
total_maxwell = 0
total_mine = 0
for c in tqdm(range(spike_times_mine.size)):
    st_mine = spike_times_mine[c]
    st_mine = np.atleast_1d(st_mine)
    total_mine += len(st_mine)

    if c in spike_times_maxwell:
        st_maxwell = spike_times_maxwell[c]
        total_maxwell += len(st_maxwell)

print(total_maxwell)
print(total_mine)

total = 0
for c in spike_times_maxwell:
    total += len(spike_times_maxwell[c])
print(total)

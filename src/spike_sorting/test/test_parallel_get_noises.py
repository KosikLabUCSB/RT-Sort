import numpy as np
from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.preprocessing import scale, bandpass_filter
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_chunk_noise(start_frame):
    traces = rec.get_traces(start_frame=start_frame, end_frame=start_frame+CHUNK_SIZE, channel_ids=CHANNEL_IDS)
    # return np.median(np.abs(traces - np.median(traces))) / 0.6744897501960817
    return np.sqrt(np.mean(traces ** 2, axis=0))


rec = MaxwellRecordingExtractor("/data/MEAprojects/organoid/intrinsic/200123/2953/network/data.raw.h5")
rec = scale(rec, rec.get_channel_gains(), rec.get_channel_offsets())
rec = bandpass_filter(rec, freq_min=300, freq_max=3000)  # type: MaxwellRecordingExtractor
CHANNEL_IDS = rec.get_channel_ids()

CHUNK_SIZE = 1000

from multiprocessing import Pool

noises = []
with Pool(processes=20) as pool:
    tasks = range(0, rec.get_total_samples()-CHUNK_SIZE, CHUNK_SIZE)
    for noise in tqdm(pool.imap(get_chunk_noise, tasks, chunksize=len(tasks) // 20), total=len(tasks)):
        noises.append(noise)
# plt.hist(noises)
# plt.show()
# exit()

noises = np.array(noises)
plt.hist(np.mean(noises, axis=0))
plt.title("All")
plt.show()
for noise in noises.T:
    plt.hist(noise)
    plt.show()
print()

# from spikeinterface.core import get_noise_levels
# noise = get_noise_levels(rec, return_scaled=False)
# plt.hist(noise)
# plt.show()


# CHANNEL_IDS = rec.get_channel_ids()[100]
# CHUNK_SIZE = 1000
# noises = []
# for start in tqdm(range(0, rec.get_total_samples()-CHUNK_SIZE, CHUNK_SIZE)):
#     trace = rec.get_traces(start_frame=start, end_frame=start+CHUNK_SIZE, channel_ids=CHANNEL_IDS)
#     noise = np.sqrt(np.mean(trace ** 2))
#     noises.append(noise)
# plt.hist(noises)
# plt.show()

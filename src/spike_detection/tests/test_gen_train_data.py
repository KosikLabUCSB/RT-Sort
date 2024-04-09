import braingeneers.data.datasets_electrophysiology as de
from braingeneers.utils.numpy_s3_memmap import NumpyS3Memmap
import numpy as np
import matplotlib.pyplot as plt


def get_non_spike_time(spike_times, buffer, rec_n_samples):
    """
    Get a time from recording that does not contain a spike
    (all units in samples)

    :param spike_times: np.array
        Contains spike times
    :param buffer: int
        If s is a spike, then returned time will not be within [s-buffer, s+buffer]
    :param rec_n_samples: int
        Total number of samples in recording
    :return: non_spike_time: int
        Time that does not contain a spike
    """
    n = 0
    while True:
        n += 1
        non_spike_time = np.random.randint(rec_n_samples)
        for s in spike_times:
            if s - buffer <= non_spike_time <= s + buffer:
                break
        else:
            print(f"{n} tries to get non-spike time")
            return non_spike_time


S3_ROOT = "s3://braingeneers/ephys"
BATCH_UUID = "2022-06-30-e-UCSB-200123-2950-control"
EXPERIMENT_NUM = "experiment1"
SAMPLE_SIZE = 100
ATTR_CSV = "2950_atr.csv"
START = 0

SPIKE_TIMES_PATH = "derived/kilosort/spike_times.npy"
TEMPLATES_PATH = "derived/kilosort/templates_average.npy"

SPIKE_TIME_BUFFER = 10000  # 50
RECORDING_N_SAMPLES = int(1e6)

TRACE_BUFFER = 400

spike_times_path = f"{S3_ROOT}/{BATCH_UUID}/{SPIKE_TIMES_PATH}"
templates_path = f"{S3_ROOT}/{BATCH_UUID}/{TEMPLATES_PATH}"


print("Getting spike times")
spike_times = NumpyS3Memmap(spike_times_path)[:]

print("Getting templates metadata")
templates = NumpyS3Memmap(templates_path)

print("Getting non-spike time")
non_spike_time = get_non_spike_time(spike_times, SAMPLE_SIZE//2, RECORDING_N_SAMPLES)
print(f"non-spike time: {non_spike_time}")

print("Getting template")
random_unit = 0  # np.random.randint(templates.shape[0])
template = templates[random_unit, :, :]
chan_max = np.argmin(np.min(template, axis=0))
template = template[:, chan_max] / 6.295  # TODO: Recording and waveforms either have to be BOTH scaled or BOTH unscaled

print("Plotting template")
plt.plot(template)
plt.savefig("template.png")
plt.clf()

print("Getting raw data")
# TODO: Maximimum template channel is not the same as channel from raw data (index vs ID)
raw_data = de.load_data(de.load_metadata(BATCH_UUID), EXPERIMENT_NUM, non_spike_time, template.size + TRACE_BUFFER, chan_max)

print("Plotting raw data")
plt.plot(raw_data)
plt.ylim(470, 520)
plt.savefig("raw_data.png")
plt.clf()

print(f"Plotting raw data + template")
template_buffer = (raw_data.size - template.size) // 2
plt.plot(raw_data + np.pad(template, template_buffer))
plt.ylim(470, 520)
plt.axvline(template_buffer, c="black", linestyle="dotted")
plt.axvline(template_buffer + template.size, c="black", linestyle="dotted")
plt.savefig("raw_data+template.png")



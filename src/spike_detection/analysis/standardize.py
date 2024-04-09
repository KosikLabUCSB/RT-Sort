# Functions and classes to standard analysis across scripts and notebooks

REC_ROOT = "/data/MEAprojects/DLSpikeSorter/{}"
REC_FILE = REC_ROOT + "/data.raw.h5"


def count_spikes(spike_times):
    # Count number of spikes for data in standard format
    return sum(len(st) for st in spike_times)

MODEL_PATH = r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\models\v0_2\2950"
PATH_REC_NPY = r"C:\KosikLab\2950\data.raw.npy"
PATH_SORTED_NPZ = r"C:\KosikLab\2950\sorted.npz"


import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import ModelSpikeSorter
from tqdm import tqdm


class RealTest:
    """
    Class to test model's performance on real traces from a recording by taking windows of traces
    and feeding these windows into the model
    """
    def __init__(self, model, path_rec_npy, path_sorted_npz, stride=None, channels_first=False, windows_per_forward=32):
        """
        :param model: torch.nn.Model
            Model whose performance will be tested
        :param path_rec_npy: str or Path
            Path to .npy file containing raw traces from Maxwell .h5 file (see Recording.write_npy)
            that will be the test samples for the model
        :param path_sorted_npz: str or Path
            Path to .npz file containing spike sorted results from spikesort_matlab4.py
        :param stride: int
            Number of frames between adjacent windows
            If None, stride = model.sample_size
        :param channels_first: bool
            Traces have shape (n_channels, n_timepoints)
            If channels_first, traces will be looped through n_channels first then n_timepoints
            If not channels_first, traces will be looped through n_timepoints first then n_channels
        :param windows_per_forward: int
            Number of windows that will be passed into the model at a time
            Input to model: (windows_per_forward, 1, model.sample_size)
                1 since model only takes input from 1 channel in the recording at a time
        """
        self.model = model  # type: ModelSpikeSorter
        self.traces = np.load(path_rec_npy, mmap_mode="r")
        self.sorted = np.load(path_sorted_npz, mmap_mode="r", allow_pickle=True)

        if stride is not None:
            raise NotImplementedError("Setting stride to smaller/larger than model.sample_size (stride=None) will be slower")
        self.stride = stride if stride is not None else model.sample_size

        self.channels_first = channels_first
        self.windows_per_forward = windows_per_forward

    def perf(self, channels, start_frame, end_frame):
        if self.channels_first:
            raise NotImplementedError("Setting channels_first to True for RealTest is not implemented yet")

        model = self.model
        sample_size = self.model.sample_size
        windows_per_forward = self.windows_per_forward
        # stride = self.stride
        trace_len = sample_size * windows_per_forward

        preds_all = []
        for chan in tqdm(channels):
            # unit = self.sorted["units"][0]
            # start_frame = unit["spike_train"][0] - 100
            # chan = unit["max_channel_si"]
            for i in tqdm(range(start_frame, end_frame, trace_len)):
                # unit = self.sorted["units"][0]
                # start_frame = unit["spike_train"][0] - 100
                # chan = unit["max_channel_si"]

                print("Windows must overlap because of model.buffer_sample_front. If a spike occurs in buffer region, it will be impossible to detect")
                windows = self.traces[chan, i:i+trace_len].reshape((-1, 1, sample_size))
                windows = torch.tensor(windows - np.mean(windows, axis=2, keepdims=True),
                                       device="cuda", dtype=torch.float32)
                with torch.no_grad():
                    outputs = model(windows)
                preds = model.outputs_to_preds(outputs)
                preds_all.extend(preds)

                for j, p in enumerate(preds):
                    if len(p) > 0:
                        model.plot_preds(
                            [windows[j].cpu()],
                            [outputs[j]],
                            [p],
                            [1],
                            [100],
                            [30]
                        )


def main():
    model = ModelSpikeSorter.load(MODEL_PATH)
    real_test = RealTest(model, PATH_REC_NPY, PATH_SORTED_NPZ, windows_per_forward=5000)
    real_test.perf([444], 0, 3599800)


if __name__ == "__main__":
    main()

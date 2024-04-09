from src.sorters.base import *


class DLModelWindows:
    """Class to represent DL model's output on sliding window of recording"""

    def __init__(self, rec_windows_path, window_stride, num_channels,
                 model_outputs_path):

        self.stride = window_stride
        self.num_chans = num_channels
        self.windows = np.load(rec_windows_path, mmap_mode="r")
        self.outputs = np.load(model_outputs_path, mmap_mode="r")
        self.size_windows = self.windows.shape[-1]  # Size of windows
        self.size_outputs = self.outputs.shape[-1]  # Size of outputs

    # def plot_window_output(self, idx, title=None,
    #                        subplots=None):
    #     # Plot window and model's output
    #
    #     # Get data
    #     window = self.windows[idx].flatten()
    #     window -= np.mean(window)
    #     output = self.outputs[idx]
    #
    #     # Set up plots
    #     if subplots is None:
    #         fig, (a0, a1) = plt.subplots(2, figsize=(6, 4))
    #     else:
    #         fig, a0, a1 = subplots
    #     axes = a0, a1
    #     for a in axes:
    #         a.set_xlim(0, self.size_windows)
    #     set_ticks(axes, window)
    #
    #     if title is not None:
    #         a0.set_title(title)
    #
    #     # Plot trace
    #     a0.plot(window)
    #
    #     # Plot model's output
    #     output = torch.tensor(output)
    #     model.plot_loc_probs(output, a1)
    #     a1.set_title("")
    #
    #     if subplots is None:
    #         plt.show()

    def time_to_window(self, time, chan_idx):
        """
        Find i such that self.windows[i] contains :param time: within the model output region
        on the channel :param chan_idx:

        time of beginning of window as a function of idx
            channel = idx % num_channels
            idx -= channel  # Idx if getting window for 0th channel
            # For every num_channels indices, there is an increase in time of stride
            time = idx / num_channels * stride

            # Test
            IDX = 677576
            NUM_CHANNELS = 1020
            STRIDE = 120
            FRONT_BUFFER = 40

            channel = IDX % NUM_CHANNELS
            idx = IDX - channel
            time = idx / num_channels * STRIDE

            time = int(time)
            rec_trace = recording.get_traces_raw(time, time+200, channel).flatten()
            window = rec_windows.windows[IDX].flatten()
            plt.plot(rec_trace)
            plt.plot(window)
            plt.show()
            np.all(rec_trace==window)

        :param time:
        :param chan_idx:
        :return: tuple
            0) idx
            1) offset: location of spike in window (recording window, not output window)
        """
        assert self.stride == 120, "Not sure if time_to_window works if stride != model.num_output_locs"

        front_buffer = (self.size_windows - self.size_outputs) // 2

        idx_1_chan = max(0, time-front_buffer) // self.stride  # Index if getting recording only had 1 channel
        idx = int(idx_1_chan * self.num_chans + chan_idx)

        window_start = idx_1_chan * self.stride  # Time of first frame in window
        offset = int(time - window_start)

        return idx, offset

    def __len__(self):
        # Return number of windows
        return len(self.windows)

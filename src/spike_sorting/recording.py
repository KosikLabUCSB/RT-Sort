from spikeinterface import load_extractor
from spikeinterface.core import BaseRecording
from spikeinterface.extractors import MaxwellRecordingExtractor, NwbRecordingExtractor
from spikeinterface.toolkit.preprocessing import scale, bandpass_filter
import numpy as np
import matplotlib.pyplot as plt

from src.plot import set_ticks


class Recording:
    def __init__(self, path, freq_min=300, freq_max=3000, rec_name="", gain=True):
        # MaxwellRecordingExtractor is used for type hints because quacks like a duck
        if isinstance(path, BaseRecording):
            self.rec_raw : MaxwellRecordingExtractor = path
        elif str(path).endswith("h5"):
            self.rec_raw = MaxwellRecordingExtractor(path)
        elif str(path).endswith("nwb"):
            self.rec_raw = NwbRecordingExtractor(path)
        elif str(path).endswith("si"):  # Stored with {spikeinterface_extractor}.save_to_folder()
            self.rec_raw : MaxwellRecordingExtractor = load_extractor(path)
        else:
            raise ValueError(f"Recording '{path}' is not in .h5 or .nwb format")

        self.name = rec_name

        if not gain:
            gain = 1
            offset = 0
        elif self.rec_raw.has_scaled_traces():
            gain = self.rec_raw.get_channel_gains()
            offset = self.rec_raw.get_channel_offsets()
        else:
            print("Recording does not have scaled traces. Setting gain to 0.195")
            gain = 0.195  # 1.0
            offset = 0.0
        rec_filt = scale(self.rec_raw, gain=gain, offset=offset, dtype="float32")
        self.rec_filt = bandpass_filter(rec_filt, freq_min=freq_min, freq_max=freq_max)

        """
        If i is the index of a channel, then the ith element of self.nearest_chan is a tuple
            0) Closest channels indices: An where the 0th element is i, 1st element is closest channel, 2nd element is second closest channel, etc.
            1) Distance of channels
        """
        nearest_chan = []
        locs = self.get_channel_locations()
        for i in range(len(locs)):
            loc = locs[i]
            dists = np.sqrt(np.sum(np.square(locs - loc), axis=1))
            dists_sorted = np.sort(dists)
            chans_sorted = np.argsort(dists)

            nearest_chan.append((chans_sorted, dists_sorted))
        self.nearest_chan = nearest_chan

    def get_total_duration(self):
        return self.rec_raw.get_total_duration()

    def get_total_samples(self):
        return self.rec_raw.get_total_samples()

    def get_channel_locations(self):
        return self.rec_raw.get_channel_locations()

    def get_num_channels(self):
        return self.rec_raw.get_num_channels()

    def get_sampling_frequency(self):
        return self.rec_raw.get_sampling_frequency() / 1000  # in kHz

    def get_traces_raw(self, start_frame=None, end_frame=None, channel_ind=None):
        return Recording._get_traces(self.rec_raw, start_frame, end_frame, channel_ind)

    def get_traces_filt(self, start_frame=None, end_frame=None, channel_ind=None):
        return Recording._get_traces(self.rec_filt, start_frame, end_frame, channel_ind)

    def plot_traces(self, start_frame, end_frame, channel_ind=None, show=True):
        """

        :param start_frame:
        :param end_frame:
        :param channel_ind:
        :param show:
            If True, show figure
            If False, don't show figure and return (fig, plots)
        """

        traces = (self.get_traces_raw(start_frame, end_frame, channel_ind),
                  self.get_traces_filt(start_frame, end_frame, channel_ind))

        fig, plots = plt.subplots(1, 2, figsize=(10, 3))
        plots[0].set_title("Raw traces")
        plots[1].set_title("Filtered traces")

        for i in range(2):
            for trace in traces[i]:
                plots[i].plot(trace)
            plots[i].set_xlim(0, end_frame-start_frame)

        if show:
            plt.show()
        else:
            return fig, plots

    def plot_waveform(self, st, chan_center=None, n_before=40, n_after=40,
                      subplots=None,
                      save_path=None, close_window=False):
        """
        :param st:
            Spike time
        :param chan_center:
            Which channel the center waveform comes from
            (usually the channel the spike is detected on)
            If None, choose channel with maximum negative amplitude
        :param n_before:
            Number of frames before st to include
        :param n_after:
            Number of frames after st to include
        :param subplots:
            If None, create subplots
            Else, plot on subplots which is (fig, a0, a1)
        :param save_path:
            If not None, save figure to save_path
        :param close_window:
            If True, close window and do not show figure
            If false, show figure
        """
        # Get waveforms
        # waveforms = self.get_traces_raw(st-n_before, st+n_after+1).astype(float)
        # waveforms -= np.mean(waveforms, axis=1, keepdims=True)
        waveforms = self.get_traces_filt(st-n_before, st+n_after+1).astype(float)

        if chan_center is None:
            chan_center = np.argmin(waveforms[:, n_before])  # np.argmax(np.abs(waveforms[:, n_before]))

        # Create plotting figure
        if subplots is None:
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig, (a0, a1) = subplots

        # Plot max waveform
        a0.set_xlabel("Rel. time (frames, 20kHz)")
        a0.set_ylabel("Voltage (µV)")
        set_ticks((a0,), waveforms[chan_center, :], center_xticks=True)
        a0.set_xlim(0, n_before+n_after)
        a0.plot(waveforms[chan_center, :])

        # Plot waveform channels
        WINDOW_HALF_SIZE = 90  # length and width of window will be WINDOW_HALF_SIZE * 2
        SCALE_W = 0.25  # Multiply width of waveform by this to scale it down
        SCALE_H = 0.01 #*6  # Multiple height of waveform by this to scale it down

        a1.set_aspect("equal")
        a1.set_xticks([])
        a1.set_yticks([])

        # Each grid space is PITCHxPITCH in area
        a1.set_xlim(-100, 80)
        a1.set_ylim(-WINDOW_HALF_SIZE, WINDOW_HALF_SIZE)

        # Get channel waveforms to plot
        chans, dists = self.nearest_chan[chan_center]
        locs = self.get_channel_locations()
        locs[:, 0] *= 1.5

        loc_center = locs[chan_center]  # Location of channel with max amp

        max_dist = np.sqrt(2) * WINDOW_HALF_SIZE # Distance from center of window to corner

        # Plot each channel waveform
        for c in chans:
            # Offset the location of the waveform to make max amplitude channel (0, 0)
            loc = locs[c] - loc_center
            if np.sqrt(np.sum(np.square(loc))) >= max_dist:
                break

            wf = waveforms[c]
            x_values = (np.arange(wf.size) - n_before) * SCALE_W + loc[0]
            y_values = wf * SCALE_H + loc[1]
            a1.plot(x_values, y_values, c="red" if c == chan_center else "black")

            # For prop signal adding rms (see end of run_alg_decomp_2.ipynb)
            rms_buffer = 500
            prop_buffer = int(1.5*30)
            traces = self.get_traces_filt(st - rms_buffer, st + rms_buffer, c).flatten()
            rms = np.sqrt(np.mean(np.square(traces)))
            factor = -np.min(traces[rms_buffer-prop_buffer:rms_buffer+prop_buffer+1]) / rms
            a1.text(x_values[x_values.size // 2], y_values[y_values.size // 2], c, c="blue", size=15)
            a1.text(x_values[x_values.size // 2]+20, y_values[y_values.size // 2], f"{factor:.1f}", c="brown", size=13)

            # a1.scatter(*loc)

        if save_path is not None:
            plt.savefig(save_path)

        fig.suptitle(f"Rec: {self.name}, c: {chan_center}, st: {st}")

        if close_window:
            plt.close()

        if subplots is None:
            plt.show()

    @staticmethod
    def _get_traces(rec, start_frame=None, end_frame=None, channel_ind=None):
        # Helper function for get_traces_raw and get_traces_filt
        if channel_ind is None:
            channel_ids = rec.get_channel_ids()
        else:
            channel_ind = np.atleast_1d(channel_ind)
            channel_ids = rec.get_channel_ids()[channel_ind]
        traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_scaled=False)  # (n_frames, n_channels)
        return traces.T  # (n_channels, n_frames)


def get_channel_locations(recording):
    """
    Get channel map

    :param recording:

    :return: np.ndarray
        channel_map[chan_idx] = (x, y)
    """

    if isinstance(recording, Recording):
        return recording.get_channel_locations()
    elif isinstance(recording, str):
        if recording.endswith(".h5"):
            return MaxwellRecordingExtractor(recording).get_channel_locations()
        elif recording.endswith(".nwb"):
            return NwbRecordingExtractor(recording).get_channel_locations()
        else:
            raise ValueError(f'Only .h5 and .nwb recording formats are supported, not \'.{recording.split(".")[-1]}\'')
    else:
        raise ValueError("'recording' must be a Recording object or a str")
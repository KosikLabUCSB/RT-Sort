import numpy as np
import matplotlib as mpl  # For colorbar in Unit.plot()
import matplotlib.pyplot as plt
from pathlib import Path

from src.recording import Recording, get_channel_locations
from src import plot

class Unit:
    """Class to represent a detected unit from a spike sorter"""
    def __init__(self, idx: int, spike_train, channel_idx: int, recording: Recording=None):
        self.idx = idx
        self.spike_train = spike_train
        self.chan = int(channel_idx)
        self.recording = Recording(recording) if recording is not None else None

    def __len__(self):
        return len(self.spike_train)

    def plot_isi_dist(self, **hist_kwargs):
        isis = np.diff(self.spike_train)
        plot.hist(isis, **hist_kwargs)
        
        plt.title(f"ISI distribution for unit idx {self.idx}")
        
        plt.xlabel("ISI (ms)")
        
        xmax = None
        if "range" in hist_kwargs:
            xmax = hist_kwargs["range"][1]
        plt.xlim(0, xmax)
        
        plt.ylabel("Number of spikes")

    def get_isi_viol_f(self, isi_viol=1.5):
        """
        Get fraction of spikes that violate ISI

        :param isi_viol: Any consecutive spikes within isi_viol (ms) is considered an ISI violation
        """
        isis = np.diff(self.spike_train)
        violation_num = np.sum(isis < isi_viol)
        # violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        #
        # total_rate = num_spikes / total_duration
        # violation_rate = violation_num / violation_time
        # violation_rate_ratio = violation_rate / total_rate

        return violation_num / len(self.spike_train)

    def get_waveforms(self, spike_time_idx=None, ms_before=2, ms_after=2, chans=None) -> np.ndarray:
        """
        Get waveforms at each constituent electrode at self.spike_train[spike_time_idx]

        :param spike_time_idx:
            Which spike time to choose
            If None, pick a random spike time
        :param ms_before:
            How many ms before spike time to extract
        :param ms_after:
            How many ms after spike time to extract
        :param chans:
            Which channels to select
            If None, select all

        :return:
            np.ndarray with shape [num_electrodes, num_frames]
            num_frames = ms_before (in frames) + 1 (for spike time) + ms_after (in frames)
        """

        sf = self.recording.get_sampling_frequency()

        if spike_time_idx is None:
            spike_time_idx = np.random.choice(len(self.spike_train))
        if chans is None:
            chans = np.arange(self.recording.get_num_channels())
        
        spike_time = int(self.spike_train[spike_time_idx] * sf)        
        n_before = int(ms_before * sf)
        n_after = int(ms_after * sf)
        
        start_frame = spike_time - n_before
        end_frame = spike_time + n_after + 1
        wf = self.recording.get_traces_filt(max(0, start_frame), min(end_frame, self.recording.get_total_samples()), chans)
        if start_frame < 0:
            pad = np.zeros((len(chans), -start_frame))
            wf = np.concatenate([pad, wf], axis=1)
        elif end_frame > self.recording.get_total_samples():
            pad = np.zeros((len(chans), end_frame-self.recording.get_total_samples()))
            wf = np.concatenate([wf, pad], axis=1)
        return wf

    def get_templates(self, num_wfs=300, ms_before=2, ms_after=2):
        sf = self.recording.get_sampling_frequency()
        templates = np.zeros((self.recording.get_num_channels(), int(ms_before * sf)+int(ms_after * sf)+1), dtype=float)

        # templates_median = []
    
        if num_wfs is None or num_wfs >= len(self):
            spike_ind = range(len(self))
        else:
            spike_ind = np.random.choice(len(self), replace=False, size=num_wfs)
        
        for idx in spike_ind:
            waveforms = self.get_waveforms(spike_time_idx=idx, ms_before=ms_before, ms_after=ms_after)

            
            large_window = self.get_waveforms(spike_time_idx=idx, ms_before=33, ms_after=33)
            medians = np.median(np.abs(large_window), axis=1, keepdims=True) / 0.6745
            waveforms = waveforms / medians
            templates += waveforms
            # templates_median.append(waveforms)
        return templates / len(spike_ind)
        # print("Using templates median")
        # return np.median(templates_median, axis=0)
    
    def set_templates(self, num_wfs=300, ms_before=2, ms_after=2):
        self.templates = self.get_templates(num_wfs, ms_before, ms_after)

    def plot(self, num_wfs=300, ms_before=2, ms_after=2,
             chans_rms=None, add_lines=[],
             wf=None, time=None, sub=None,
             mea=True,# False,  
             save=False, 
             axis=None, fig=None,
             ylim=None, scale_h=None,
             window_half_size=120,# 120,
             wf_alpha=1,  
             wf_widths=1, wf_colors="black",
             min_c=None, max_c=None,
             colorbar_location="right", use_colorbar_label=True,  # "Latency (frames)"
             ): 
        """Plot waveforms at location of electrodes

        Args:
            num_wfs (int, optional): _description_. Defaults to 300.
            ms_before (int, optional): _description_. Defaults to 2.
            ms_after (int, optional): _description_. Defaults to 2.
            
            chans_rms (list, optional): 
                If None, do not plot
                Else, plot mean waveforms and chans_rms[c] = rms on channel c
            add_lines (list, optional):
                Should be a list of lists. Inner list has size (num_elecs,)
                For each inner lists, plot horizontal lines the same way as would be for chans_rms
                (Used for plotting DL detections so that there is one line for 10% and one line for 17.5%)
            
            wf (np.array, optional):
                If not None, plot this wf
            time (float, None, optional):
                If None, plot templates
                Else, plot at time (ms) 
            sub (np.ndarray, None, optional):
                If not None, subtract sub from wfs
            mea (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            
            axis (optional): 
                If not None, create axis
            fig (optional):
                If need colorbar, fig cannot be None
            
            return_steup (bool, optional): If True, return the following as kwargs:
            scale_h (float, optional): If not None, scale height of waveforms so waveforms on different electrodes won't overlap
            

            wf_widths: float, list
                If float, all waveforms given width of wf_widths
                
                If list, should be list of size (num_elecs,)
                num_elecs[c] gives thickness of waveform on elec c
            wf_colors:
                Same as wf_widths but for color
                
                If color is str, should be name ("green") or hex value
                Otherwise, should be float used for colormap
        """
        
        if chans_rms is not None:
            if wf is not None:
                wfs = wf
            elif time is None:
                if not hasattr(self, "templates"):
                    wfs = self.get_templates(num_wfs=num_wfs, ms_before=ms_before, ms_after=ms_after)
                else:
                    wfs = self.templates  # For coactivations.ipynb
            else:
                sf = self.recording.get_sampling_frequency()
                wfs = self.recording.get_traces_filt(
                    start_frame=round((time-ms_before)*sf), 
                    end_frame=round((time+ms_before)*sf+1)
                )
        
        if sub is not None:
            wfs = wfs - sub
        
        # Set self.chan
        if self.chan == -1:
            self.chan = np.argmin(np.min(wfs, axis=1))
        
        # Setup wf_widths
        try:
            wf_widths[0]
        except TypeError:
            wf_widths = [wf_widths] * wfs.shape[0]
        
        # Plot parameters        
        
        electrode_size = 20
        electrode_color = "#888888"
        
        xlabel = "x (µm)"
        ylabel = "y (µm)"

        if mea:
            # Plot parameters for MEA
            xlim = None
            window_half_size = 75  # 75 for 200724/2602/cell7 
            
            if scale_h is None and chans_rms is not None:
                max_val = np.max(np.abs(wfs[max(0, self.chan-5):self.chan+5]))
                max_val = max(max_val, chans_rms[self.chan] * 5)  # max_val before this could be below 5RMS, so red line for 5RMS could be too high
                scale_h = 15 / max_val  # 17.5 is pitch of electrode

        else:
            # Plot parameters for neuroxpixels
            xlim = (2, 68)  # for SI rec
            
            # xlim = (-35, 35)  # for SI ground truth rec
            # window_half_size = 110  # for SI ground truth rec
            
            if scale_h is None and chans_rms is not None:
                scale_h = 20 / np.max(np.abs(wfs[max(0, self.chan-5):self.chan+5]))  # 20 is dist between two rows of electrodes

        # scale_h *= 0.8  # For MEA below 5SNR units
        # scale_h *= 0.7  # For SI ground-truth weird-looking units

        # cmap_latency = plt.get_cmap(cmap_latency, max_latency)

        # Find which channels to plot
        chan_center = self.chan
        locs = self.recording.get_channel_locations().astype("float32")
        loc_center = locs[chan_center]

        chans, dists = self.recording.nearest_chan[chan_center]
                    
        # Create axis
        if axis is None:
            fig, axis = plt.subplots(1)
            
        # Setup axis
        axis.set_aspect("equal")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        if ylim is None:
            ymin = max(0, loc_center[1] - window_half_size)
            # ymin = max(-1950, loc_center[1] - window_half_size)  # SI simulated ground truth
            ymax = ymin + window_half_size * 2
        else:
            ymin, ymax = ylim
        axis.set_ylim(ymin, ymax)
        # axis.set_ylim(ymin-12, ymax)  # SI simulated ground truth weird-looking units have such low SNR that need extra space at bottom for 5SNR line

        if xlim is None:
            xmin = max(0, loc_center[0] - window_half_size)
            xmax = xmin + window_half_size * 2
        else:
            xmin, xmax = xlim
        axis.set_xlim(xmin, xmax)

        chans = [c for c in chans if xmin <= locs[c, 0] <= xmax and ymin <= locs[c, 1] <= ymax]
        # Setup wf_colors:
        try:
            if len(wf_colors) != wfs.shape[0]:  # wf_colors could be a str (e.g. "black")
                wf_colors = [wf_colors] * wfs.shape[0]
        except TypeError:
            wf_colors = [wf_colors] * wfs.shape[0]
        if not isinstance(wf_colors[0], str): # wf_colors are floats for color map
            cmap = plt.cm.ocean
            # Only normalize colormap based on elecs used in plot
            wf_colors = np.array(wf_colors)
            
            # Adaptive color bar. -1 and +1 for when plotting individual spikes, values above/below max/min seq's latency is at ends of colorbar
            if min_c is None:
                min_c = round(np.floor(np.min(wf_colors[chans]))) - 1
            if max_c is None:
                max_c = round(np.ceil(np.max(wf_colors[chans]))) + 1
            # min_c = max(-10, min_c)
            # max_c = min(max_c, 10)
            
            num_levels = (max_c - min_c + 1)
            cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 0.9, num_levels)))  # Only use up to 0.9 since ocean is white at end
            
            bounds = np.linspace(min_c, max_c, num_levels+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            wf_colors = [cmap(norm(c)) for c in wf_colors]
            
            # region Figuring out how to use color map and color bar
            # fig, ax = plt.subplots(figsize=(6, 1))

            # min_c = -3
            # max_c = 3
            # num_levels = (max_c - min_c) * 2
            # cmap = mpl.colors.ListedColormap(plt.cm.gist_rainbow(np.linspace(0, 1, num_levels)))

            # bounds = np.linspace(min_c, max_c, num_levels+1) # range(min_c, max_c+1)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # colorbar = fig.colorbar(
            #     mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            #     cax=ax, orientation='horizontal',
            # )
            # colorbar.set_ticks(bounds)

            # plt.show()

            # # Test values
            # for v in np.arange(min_c, max_c+1):
            #     plt.axvline(v, color=cmap(norm(v)))
            #     v += 0.5
            #     plt.axvline(v, color=cmap(norm(v)))
            # plt.show()
            # endregion
        else:
            cmap = min_c = max_c = None

        # Plot each channel waveform
        for c in chans:
            loc = locs[c]

            # Check if channel is out of bounds
            # if mea:
            #     if np.sqrt(np.sum(np.square(loc - loc_center))) >= max_dist:
            #         break
            # if not (xlim[0] <= loc[0] <= xlim[1] and ymin <= loc[1] <= ymax):
            #     continue

            if chans_rms is None:
                axis.scatter(*loc, marker="s", color=electrode_color, s=electrode_size) # Mark electrodes with square
            else:
                # Plot waveform
                wf = wfs[c]  # shape is (num_channels, num_waveforms, num_frames)
                wf = np.atleast_2d(wf)
                for w in wf:
                    x_values = np.linspace(loc[0]-7, loc[0]+7, w.size)
                    y_values = w * scale_h + loc[1]
                    
                    # Plot 5RMS
                    if isinstance(chans_rms, np.ndarray):
                        # Horizontal line indicating 5RMS
                        rms_scaled = 5 * (-chans_rms[c] * scale_h) + loc[1]  # 5 for 5RMS
                        axis.plot(x_values, [rms_scaled] * len(x_values),
                                    linestyle="dashed",
                                    c="red", alpha=0.6, zorder=5)
                        # Vertical line connecting horizontal line to electrode position
                        axis.plot([loc[0]] * 10, np.linspace(rms_scaled, loc[1], 10),
                                    linestyle="dashed",
                                    c="red", alpha=0.3, zorder=5)
                    # Plot additional lines
                    for lines in add_lines:
                        y_value = lines[c] * scale_h + loc[1]
                        axis.plot(x_values, [y_value] * len(x_values),
                                  linestyle="dashed",
                                  c="red", alpha=0.6, zorder=5
                                  )
                    
                    # # If cross chans_rms, make black instead of gray
                    # wf_alpha_elec = wf_alpha
                    # wf_alpha_elec -= 0.4 * (np.max(np.abs(w)) < np.abs(5*chans_rms[c]))  # Make wfs less than 5RMS gray
                    # wf_alpha_elec = max(0.1, wf_alpha_elec)
                    
                    # If cross chans_rms, make line thicker
                    
                    
                # axis.plot(x_values, y_values, c="red" if c == chan_center else "black")
                # axis.plot(x_values, y_values, color=wf_color, alpha=wf_alpha_elec, zorder=15)  # If cross chans_rms, make black instead of gray
                axis.plot(x_values, y_values, c=wf_colors[c], alpha=wf_alpha, linewidth=wf_widths[c], zorder=15)  # If cross chans_rms, make line thicker

        if fig is not None and cmap is not None:
            colorbar = fig.colorbar(
                mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                ax=axis,
                location=colorbar_location, 
                label="" if not use_colorbar_label else "Latency (frames)"
            )
            if use_colorbar_label:
                ticklabels = range(min_c+1, max_c)  # colorbar.set_ticklabels(range(min_c, max_c+1))
                ticklabels = [f"≤{min_c}"] + list(ticklabels) + [f"≥{max_c}"]
                colorbar.set_ticks(bounds[:-1] + np.diff(bounds)/2)
                colorbar.set_ticklabels(ticklabels)
            else:
                colorbar.set_ticks([])
                colorbar.set_ticklabels([])
            
        axis.set_title(f"Index: {self.idx}, #spikes: {len(self.spike_train)}")

        if save:
            plt.savefig(save)

        return {
            "chans_rms": chans_rms,
            "add_lines": add_lines,
            "ylim": (ymin, ymax), "scale_h": scale_h,
            "ms_before": ms_before,
            "ms_after": ms_after,
            "window_half_size": window_half_size,
            "min_c": min_c,
            "max_c": max_c
            }

class SpikeSorter:
    """Class to represent any spike sorter"""
    def __init__(self, recording, name, spike_times=None, units=None):
        """
        Set spike_times or units or neither
        """
        
        self.name = name
        self.iter_i = -1
        # self.set_unit_ids(range(len(self)))

        if recording is not None:
            if isinstance(recording, Recording):
                self.recording = recording
            else:
                self.recording = Recording(recording)

        # Store units
        self.units = units
        if units is not None:
            self.spike_times = [unit.spike_train for unit in units]
        else:
            self.spike_times = spike_times

    def set_unit_ids(self, unit_ids):
        self.unit_ids = list(unit_ids)

    def get_spike_times(self):
        return self.spike_times

    def __iter__(self):
        # Loop through each unit
        self.iter_i = -1
        return self

    def __next__(self):
        # Get next unit
        self.iter_i += 1
        if self.iter_i == len(self):
            raise StopIteration

        return self[self.iter_i]

    def __len__(self):
        # Number of units
        return len(self.get_spike_times())

    def __getitem__(self, idx):
        # Get a specific unit
        if self.units is not None:
            return self.units[idx]
        
        return Unit(
            idx,
            self.get_spike_times()[idx],
            channel_idx=-1,
            recording=self.recording
        )  # type: Unit

    def uid_to_idx(self, id):
        return self.unit_ids.index(id)

    def get_unit(self, id) -> Unit:
        return self[self.uid_to_idx(id)]

    def plot_isis(self, isi_viol=1.5, **hist_kwargs):
        """
        Plot interspike-interval distribution

        :param isi_viol:
            Any consecutive spikes within isi_viol (ms) is considered an ISI violation
        :param hist_kwargs:
            kwargs for plt.hist

        :return:
            % isi violation for each propagation in propagating_times
        """

        # violations = [unit.get_isi_viol_f(isi_viol) * 100 for unit in self]

        violations = []
        for st in self.get_spike_times():
            isis = np.diff(st)
            violation_num = np.sum(isis < isi_viol)
            violations.append(violation_num / len(st) * 100)

        plt.hist(violations, **hist_kwargs)
        plt.title("ISI violation distribution.")
        plt.xlabel("Percent ISI violations")
        plt.ylabel("Count")
        plt.xlim(0)
        plt.show()

        return violations

    def plot_nums_spikes(self, show=True, **hist_kwargs):
        """
        Plot a histogram showing the number of spikes on each unit
        """
        nums_spikes = [len(unit) for unit in self.get_spike_times()]
        # plt.hist(nums_spikes, **hist_kwargs)
        # plt.title("Number of spikes distribution.")
        # plt.xlabel("Number of spikes")
        # plt.ylabel("Count")
        # plt.xlim(0)
        # if show:
        #     plt.show()
        # print(f"Mean: {np.mean(nums_spikes):.2f}")
        # print(f"STD: {np.std(nums_spikes):.2f}")
        
        plt.title(f"{len(nums_spikes)} units")
        plt.hist(nums_spikes, bins=40, range=(0, 11000))
        plt.xlim(0, 11000)
        plt.ylim(0, 50)
        plt.xlabel("#spikes")
        plt.ylabel("#units")
        plt.show()

    def plot_activity_map(self,
                          size_scale=250, spike_alpha=0.9,
                          elec_color="black",
                          xlabel="x (µm)", ylabel="y (µm)"):
        """
        For each propagation, plot a dot that represents the number of spikes from that
        propagation. Location is location of first electrode. Size is number of spikes
        when crossings from all electrodes in propagation are included

        :param recording:

        :param size_scale:
            Size of dot = num_spikes / max_num_spikes * size_scale
        :param spike_alpha
            Alpha of dot
        :param elec_color:
            Color of electrodes
        :param xlabel:
            x-label of plot
        :param ylabel:
            y-label of plot
            y (µm)
        """

        channel_locs = get_channel_locations(self.recording)
        plt.scatter(channel_locs[:, 0], channel_locs[:, 1], s=1, c=elec_color, marker=",", zorder=-100)

        x_values = []
        y_values = []
        nums_spikes = []
        for unit in self:
            # for elec_id in prop.ID:
            x, y = channel_locs[unit.chan]
            x_values.append(x)
            y_values.append(y)
            nums_spikes.append(len(unit))

        num_spikes_max = max(nums_spikes)
        for x, y, size in zip(x_values, y_values, nums_spikes):
            plt.scatter(x, y, s=size / num_spikes_max * size_scale, alpha=spike_alpha)

        plt.title("Spike Map")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
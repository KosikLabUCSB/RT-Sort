from pandas import DataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.sorters.base import *


class PropUnit(Unit):
    """Class to represent a detected unit from prop signal"""
    def __init__(self, prop_dataframes: list, idx: int, spike_train, recording: Recording):
        if isinstance(prop_dataframes, DataFrame):
            prop_dataframes = [prop_dataframes]
        
        super().__init__(id, spike_train, prop_dataframes[0].ID[0], recording)
        self.df = prop_dataframes
        self.chans = np.unique(np.concatenate([df.ID.values for df in self.df]))
        
        self.idx = idx
        self.elec_id = prop_dataframes[0].ID[0]
        
        self.plot_title = "Elec ID: " + " ".join(str(df.ID[0]) for df in self.df)

    def plot(self, **kwargs):
        """Plot waveforms at location of electrodes

        Args:
            num_wfs (int, optional): _description_. Defaults to 300.
            ms_before (int, optional): _description_. Defaults to 2.
            ms_after (int, optional): _description_. Defaults to 2.
            chans_rms (list, optional): 
                If None, do not plot
                Else, plot mean waveforms and chans_rms[c] = rms on channel c
            time (float, None, optional):
                If None, plot templates
                Else, plot at time (ms) 
            sub (np.ndarray, None, optional):
                If not None, subtract sub from wfs
            mea (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            subplot (optional): If not None, should be (fig, axis)
            return_steup (bool, optional): If True, return the following as kwargs:
            vmax (float, optional): Maximum value in colorbar
            scale_h (float, optional): If not None, scale height of waveforms so waveforms on different electrodes won't overlap
        """
        
        if "subplot" not in kwargs:
            fig, axis = plt.subplots(1)
        else:
            fig, axis = kwargs["subplot"]
        
        cmap_latency = "gist_rainbow"
        
        chans_rms = kwargs.get("chans_rms", None)
        
        if chans_rms is None:
            marker_size = 200
        else:
            marker_size = 50
        marker_alpha = 1  # 0.6
        merge_dec = 4
        markers = "oXD*pPv^<>+d"
        
        locs = self.recording.get_channel_locations()
        
        vmax = kwargs.get("vmax", None)
        
        base_kwargs = {k:v for k, v in kwargs.items() if k not in {"subplot"}}
        return_kwargs = super().plot(axis=axis, **base_kwargs)

        latencies = set()
        for prop in self.df:
            latencies.update(prop.latency)
        latencies = list(latencies)
        if vmax is not None:
            if not np.any(np.abs(np.array(latencies) - vmax) < 1e-9):  # Prevent adding same value twice
                latencies.append(vmax)
        vmin = min(0, *latencies)

        # Mark prop electrodes 
        for i, prop in enumerate(self.df):
            locs_prop = locs[prop.ID.astype(int)].copy()
            if chans_rms is not None:  # Offset marker so not on top of waveform
                locs_prop -= merge_dec
            locs_prop[:, 1] -= merge_dec * i
            sizes = prop.cooccurrences.values / np.max(prop.cooccurrences.values) * marker_size
            axis.scatter(locs_prop[1:, 0], locs_prop[1:, 1], c=prop.latency.values[1:], s=sizes[1:],
                                   zorder=10, marker=markers[i], cmap=cmap_latency, vmin=vmin, vmax=vmax,
                                   alpha=marker_alpha)
            axis.scatter(*locs_prop[0, :], color="black", s=sizes[0], zorder=9, marker=markers[i], alpha=marker_alpha)
        scatter = axis.scatter([-1000]*len(latencies), [-1000]*len(latencies), c=latencies, cmap=cmap_latency, vmin=vmin)  # For creating colorbar

        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(scatter, cax=cax) #, label="Latency (ms)")
            
        colorbar.set_ticks(latencies)
        colorbar.set_ticklabels([round(l, 2) for l in latencies])       
        
        axis.set_title(self.plot_title)
        
        return return_kwargs

    def plot_templates(self, num_wfs=300, ms_before=2, ms_after=2, dec=50, vmax=2/30):
        # dec is vertical decrement of waveforms
        # vmax is max value for latency colormap
        
        CMAP = plt.get_cmap("gist_rainbow")
                
        fig, ax = plt.subplots(1)
        
        chan_locs = self.recording.get_channel_locations()
        
        templates = self.get_templates(num_wfs, ms_before, ms_after)        
        x_values = (np.arange(templates.shape[1]) - templates.shape[1] // 2) / self.recording.get_sampling_frequency()
        for i, chan in enumerate(self.chans):
            temp = templates[chan, :] - dec * i
            
            color = CMAP(self.df[0].latency[i]/vmax) if i > 0 else "black"
            
            ax.plot(x_values, temp, alpha=1, label=f"({chan_locs[chan, 0]}, {chan_locs[chan, 1]})", 
                    c=color
                    )
            
        ax.axvline(0, color="black", linestyle="dashed", alpha=0.5)
            
        ax.set_yticks([])
            
        ax.set_xlim(x_values[0], x_values[-1])
        ax.set_xlabel("Time rel. trough of first electrode (ms)")
        
        ax.set_title("Elec ID: " + str(self.df[0].ID[0]))
        
        # plt.legend(loc="lower left")
        plt.show()
      
    def copy(self):
        """Return a copy of self"""
        return PropUnit(self.df, self.idx, self.spike_train, self.recording)
        

class PropSignal(SpikeSorter):
    """Class to represent results from propagation signal algorithm"""

    PROPAGATIONS_SAVE_NAME = "propagations.npy"
    PROPAGATING_TIMES_SAVE_NAME = "propagating_times.npy"

    def __init__(self, data, recording, name: str = "Prop."):
        """
        :param data: (Path or str) or (list or tuple)
            If Path or str: Path to folder containing "propagations.npy" and "propagating_times.npy"
            If list or tuple: must contain (propagations, propagating_times)
        :param recording
        :param name
        """
        if isinstance(data, Path) or isinstance(data, str):
            self.path = Path(data)
            self.props = np.load(str(self.path / self.PROPAGATIONS_SAVE_NAME), allow_pickle=True)
            self.props_times = np.load(str(self.path / self.PROPAGATING_TIMES_SAVE_NAME), allow_pickle=True)
        else:
            self.props, self.props_times = data
            if isinstance(self.props[0], DataFrame):  # Reformat props
                self.props = [[p] for p in self.props]

        super().__init__(recording, name)

    def get_unit_spike_train(self, idx):
        # Get spike train of a single unit
        prop_times = self.props_times[idx]
        elec_idx = -1 if self.elec_idx >= len(prop_times) else self.elec_idx
        return prop_times[elec_idx]

    def get_spike_times(self):
        """
        Get spike times of the propagations.

        :return: np.array
            ith element contains spike times of ith prop in self.props
        """

        # spike_times = []
        # for i in range(len(self)):
        #     spike_times.append(self.get_unit_spike_train(i))
        # return spike_times
        return self.props_times

    def __len__(self):
        # Get number of props
        return len(self.props)

    def __getitem__(self, idx):
        # Get a unit from index
        # prop = self.props[idx][0]
        # # chan = prop.ID[0]
        # chan = round(np.mean([p.ID[0] for p in self.props[idx]]))
        # spike_train = self.get_spike_times()[idx]
        # # return PropUnit(self.unit_ids[idx], spike_train, chan, self.recording, prop)
        # return PropUnit(idx, spike_train, chan, self.recording, prop)
        prop = self.props[idx]
        spike_train = self.props_times[idx]
        return PropUnit(prop, idx, spike_train, self.recording)

    def plot_props_all(self, max_plots=None, **plot_prop_kwargs):
        """
        Plot a propagation figure for each propagation. Relative size of dots in figure
        represent number of small window cooccurrences with first electrode.
        Color represents latency with first electrode

        :param max_plots:
            If max_plots is not None, plot at most max_plots
        """

        if max_plots is not None and max_plots < len(self):
            plot_ind = np.random.choice(len(self), max_plots, replace=False)
        else:
            plot_ind = range(len(self))
        for i in plot_ind:
            self[i].plot_prop(**plot_prop_kwargs)

    def get_unit(self, elec_id):
        # Get a unit by the ID of the starting electrode in first prop in the merge
        for idx in range(len(self)):
            unit = self[idx]
            if unit.df[0].ID[0] == elec_id:
                return unit
        return None
    
    def get_idx(self, unit):
        # Get index of unit in self.props 
        for idx in range(len(self)):
            if self.props[idx] is unit.df:
                return idx

    @staticmethod
    def get_sequence_duration(prop):
        # Get sequence duration of prop, i.e. latency between first and last electrode
        return prop.latency.values[-1]

    def save(self, path, notes=""):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        np.save(path / self.PROPAGATIONS_SAVE_NAME, self.props)
        np.save(path / self.PROPAGATING_TIMES_SAVE_NAME, self.props_times)

        with open(path / "notes.txt", "w") as txt:
            txt.write(notes)

    # @staticmethod
    # def run(save_path, version, spike_times, **prop_signal_kwargs):
    #     if version == 1:
    #         propagations, propagating_times = prop_signal.v1.prop_signal(spike_times, **prop_signal_kwargs)
    #     elif version == 2:
    #         propagations, propagating_times = prop_signal.v2.prop_signal(spike_times, **prop_signal_kwargs)
    #     else:
    #         raise ValueError(f"version '{version}' is not a valid version")
    #
    #     path_root = Path(save_path)
    #     i = 0
    #     while path_root.exists():
    #         path_root = path_root.parent / (path_root.name + " (" + str(i) + ")")
    #         i += 1
    #     path_root.mkdir(parents=True)
    #     np.save(str(path_root / "propagations.npy"), np.array(propagations, dtype=object))
    #     np.save(str(path_root / "propagating_times.npy"), np.array(propagating_times, dtype=object))
    #     with open(path_root / "params.json", "w") as f:
    #         json.dump(prop_signal_kwargs, f)
    #
    #     return PropSignal(save_path, elec_idx=-1)

    @staticmethod
    def format_thresh_crossings(thresh_crossings):
        # Format thresh_crossings from np.load so they are numpy arrays instead of lists
        return [np.asarray(st) for st in thresh_crossings]

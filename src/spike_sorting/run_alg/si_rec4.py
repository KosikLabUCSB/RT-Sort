"""
Functions and classes for si_rec3.ipynb
"""

from math import ceil
from multiprocessing import Pool
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import HTML  # For displaying pandas dataframe without index

from tqdm import tqdm

from src import utils
from src.recording import Recording
from src.sorters.base import Unit
from src.sorters.prop_signal import PropUnit
from src.comparison import Comparison

# region Form Propagations
"""
Pseudocode: 
1. For each pair of electrodes (A and B, A is reference, B is target):
    a. Find all 5RMS threshold crossings that are detected on A and B within 0.5ms of each other
    b. Create array of shape (n_thresh_crossings_found_in_a, 2) where first column is time delay 
        between cooccurrence on B relative to A. Second column is amp_B/amp_A
    c. Fit Gaussian Mixture Model to cluster spikes into unique sequence pairs
    d. For each unique sequence pairs, repeat step 1. Treat each sequence pair as a single electrodes.
        On this pair, a threshold crossing counts as 5RMS on one electrode and max(mean amp +- STD, 4)
        on the other. After multiple iterations where pair contains more than 2 electrodes, one elec
        has to be greater than 5RMS. 
"""

def plot_gmm(gmm, x, outlier_threshold=-10):
    """
    Plot GaussianMixture model on 2d plane
    
    Params:
        outlier_threshold: not implemented
            If log likelihood of point is less than outlier_threshold, 
            plotted as outlier (black dot)
    """
    # outliers = gmm.score_samples(x) < outlier_threshold
    # plt.scatter(x[outliers, 0], x[outliers, 1], color="black", alpha=0.1)
    # x = x[~outlier_threshold]
    
    for i in range(len(gmm.means_)):   
        # Plot individual points
        crossings = x[gmm.predict(x) == i]
        print(len(crossings))
        plt.scatter(crossings[:, 0], crossings[:, 1], color=f"C{i}", alpha=0.05)

        # Plot center
        plt.scatter(*gmm.means_[i], color="black")
    
    # print("Using xlim")
    # plt.xlim(-0.2, 0.2)

def test_mean_std_overlap(mean1, std1,
                          mean2, std2, 
                          x_std=1):
    """
    Return true if mean1+-std1*x_std overlaps with mean2+-std2*x_std
    """
    
    range1_low = mean1 - std1*x_std
    range1_high = mean1 + std1*x_std

    range2_low = mean2 - std2*x_std
    range2_high = mean2 + std2*x_std

    if (range2_low <= range1_low <= range2_high) or (range2_low <= range1_high <= range2_high):
        return True  # Ranges overlap
    elif (range1_low <= range2_low <= range1_high) or (range1_low <= range2_high <= range1_high):
        return True  # Ranges overlap
    else:
        return False  # Ranges do not overlap


class ElecCluster:
    """Represents a cluster of electrodes with mean/std latency and rel amps"""
    def __init__(self, df, gmm=None, spike_train=None):
        """
        This is used in from_pair and add and merge
        """
        self.df = self.format_df(df)
        self.gmms = gmm  # One gmm for each new electrode added

        self.elecs = self.df.idx.values  # List of electrode ind
        self.elecs_stats = {} # {elec_idx: data} Quick access for each electrode latency and rel amp stats
        for i, row in self.df.iterrows():
            if i == 0:  # Don't include first elec
                continue
            self.elecs_stats[int(row["idx"])] = row.iloc[1:]

        self.spike_train = spike_train

    @staticmethod
    def from_pair(elec_pair, 
                  latency, std_latency,
                  rel_amp, std_rel_amp,
                  gmm=None,
                  spike_train=None):
        """
        Create ElecCluster object from a pair of electrodes
        
        Parameters:
            elec_pair:
                Indices of two elecs that initially started cluster
            latency, std_latency, rel_amp, std_rel_amp:
                Refer to the stats of one electrode in pair compared to other
            gmm: optional
                The scipy Gaussian Mixture Model that resulted in cluster being formed
        """
        
        df = pd.DataFrame(
            [
                [elec_pair[0], 0, 0, 1, 0],
                [elec_pair[1], latency, std_latency, rel_amp, std_rel_amp]
            ],
            columns=["idx", "latency", "std_latency", "rel_amp", "std_rel_amp"]
        )
        return ElecCluster(df, [gmm], spike_train)

    def add(self, elec_idx,
            latency, std_latency,
            rel_amp, std_rel_amp,
            gmm=None):
        """
        Create new ElecCluster object with the electrode described in attributes added to cluster
        
        This returns instead of modifies because the original ElecCluster needs to be kept 
        in case this root has branches (multiple neurons with similar AIS locs)
        """
        new_elec = {"idx": elec_idx, "latency": latency, "std_latency": std_latency, "rel_amp": rel_amp, "std_rel_amp": std_rel_amp}
        df = self.df.append(new_elec, ignore_index=True)
        gmms = self.gmms + [gmm]
        return ElecCluster(df, gmms, self.spike_train)
    
    def merge(self, other):
        """
        Create new ElecCluster object by merging with elec_cluster

        """
        # gmms = other.gmms + self.gmms
        # Merge electrodes (electrode ids in dataframe will be sorted)
        # df = []
        # i = j = 0
        # while i < len(self.elecs) and j < len(other.elecs):
        #     if self.elecs[i] < other.elecs[j]:
        #         df.append(self.df.iloc[i])
        #         i += 1
        #     else:
        #         df.append(other.df.iloc[j])
        #         j += 1
        # df.extend(self.df.iloc[i:])
        # df.extend(other.df.iloc[j:])
        # # df = pd.DataFrame(df)
        # return df
        # return ElecCluster(df, gmms, self.spike_train)
                
        self_elecs = set(self.elecs)
        add_elecs = [i for i, row in other.df.iterrows() if row.idx not in self_elecs]
        if len(add_elecs) > 0:
            df = pd.concat((self.df, other.df.iloc[add_elecs]), axis=0)
        else:
            df = self.df
        return ElecCluster(df, self.gmms + other.gmms, self.spike_train)        
        
    def set_footprint_stats(self, 
                            num_spikes=500,
                            ms_before=0.5, ms_after=0.5):
        """
        Extract the mean+-STD latency and relative amplitude of each electrode
        relative to ref_elec and set self.footprint_stats_ to
        [(mean_latency, std_latency, mean_rel_amp, std_rel_amp)] ith element is for elec with index i
        
        Parameters
        ----------
        num_spikes: int or None
            #spikes to sample from spike_train to extract stats
        ms_before, ms_after: float
            For extracting windows for each spike: on each electrode,
            extract [ms_before, ms_after] around spike on self's single root elec
        """
        
        if num_spikes is not None and num_spikes > len(spike_train):
            spike_train = np.random.choice(spike_train, size=num_spikes, replace=False)
    
        sf = recording.get_sampling_frequency()
        for spike in spike_train:
            frame = round(spike * sf)

        
        
    @staticmethod
    def format_df(df):
        """
        Format dataframe to have correct datatypes
        """
        return df.astype({"idx": int, "latency": float, "std_latency": float, "rel_amp": float, "std_rel_amp": float})


# class Propagation(Unit):
#     def __init__(self, idx: int, spike_train, channel_idx: int, recording: Recording = None,
#                  footprint_stats=None, elecs=None):
#         super().__init__(idx, spike_train, channel_idx, recording)
        
#         # roots 
#         self.footprint_stats = footprint_stats
#         self.elecs = elecs
        
    # def 
        
        
# endregion

# region Split spikes
class MultiDetection:
    """Represents a spike detected by multiple units"""
    def __init__(self, time, unit,
                 cross_times, cross_amps,
                 chans_rms, rms_thresh,
                 ms_before, ms_after,
                 min_coactivations_n, min_coactivations_p) -> None:
        """
        unit:
            First unit to be added to detection (order does not matter)
        """
        self.times = []
        self.units = []
        
        self.add_time(time)
        self.add_unit(unit)
        
        self.history = []  # after self.split, self.history = [(unit_detecting_spike, sub_amps)]
        """
        After self.split(), self.history will be filled with one element per unit that detected spike
        
        Adding data to self.history is slow. Comment this line of code for speed
        """
        
        # Store values now since MultiDetection is created by :func split_spikes:, so not have to pass these values in for 
        # MultiDetection's methods
        self.cross_times = cross_times
        self.cross_amps = cross_amps
        self.chans_rms = chans_rms
        self.rms_thresh = rms_thresh
        self.ms_before = ms_before
        self.ms_after = ms_after
        self.min_coactivations_n = min_coactivations_n
        self.min_coactivations_p = min_coactivations_p

    def __len__(self):
        return len(self.times)
        
    def get_duration(self):
        return self.times[-1] - self.times[0]
        
    def add(self, time, unit):
        self.add_time(time)
        self.add_unit(unit)
        
    def add_time(self, time):
        self.times.append(time)
        
    def add_unit(self, unit):
        self.units.append(unit)
        
    def split(self, record_history=False):             
        """
        Pseudocode:
        Keep track of local copy of {(elec, st): mean subtraction} and apply to loop1
        Loop:
            1. Find which units have enough coactivations to detect spike
                If no units, break loop
            2. Find which unit has largest amplitude
                Assign spike to this unit
        Update units' spike trains
        
        :param record_history:
            Whether to keep track of history for self.plot_splitting
            True: much slower, but needed for self.plot_splitting
        """
        
        cross_times = self.cross_times
        cross_amps = self.cross_amps
        chans_rms = self.chans_rms
        rms_thresh = self.rms_thresh
        ms_before = self.ms_before
        ms_after = self.ms_after
        min_coactivations_n = self.min_coactivations_n
        min_coactivations_p = self.min_coactivations_p
        
        class UnitData:
            """Store data about a unit"""
            def __init__(self, unit, time, amp, num_coacs) -> None:
                self.unit = unit
                self.time = time
                self.amp = amp
                self.num_coacs = num_coacs
                
            def unravel(self):
                return self.unit, self.time, self.amp
        
        self.history = []
        tracked_elecs = set()  # All electrodes that exist in at least 1 propagation (only need to keep track of these electrodes)
        for unit in self.units:
            tracked_elecs.update(unit.elecs)
        
        sub_amps = {}  # {(elec, cross_idx): mean amplitude to subtract from thresh crossing}      
        units_detected = set()  
        while True:
            max_unit = None  # UnitData with largest amplitude
                          
            # Loop through each unit          
            for time, unit in zip(self.times, self.units):
                if unit in units_detected:
                    continue
                    
                # Count coactivations
                num_coacs = 0
                for e, elec in enumerate(unit.elecs):
                    elec_cross_times = cross_times[elec]
                    elec_cross_amps = cross_amps[elec]
                    search_time = time if e == 0 else time-ms_before  # If first elec, time is in elec_cross_times, find exact idx
                    cross_idx = np.searchsorted(elec_cross_times, search_time, side="left")
                    # For each electrode, check if 
                    #   1) Thresh crossing exists 
                    #   2) Thresh crossing within ms_before and ms_after of unit's spike time
                    #   3) Thresh crossing >= X*RMS                    
                    if cross_idx < len(elec_cross_times) and \
                       elec_cross_times[cross_idx] <= time + ms_after and \
                       elec_cross_amps[cross_idx] - sub_amps.get((elec, cross_idx), 0) >= chans_rms[elec] * rms_thresh:
                           num_coacs += 1
                           
                           if e == 0:
                               time_idx = cross_idx  # For creating UnitData if prop detects spike
                           
                    elif elec == unit.elecs[0]:
                        # If first elec does not detect spike, then sequence does not detect spike (num_coacs already set to 0)
                            break
                        
                # Check if seq could have detected spike
                coacs_min = max(min_coactivations_n, ceil(min_coactivations_p/100 * len(self)))
                if num_coacs >= coacs_min:
                    amp = cross_amps[unit.elecs[0]][time_idx]
                    # Check if unit detecting spike has largest amplitude
                    if max_unit is None or (amp > max_unit.amp and num_coacs > max_unit.num_coacs):
                        max_unit = UnitData(unit, time, amp, num_coacs)

            if max_unit is None:
                break
            
            units_detected.add(max_unit.unit)
            
            # Find scaling factor for mean amplitude subtraction
            
            unit, time, amp = max_unit.unravel()
            first_elec = unit.elecs[0]
            scale_sub = amp / np.abs(unit.mean_amps_all[first_elec])
                        
            # Store mean amplitude subtraction
            for elec in tracked_elecs: 
            # for elec in range(len(cross_times)): 
                elec_cross_times = cross_times[elec]
                elec_cross_amps = cross_amps[elec]
                cross_idx = np.searchsorted(elec_cross_times, time-ms_before)
                # Find closest cross time that crosses RMS_THRESH and is within bounds of causing detection                
                while cross_idx < len(elec_cross_times) and \
                      elec_cross_times[cross_idx] <= time + ms_after and \
                      elec_cross_amps[cross_idx] - sub_amps.get((elec, cross_idx), 0) < chans_rms[elec] * rms_thresh:
                        cross_idx += 1
                        
                # Now store data
                key, value = (elec, cross_idx), np.abs(unit.mean_amps_all[elec]) * scale_sub
                if key not in sub_amps:
                    sub_amps[key] = value
                else:
                    sub_amps[key] += value
                    
            if record_history:
                self.history.append((max_unit.unit, deepcopy(sub_amps)))  # Takes long time to deepcopy

            # For debugging
            if len(units_detected) == 1:
                self.sub_amps = sub_amps

        # assert len(units_detected) == 1, "Test, hopefully error is raised at least once to show spike splitting works"
        
        for time, unit in zip(self.times, self.units):
            if unit not in units_detected:
                idx = np.searchsorted(unit.spike_train, time)
                unit.spike_train = np.delete(unit.spike_train, idx)

        return len(units_detected)
    
    def plot(self, recording):
        """
        For first spike in self.times, for each unit in self.units, do unit.plot(spike waveform)
        
        no show()
        """
        
        chans_rms = self.chans_rms
        
        fig, axes = plt.subplots(1, 2, figsize=(3.6*len(self.units), 4.8))

        # Get waveforms to plot        
        frame = round(self.times[0] * recording.get_sampling_frequency())
        wfs = recording.get_traces_filt(frame-30, frame+30+1)
        
        # Plot units
        kwargs = self.units[0].plot(recording, subplot=(fig, axes[0]), chans_rms=chans_rms, wf=wfs)
        for i, unit in enumerate(self.units[1:]):
            unit.plot(recording, subplot=(fig, axes[i+1]), wf=wfs, **kwargs)
            
        # Return data (used for self.plot_splitting)
        return kwargs, wfs, frame-30
        
    def plot_splitting(self, recording):
        """
        Plot process of splitting spike (use after self.split())
        First plot is from self.plot()
        Second plot: each column shows wf after a split on the unit that detected it
        
        The second plot does not change amplitudes of electrodes if there is no thresh crossing (gray)
        """
        
        chans_rms = self.chans_rms
        cross_times = self.cross_times
        
        assert len(self.history) > 0, "Use self.plot_splitting() after self.split()"

        print(self.times)
        
        kwargs, wfs, start_frame = self.plot(recording)
        plt.show()
        
        # Create axes
        num_axes = len(self.history)
        fig, axes = plt.subplots(1, num_axes, figsize=(3.6*num_axes, 4.8))
        axes = np.atleast_1d(axes)
        
        # Plot each split
        for h in range(len(self.history)):
            unit = self.history[h][0]
            if h > 0:
                sub_amps = self.history[h-1][1]
                for (elec, cross_idx), amp in sub_amps.items():
                    wf = wfs[elec]
                    # Convert cross_idx to frame in wfs to subtract amp from wfs
                    if cross_idx < len(cross_times[elec]):
                        cross_time = cross_times[elec][cross_idx]
                        cross_frame = round(cross_time * recording.get_sampling_frequency()) - start_frame
                        if 0 <= cross_frame < wf.size: 
                            # Account for thresh crossing lasting more than one frame
                            ## Find left side of peak
                            left_idx = cross_frame
                            while left_idx-1 >= 0 and wf[left_idx-1] < 0 and wf[left_idx-1] >= wf[left_idx]:
                                left_idx -= 1
                            ## Find right side of peak
                            right_idx = cross_frame + 1
                            while right_idx < wf.size and wf[right_idx] < 0 and wf[right_idx] >= wf[right_idx-1]:
                                right_idx += 1
                            
                            left_idx = max(left_idx, left_idx-2)  # Add a couple frames of buffer
                            right_idx = min(right_idx, right_idx+2)
                            
                            wf_amp = wf[cross_frame]
                            rel_amps = wf[left_idx:right_idx] / wf_amp
                            wf[left_idx:right_idx] = rel_amps * (wf_amp + amp) 
            unit.plot(recording, subplot=(fig, axes[h]), wf=wfs, **kwargs)
                
        plt.show()


def split_spikes(merged_propagations,
                 cross_times, cross_amps,
                 chans_rms, rms_thresh,
                 split_overlap,
                 ms_before, ms_after,
                 min_coactivations_n, min_coactivations_p,
                 recording=None):
    
    # Sort spikes
    all_spikes = []
    all_units = []
    for unit in merged_propagations:
        all_spikes.extend(unit.spike_train)
        all_units += [unit] * len(unit.spike_train)
    order = np.argsort(all_spikes)
    all_spikes = np.array(all_spikes)[order]
    all_units = np.array(all_units)[order]
    
    multidetections = [MultiDetection(all_spikes[0], all_units[0],
                                      cross_times, cross_amps,
                                      chans_rms, rms_thresh,
                                      ms_before, ms_after,
                                      min_coactivations_n, min_coactivations_p
                                      )]
    for time, unit in zip(all_spikes[1:], all_units[1:]):
        # if time not in [126705.6, 126705.63333333333]:
        #     continue
        
        # if time < 28790.6 or time > 28790.7:
        #     continue
        
        if time - multidetections[-1].times[-1] <= split_overlap:
            multidetections[-1].add(time, unit)
        else:
            multidetections.append(MultiDetection(time, unit,
                                                  cross_times, cross_amps,
                                                  chans_rms, rms_thresh,
                                                  ms_before, ms_after,
                                                  min_coactivations_n, min_coactivations_p
                                                  ))

    # Split spikes
    count = 0
    for multi in tqdm(multidetections):
        if count > 50:
            break
        
        if len(multi) < 2:
            continue
        
        num_overlaps = multi.split()
        # if num_overlaps > 1 and recording is not None:
            # if count == 3:
            #     print("-"*50)
            #     multi.split(record_history=True)
            #     multi.plot_splitting(recording)
            #     for key, value in multi.sub_amps.items():
            #         print(key, value)
            #     return multi
            # print("-"*50)
            # multi.split(record_history=True)
            # multi.plot_splitting(recording)
            
            # assert False
            # return multi
            # count += 1
            
    return multidetections
   
    
def summarize_pair_overlap(unit1, unit2, delta=0.4):
    """
    Summarize overlap between two units' spike trains
    """
    times1 = unit1.spike_train
    times2 = unit2.spike_train
    
    same_frames = Comparison.count_same_frames(times1, times2)
    overlaps = Comparison.count_matching_events(times1, times2, delta)
    print(f"""#spikes:
Same frame: {same_frames}
Overlaps: {overlaps}
Unit A: {len(times1)}
Unit B: {len(times2)}""")
        
        
"""
Potential pseudocode for real-time spike sorting/splitting:
Main:
Cat all thresh crossings of electrodes into one long 1d array

Loop through each thresh crossing

    For each propagation that has an electrode that had crossing
        prop[elec] = time
        if enough coactivations and first elec detects and no ISI violation
            Check if spike belongs in previous MultiDetection
                If yes: add to multidetection
                If no: split multidetection and create new one starting with spike

Split MultiDetection:
Same as before but only need to keep track of electrodes that are in at least 1 propagation
    
"""
# endregion


# region Merge kilosort units
class KilosortMerge(Unit):
    """
    Represent merged kilosort units
    """
    def __init__(self, prop_unit, initial_unit):
        super().__init__(initial_unit.idx, initial_unit.spike_train, initial_unit.chan, initial_unit.recording)
        
        self.prop_unit = prop_unit
        
        self.units = [initial_unit]
        self.templates = initial_unit.templates
    
    def add_unit(self, unit):
        self.units.append(unit)
        
        cat_spike_times = self.cat_spike_times(self.spike_train, unit.spike_train)
        self.templates = (self.templates * len(self.spike_train)/len(cat_spike_times)) + (unit.templates * len(unit.spike_train)/len(cat_spike_times))
        self.spike_train = cat_spike_times
    
    def plot(self, chans_rms):
        fig, axes = plt.subplots(1, 1+len(self.units), figsize=(2+3.6*len(self.units), 4.8))
        
        kwargs = self.prop_unit.plot(subplot=(fig, axes[0]), chans_rms=chans_rms)
        
        isi_f = self.get_isi_viol_f()
        fig.suptitle(f"{len(self.spike_train)} spikes. {isi_f*100:.2f}% ISI viol")
        
        # kwargs = None
        for ax, unit in zip(axes[1:], self.units):
            # if kwargs is None:
            #     kwargs = unit.plot(axis=ax, chans_rms=CHANS_RMS)
            # else:
            unit.plot(axis=ax, **kwargs)
                                
        plt.show()        
            
        # ISI Violations
        print(f"% ISI violations for each unit")
        df_index = []
        isi_viol_p = []
        for unit in self.units:
            df_index.append(unit.idx)
            isi_viol_p.append(round(unit.get_isi_viol_f() * 100, 2))
            
        display(HTML(pd.DataFrame([isi_viol_p], columns=df_index).to_html(index=False)))
            
        # Calculate waveform dissimilarity matrix
        wf_diff_matrix = np.full((len(self.units), len(self.units)), -1)
        df_index = []
        for i in range(len(self.units)):
            df_index.append(self.units[i].idx)
            for j in range(i+1, len(self.units)):
                wf_diff_matrix[i, j] = self.compare_wf_diff(self.units[i].templates, self.units[j].templates)
                
        print("\nWaveform dissimilarity for each unit pair")
        display(pd.DataFrame(wf_diff_matrix, index=df_index, columns=df_index))
        
    def can_merge(self, unit, max_isi_viol_p, max_wf_diff):
        """
        Determine if another unit can merge with self
        
        :param max_isi_viol_p:
            If None, use min of ISI violation percentage of either unit
        """
        # ISI violation
        if max_isi_viol_p is None:
            max_isi_viol_p = min(self.get_isi_viol_f(), unit.get_isi_viol_f()) * 100
        merge_isi_viol = self.compare_isi_viol(self.spike_train, unit.spike_train)
        if merge_isi_viol > max_isi_viol_p / 100:
            return False

        # Wf diff
        wf_diff = self.compare_wf_diff(self.templates, unit.templates)
        return wf_diff <= max_wf_diff
          
    def __len__(self):
        return len(self.units)
          
    @staticmethod
    def cat_spike_times(spike_times1, spike_times2):
        return np.sort(np.concatenate((spike_times1, spike_times2)))
          
    @staticmethod      
    def compare_isi_viol(spike_times1, spike_times2, isi_viol=1.5):
        cat_spike_times = KilosortMerge.cat_spike_times(spike_times1, spike_times2)
        isis = np.diff(cat_spike_times)
        violation_num = np.sum(isis <= isi_viol)
        return violation_num / len(cat_spike_times)
    
    @staticmethod
    def compare_wf_diff(templates1, templates2):
        templates1 = templates1 / np.abs(np.min(templates1))
        templates2 = templates2 / np.abs(np.min(templates2))
        return np.sum(np.abs(templates1 - templates2))
        
            
def split_ks_merge(prop_unit, ks_units,
                   max_isi_viol_p, max_wf_diff,
                   chans_rms=None):
    """
    :param chans_rms:
        If not None, plot
    
    """
    
    # Sort units from most to least spikes
    nums_spikes = [-len(unit) for unit in ks_units]
    order = np.argsort(nums_spikes)

    # For each ks unit, add to existing merge or split
    merges = [KilosortMerge(prop_unit, ks_units[order[0]])]
    for idx in order[1:]:
        unit = ks_units[idx]
        # Find which merge (if any) unit belongs
        for merge in merges:
            if merge.can_merge(unit, max_isi_viol_p, max_wf_diff):
                merge.add_unit(unit)
                break  # Unit can only merge with one merge
        else:  # Own new merge if not merged with preexisting one
            merges.append(KilosortMerge(prop_unit, unit))
        
    if chans_rms is not None:        
        for merge in merges:
            merge.plot(chans_rms)
            
    return merges

            
def merge_ks_units(prop_units, ks_units, agreement_scores,
                   max_isi_viol_p, max_wf_diff,
                   chans_rms=None):
    """
    Pseudocode:
    Merge kilosort units

        1. Initially merge units if overlap score with same prop unit >= 0.5
            a. Store unmerged units
            b. Store which units could be merged
        2. Potentially merge units with highest spike count first
    
    :params agreement_scores:
        Has shape (num_prop_units, num_ks_units)
    :param chans_rms:
        If not None, plot each merge
    """

    # Merge ks units that match with same prop unit
    merged_ks_units = []  # [[ks_units in merge]]

    prop_to_ks = {}  # {prop_idx: [ks_unit1, ks_unit2, ... with score >= 0.5]}
    for ks_idx, ks_unit in enumerate(ks_units):
        scores = agreement_scores[:, ks_idx]
        prop_idx = np.argmax(scores)
        score = scores[prop_idx]
        if score < 0.5:
            merged_ks_units.append(KilosortMerge(prop_units[prop_idx], ks_unit))
        else:
            if prop_idx not in prop_to_ks:
                prop_to_ks[prop_idx] = [ks_unit]
            else:
                prop_to_ks[prop_idx].append(ks_unit)
            
    # Split ks units
    for prop_idx, ks_units in prop_to_ks.items():
        prop_unit = prop_units[prop_idx]
        if len(ks_units) > 1:
            if chans_rms is not None:
                # If plotting, print to separate merge plots
                print("-"*50)
            merges = split_ks_merge(prop_unit, ks_units, 
                                    max_isi_viol_p=max_isi_viol_p, max_wf_diff=max_wf_diff,
                                    chans_rms=chans_rms)
            merged_ks_units.extend(merges)
        else:
            merged_ks_units.append(KilosortMerge(prop_unit, ks_units[0]))
            
    return merged_ks_units


def grid_search_num_merges(prop_units, ks_units, agreement_scores,
                            max_isi_viol_p_values, max_wf_diff_values):
    # For each hyperparameter, show the number of units in each merge with more than 1 unit
    # ("merges" with only 1 unit are not counted)
    
    num_merges_matrix = np.zeros((len(max_isi_viol_p_values), len(max_wf_diff_values)), dtype=int)
    for r, max_isi_viol_p in enumerate(max_isi_viol_p_values):
        for c, max_wff_diff in enumerate(max_wf_diff_values):
            merged_ks_units = merge_ks_units(prop_units, ks_units, agreement_scores,
                                             max_isi_viol_p=max_isi_viol_p, max_wf_diff=max_wff_diff)           
            num_merges_matrix[r, c] = sum(len(merge) for merge in merged_ks_units if len(merge) > 1)
            
    dataframe = pd.DataFrame(num_merges_matrix, index=max_isi_viol_p_values, columns=max_wf_diff_values)
    display(dataframe)
# endregion


# region Only prop spikes within kilosort spikes
def select_prop_spikes_within_kilosort_spikes(prop_units, ks_units, recording,
                                              max_ms_dist, max_micron_dist):
    """
    For each prop unit, exclude spikes that occur when no spikes from kilosort are detected 
    within max_ms_dist and max_micron_dist
    
    Pseudocode:
    [sorted spike times]
    [xy position]
    for each prop unit
        for each spike
            np.serachsorted(sorted spike times, time-0.4ms)
            while loop  
                if any xy position and spike time close enough, break loop and count spike
                store which spikes are found by ks and which are not
    """
    
    # Get sorted spike times and corresponding xy-positions
    chan_locs = recording.get_channel_locations()
    
    all_ks_spike_times, all_ks_spike_locs = [], []
    for unit in ks_units:
        all_ks_spike_times.extend(unit.spike_train)
        all_ks_spike_locs += [chan_locs[unit.chan]] * len(unit.spike_train)
    order = np.argsort(all_ks_spike_times)
    all_ks_spike_times, all_ks_spike_locs = np.array(all_ks_spike_times)[order], np.array(all_ks_spike_locs)[order]
    
    # Start loop
    within_prop_units = []
    outside_prop_units = []
    for unit in prop_units:
        loc = chan_locs[unit.chan]
        within_spikes = []
        outside_spikes = []
        for spike in unit.spike_train:
            idx = np.searchsorted(all_ks_spike_times, spike-max_ms_dist, side="left")  # Find nearest spike in all_ks_spike_times
            while idx < len(all_ks_spike_times) and np.abs(all_ks_spike_times[idx] - spike) <= max_ms_dist:
                if utils.calc_dist(*all_ks_spike_locs[idx], *loc) <= max_micron_dist:
                    within_spikes.append(spike)
                    break
                idx += 1
            else:
                outside_spikes.append(spike)

        within_prop_units.append(Unit(unit.idx, np.array(within_spikes), unit.chan, unit.recording))
        outside_prop_units.append(Unit(unit.idx, np.array(outside_spikes), unit.chan, unit.recording))
        # within_prop_units.append(PropUnit(unit.df, unit.idx, np.array(within_spikes), unit.recording))
        # outside_prop_units.append(PropUnit(unit.df, unit.idx, np.array(outside_spikes), unit.recording))
        
    return within_prop_units, outside_prop_units
        
# endregion

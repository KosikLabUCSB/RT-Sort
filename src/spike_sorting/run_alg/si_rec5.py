"""
Functions and classes for si_rec5.ipynb
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

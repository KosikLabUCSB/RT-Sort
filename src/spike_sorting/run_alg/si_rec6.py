from pathlib import Path
import pickle
from math import ceil
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import diptest
from tqdm import tqdm

from src import utils
from src.sorters.base import Unit, SpikeSorter

# region Plotting
def plot_elec_probs(unit, elec_probs=None, idx=0,
                    loc_prob_thresh=17.5,
                    amp_kwargs=None, prob_kwargs=None):
    if not isinstance(unit, Unit):
        unit = Unit(idx=idx, spike_train=unit.spike_train, channel_idx=unit.root_elecs[0], recording=RECORDING)
        unit.set_templates()
        
    if elec_probs is None:
        elec_probs = np.mean(extract_detection_probs(unit), axis=0)
    # mid = elec_probs.shape[1]//2
    # elec_probs = elec_probs[:, mid-6:mid+6]
    
    # Plot elec probs on top of each other
    # plt.plot(elec_probs[[0, 2, 4, 1, 3], :].T)
    # plt.show()
    
    prob_chans_rms = [-loc_prob_thresh/5/100]*RECORDING.get_num_channels()  # Extra modifiers to loc_prob_thresh make thresh line visible in plot
    prob_chans_rms = np.array(prob_chans_rms)

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(3.2*2, 4.8))

    amp_kwargs = {"chans_rms": CHANS_RMS} if amp_kwargs is None else amp_kwargs
    amp_kwargs = unit.plot(axis=ax0, **amp_kwargs)
    
    prob_kwargs = {"chans_rms": prob_chans_rms} if prob_kwargs is None else prob_kwargs
    prob_kwargs = unit.plot(axis=ax1, wf=elec_probs, **prob_kwargs)
    
    return amp_kwargs, prob_kwargs

def plot_amp_dist(cluster, ylim=None, **hist_kwargs):
    """
    Params
    cluster
        Can be obj with attr spike_train
        Or np.array of amplitudes
    """
    
    try:
        amplitudes = np.array(get_amp_dist(cluster))
    except AttributeError:
        amplitudes = cluster
    
    # Set default number of bins
    if "bins" not in hist_kwargs:
        hist_kwargs["bins"] = 40
    
    plt.hist(amplitudes, **hist_kwargs)
    plt.xlabel("Amplitude / (median/0.6745)")
    # plt.xlabel("Amplitude (µV)")
    plt.ylabel("#spikes")
    plt.ylim(ylim)

    # dip, pval = diptest.diptest(amplitudes)
    # print(f"p-value: {pval:.3f}")
    # print(f"ISI viol%: {get_isi_viol_p(cluster):.2f}%")
    
    return amplitudes

def plot_split_amp(cluster, thresh):
    """
    Divide cluster's spike train into spikes below and above amp thresh
    Then plot resulting footprints
    """

    # If CocCluster class (._spike_train)
    spike_train = cluster.spike_train
    
    amplitudes = plot_amp_dist(cluster, bins=40)
    amplitudes = np.array(amplitudes)
    plt.show()
    
    dip, pval = diptest.diptest(amplitudes)
    print(f"Dip test p-val: {pval}")

    cluster._spike_train = spike_train
    amp_kwargs, prob_kwargs = plot_elec_probs(cluster)
    plt.show()
    print(f"ISI viol %:", get_isi_viol_p(cluster))

    cluster._spike_train = spike_train[amplitudes < thresh]
    plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.show()
    
    cluster._spike_train = spike_train[amplitudes >= thresh]
    plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.show()
    print(f"ISI viol %:", get_isi_viol_p(cluster))
    
    print(f"ISI viol %:", get_isi_viol_p(cluster))


    cluster._spike_train = spike_train
    
    # If Unit class (.spike_train)
    # spike_train = cluster.spike_train
    
    # amplitudes = plot_amp_dist(cluster, bins=40)
    # plt.show()

    # cluster.spike_train = spike_train
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = spike_train[amplitudes < thresh]
    # plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = spike_train[amplitudes >= thresh]
    # plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = og_spike_train
# endregion


# region Electrode dists
def calc_elec_dist(elec1, elec2):
    # Calculate the spatial distance between two electrodes
    x1, y1 = ELEC_LOCS[elec1]
    x2, y2 = ELEC_LOCS[elec2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_nearby_elecs(ref_elec, max_dist=100):
    nearby_elecs = []
    for elec in ALL_CLOSEST_ELECS[ref_elec]:
        if calc_elec_dist(ref_elec, elec) <= max_dist:
            nearby_elecs.append(elec)
    return nearby_elecs

def get_merge_elecs(ref_elec, max_dist=100):
    # [ref_elec] + get_nearby_elecs
    return [ref_elec] + get_nearby_elecs(ref_elec, max_dist)
# endregion


# region Load files
def load_coc_dict(elec, max_v=1, time_frame=None):
    with open(COC_DICT_ROOT / f"max{max_v}/{elec}.pickle", "rb") as f:
        coc_dict = pickle.load(f)
    if time_frame is not None:
        start_ms, end_ms = time_frame
        new_coc_dict = {}
        for (root_time, root_amp), cocs in coc_dict.items():
            if start_ms <= root_time <= end_ms:
                new_cocs = []
                for elec, time, amp in cocs:
                    if start_ms <= time <= end_ms:
                        new_cocs.append((elec, time, amp))
                if len(new_cocs) > 0:
                    new_coc_dict[(root_time, root_amp)] = new_cocs
        coc_dict = new_coc_dict
    return coc_dict
# endregion


# region Model output utils
def rec_ms_to_output_frame(ms):
    # Convert time (in ms) in recording to frame in model's outputs
    return round(ms * 30) - FRONT_BUFFER

def sigmoid(x):
    # return np.where(x>=0,
    #                 1 / (1 + np.exp(x)),
    #                 np.exp(x) / (1+np.exp(x))
    #                 )
    return np.exp(x) / (1+np.exp(x))  # Positive overflow is not an issue because DL does not output large positive values (only large negative)
# endregion


# region Extract recording/output data
def extract_waveforms(prop, num_cocs=500, ms_before=0.5, ms_after=0.5):
    """
    Parameters
    ----------
    num_cocs: int
        Number of cocs to sample to extract detection probabilities
    ms_before and ms_after: float
        Window for extracting probabilities
    """
    np.random.seed(231)

    # Load outputs 
    outputs = np.load(TRACES_FILT_PATH, mmap_mode="r")  # Load each time to allow for multiprocessing
    num_chans, total_num_frames = outputs.shape

    # Load spike train
    spike_train = prop.spike_train
    if num_cocs is not None and len(spike_train) > num_cocs:
        spike_train = np.random.choice(spike_train, size=num_cocs, replace=False)

    # Extract waveforms
    n_before = round(ms_before * SAMP_FREQ)
    n_after = round(ms_after * SAMP_FREQ)
    all_waveforms = np.zeros((len(spike_train), num_chans, n_before+n_after+1), dtype="float32")  # (n_chans, n_samples)
    for i, time_ms in enumerate(spike_train):
        time_frame = round(time_ms * SAMP_FREQ)
        if time_frame-n_before < 0 or time_frame+n_after+1 > total_num_frames :  # Easier and faster to ignore edge cases than to handle them
            continue
        
        window = outputs[:, time_frame-n_before:time_frame+n_after+1]
        all_waveforms[i] = window
    # return np.mean(all_waveforms, axis=0)
    return all_waveforms

def extract_detection_probs(prop, num_cocs=300, ms_before=0.5, ms_after=0.5):
    """
    Parameters
    ----------
    num_cocs: int
        Number of cocs to sample to extract detection probabilities
    ms_before and ms_after: float
    """
    np.random.seed(231)

    # Load outputs 
    outputs = np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")  # Load each time to allow for multiprocessing
    num_chans, total_num_frames = outputs.shape

    # Load spike train
    spike_train = prop.spike_train
    if num_cocs is not None and len(spike_train) > num_cocs:
        spike_train = np.random.choice(spike_train, size=num_cocs, replace=False)

    # Extract probabilities
    n_before = round(ms_before * SAMP_FREQ)
    n_after = round(ms_after * SAMP_FREQ)
    all_elec_probs = np.zeros((len(spike_train), num_chans, n_before+n_after+1), dtype="float16")  # (n_cocs, n_chans, n_samples) float16: Model's output is float16
    for i, time_ms in enumerate(spike_train):
        time_frame = rec_ms_to_output_frame(time_ms)
        if time_frame-n_before < 0 or time_frame+n_after+1 > total_num_frames :  # Easier and faster to ignore edge cases than to handle them
            continue
        
        window = outputs[:, time_frame-n_before:time_frame+n_after+1]
        all_elec_probs[i] = sigmoid(window)
    # elec_probs /= len(spike_train)
    return all_elec_probs
# endregion


# region Form sequence trunks
class CocCluster:
    def __init__(self, root_elec, time, latency, rel_amp) -> None:
        self.root_elecs = [root_elec]
        self._spike_train = [time]  # Unordered spike train
        self.mean_latency = latency
        self.mean_rel_amp = rel_amp
        
    def add_coc(self, time, latency, rel_amp):        
        mean_latency = len(self._spike_train) * self.mean_latency + latency
        mean_rel_amp = len(self._spike_train) * self.mean_rel_amp + rel_amp
        
        self._spike_train.append(time)
        self.mean_latency = mean_latency / len(self._spike_train)
        self.mean_rel_amp = mean_rel_amp / len(self._spike_train)
        
    def calc_latency_diff(self, other):
        try:
            other_mean_latency = other.mean_latency
        except AttributeError:
            other_mean_latency = other
        return np.abs(self.mean_latency - other_mean_latency)
    
    # def calc_rel_amp_diff(self, other):
    #     if isinstance(other, CocCluster):
    #         other_mean_rel_amp = other.mean_rel_amp
    #     else:  # Other is a value (rel amp)
    #         other_mean_rel_amp = other
    #     return np.abs(self.mean_rel_amp - other_mean_rel_amp) / self.mean_rel_amp 
    
    def merge(self, other):
        mean_latency = len(self._spike_train) * self.mean_latency + len(other._spike_train) * other.mean_latency
        mean_rel_amp = len(self._spike_train) * self.mean_rel_amp + len(other._spike_train) * other.mean_rel_amp
        
        self._spike_train += other._spike_train
        self.mean_latency = mean_latency / len(self._spike_train)
        self.mean_rel_amp = mean_rel_amp / len(self._spike_train)
        
    @property
    def spike_train(self):
        return np.sort(self._spike_train)
    

def branch_coc_cluster(root_elec, comp_elecs,
                       coc_dict, allowed_root_times,
                       max_latency_diff, min_cocs,
                       verbose=False):
    """
    Recursive function, first called in form_coc_clusters

    Params:
    allowed_root_times
        The root_times in the cluster being branched, i.e. new clusters can only be formed from the times in allowed_root_times
    max_latency_diff
        For coc to join cluster, the latency difference on the comp elec has to be at most max_latency_diff
        (Keep value as float (even if wanting value to be 0) to account for floating point rounding error)
    """
    
    comp_elec = comp_elecs[0]
    
    if verbose:
        print(f"Comparing to elec {comp_elec}, loc: {ELEC_LOCS[comp_elec]}")
        
    # Form new elec clusters
    elec_clusters = []
    # Iterate through each coc with root elec
    for (root_time, root_amp), cocs in coc_dict.items():
        # Only check cocs that are a subset of the cluster being grown
        if allowed_root_times is not None and root_time not in allowed_root_times:
            continue
        
        # Check each electrode that cooccurs
        for tar_elec, tar_time, tar_amp in cocs:
            if tar_elec == comp_elec:  # Comp elec found
                # Form new cluster for coc or add to existing cluster
                closest_cluster = None
                min_diff = max_latency_diff
                for cluster in elec_clusters:
                    diff = cluster.calc_latency_diff(tar_time - root_time)
                    if diff <= min_diff:
                        closest_cluster = cluster
                        min_diff = diff
                if closest_cluster is not None:  # Add to existing cluster
                    closest_cluster.add_coc(root_time, tar_time - root_time, tar_amp / root_amp)
                else:  # Form new cluster
                    elec_clusters.append(CocCluster(root_elec, root_time, tar_time - root_time, tar_amp / root_amp))
    
    # Due to moving averages with adding cocs to cluster, CocClusters may be within max_latency_diff, so they need to be merged
    dead_clusters = set()
    while True:
        # Find best merge
        merge = None
        min_diff = max_latency_diff
        for i in range(len(elec_clusters)):        
            cluster_i = elec_clusters[i]
            if cluster_i in dead_clusters:
                continue
            for j in range(i+1, len(elec_clusters)):            
                cluster_j = elec_clusters[j]
                if cluster_j in dead_clusters:
                    continue
                diff = cluster_i.calc_latency_diff(cluster_j)
                if diff <= min_diff:
                    merge = [cluster_i, cluster_j]
                    min_diff = diff
                    
        # If no merges are found, end loop
        if merge is None:
            break
        
        merge[0].merge(merge[1])
        dead_clusters.add(merge[1])
    
    elec_clusters = [c for c in elec_clusters if c not in dead_clusters and len(c._spike_train) >= min_cocs]
        
    if len(comp_elecs) == 1:
        # No more elecs to compare to
        return elec_clusters
    
    # Recursion
    new_elec_clusters = []
    for cluster in elec_clusters:
        branches = branch_coc_cluster(
            root_elec, comp_elecs[1:],
            coc_dict, allowed_root_times=set(cluster._spike_train),
            max_latency_diff=max_latency_diff, min_cocs=min_cocs,
            verbose=verbose
        )
    
        if len(branches) == 0:  # Maybe cluster was split into too many branches that separately did not have enough cocs
            new_elec_clusters.append(cluster)
        else:
            new_elec_clusters += branches
        
    return new_elec_clusters


def form_coc_clusters(root_elec, time_frame, 
                      max_latency_diff, min_cocs,
                      max_elec_dist=100,
                      elec_patience=6,
                      verbose=False):

    coc_dict = load_coc_dict(root_elec, time_frame=time_frame)

    comp_elecs = []
    for elec in ALL_CLOSEST_ELECS[root_elec]:
        if calc_elec_dist(elec, root_elec) <= max_elec_dist:
            comp_elecs.append(elec)
        else:
            break
        
    allowed_root_times = {root_time for root_time, root_amp in coc_dict.keys()}

    if verbose:
        print(f"Starting with elec {root_elec}, loc: {ELEC_LOCS[root_elec]}")
        print(f"{len(allowed_root_times)} cocs total")

    all_coc_clusters = []
    patience_counter = 0
    for c in range(len(comp_elecs)):    
        # Not need to compare c to elecs less than c since those cocs would have already been analyzed
        if verbose: 
            print(f"\nComparing to elec {comp_elecs[c]}, loc: {ELEC_LOCS[comp_elecs[c]]}")
            
        coc_clusters = branch_coc_cluster(root_elec, comp_elecs[c:],
                                        coc_dict, allowed_root_times=allowed_root_times,
                                        max_latency_diff=max_latency_diff, min_cocs=min_cocs,
                                        verbose=False)
        for cluster in coc_clusters:
            allowed_root_times.difference_update(cluster._spike_train)
            all_coc_clusters.append(cluster)
        
        if verbose:
            print(f"Found {len(coc_clusters)} clusters")
            print(f"{len(allowed_root_times)} cocs remaining")
            
        if len(allowed_root_times) < min_cocs:
            print(f"\nEnding early because too few cocs remaining")
            break
        
        if len(coc_clusters) == 0:
            patience_counter += 1
        else:
            patience_counter = 0

        if verbose:
            print(f"Patience counter: {patience_counter}/{elec_patience}")
            
        if patience_counter == elec_patience:
            if verbose:
                print(f"\nStopping early due to patience")
            break
            
    if verbose:
        print(f"\nTotal: {len(all_coc_clusters)} clusters")
        
    return all_coc_clusters

    # Show all root times and remaining root times after forming clusters
    # unit = Unit(0, list({root_time for root_time, root_amp in coc_dict.keys()}), root_elec, RECORDING)
    # amp_kwargs, prob_kwargs = plot_elec_probs(unit)
    # plt.show()

    # unit = Unit(0, list(allowed_root_times), root_elec, RECORDING)
    # plot_elec_probs(unit, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
# endregion


# region Form sequences v2
def get_elec_spikes(elec, time_frame,
                    min_num_coacs=2, elec_prob_thresh=-2.1972245773362196,  # inverse sigmoid (0.1)
                    max_dist=100,
                    verbose=False):
    """
    Get spikes whose max amp is on :param elec:
    """
    n_before=15
    n_after=15    
    
    elecs = get_merge_elecs(elec, max_dist=max_dist)    

    spike_times = []
    crossings_ind = ELEC_CROSSINGS_IND[elec]
    if verbose:
        crossings_ind = tqdm(crossings_ind)
    
    start_time, end_time = time_frame
    for ref_idx in ELEC_CROSSINGS_IND[elec]:
        _, ref_time, ref_amp = ALL_CROSSINGS[ref_idx]
        if ref_time < start_time:
            continue
        if ref_time > end_time:
            break
        
        # Extract output window
        output_frame = rec_ms_to_output_frame(ref_time)
        output_window = OUTPUTS[elecs, max(0, output_frame-n_before):output_frame+n_after+1]
    
        # Extract probabilities
        # elec_probs = sigmoid(np.max(output_window, axis=1))
        num_coacs = sum(np.max(output_window, axis=1) >= elec_prob_thresh)
        if num_coacs < min_num_coacs:
            continue
        
        # Extract traces window
        # rec_frame = round(ref_time * SAMP_FREQ)
        # rec_window = TRACES[elecs, max(0, rec_frame-n_before):rec_frame+n_after+1]

        # Check if :param elec: is max elec
        # max_elec = np.argmax(np.max(rec_window, axis=1))
        # if max_elec != 0:  # 0 = :param elec:
        #     continue
        
        spike_times.append(ref_time)

    return spike_times
    
# endregion


# region Setup coc clusters
def _setup_coc_clusters_parallel(cluster):
    # Job for setup_coc_clusters_parallel
    setup_cluster(cluster)
    return cluster

def setup_coc_clusters_parallel(coc_clusters):
    """
    Run setup_cluster on coc_clusters with parallel processing
    """
    new_coc_clusters = []
    with Pool(processes=20) as pool:
        for cluster in tqdm(pool.imap(_setup_coc_clusters_parallel, coc_clusters), total=len(coc_clusters)):
            new_coc_clusters.append(cluster)
    return new_coc_clusters

def setup_coc_clusters(coc_clusters):
    # Set important data needed for merging and other analyses
    for cluster in coc_clusters:        
        setup_cluster(cluster)

def setup_cluster(cluster, elec_prob_thresh=0.1,
                  rel_to_closest_elecs=3):
    """
    Parameters:
    elec_prob_thresh:
        Prob on elec needs to cross this to count as part of trunk
    rel_to_closest_elecs:
        Set relative amplitudes relative to mean amp of rel_to_closet_elecs elecs
    """
    
    # Set important data needed for merging, assigning spikes, and other analyses
    all_elec_probs = extract_detection_probs(cluster)  # (n_cocs, n_chans, n_samples)
    elec_probs = np.mean(all_elec_probs, axis=0)  # (n_chans, n_samples)
    
    # Find probabilities used for elec weighting
    elec_weight_probs = []
    for probs in elec_probs:  # 1it for each elec. probs: (n_samples)
        peak = np.argmax(probs)
        elec_weight_probs.append(np.sum(probs[peak-1:peak+2]))
    
    # Needed for latencies and amplitudes
    waveforms = extract_waveforms(cluster)
    
    latencies = np.argmax(elec_probs, axis=1)
    # latencies = np.argmin(np.mean(waveforms, axis=0), axis=1)
    cluster.latencies = latencies - elec_probs.shape[1] // 2 # in frames
    
    # Save for plotting
    cluster.all_elec_probs = all_elec_probs  
    cluster.elec_probs = elec_probs
    
    # Save for merging
    cluster.elec_weight_probs = np.array(elec_weight_probs)  # (n_chans,)
    cluster.amplitudes = get_amp_dist(cluster)
    
    # Save for assigning spikes
    elecs = get_merge_elecs(cluster.root_elecs[0])
    elec_weight_probs = cluster.elec_weight_probs[elecs]
    cluster.elec_weights = elec_weight_probs / np.sum(elec_weight_probs)
    # cluster.main_elecs = np.flatnonzero(elec_weight_probs >= elec_prob_thresh)
    cluster.main_elecs = np.flatnonzero(np.max(cluster.elec_probs[elecs], axis=1) >= elec_prob_thresh)
    cluster.elecs = elecs  # Elecs to watch for comparing latencies and rel amps
    
    # cluster.elecs = np.flatnonzero(np.max(elec_probs, axis=1)>=prob_thresh)  # No longer needed
    
    wf_amps = waveforms[:, range(waveforms.shape[1]), (latencies).astype(int)]  # (n_wfs, n_chans)
    mean_amps = np.abs(np.mean(wf_amps, axis=0))
    
    cluster.waveforms = waveforms
    cluster.mean_amps = mean_amps

    # Save for assigning spikes to increase speed
    cluster.rel_amps = mean_amps / np.mean(mean_amps[cluster.elecs[:rel_to_closest_elecs]])
    cluster.latencies_elecs = cluster.latencies[cluster.elecs]
    cluster.rel_amps_elecs = cluster.rel_amps[cluster.elecs]

def get_amp_dist(cluster, n_before=100, n_after=100):
    """
    Get amplitude distribution of cluster.spike_train
    """
    
    try:
        root_elec = cluster.chan
    except AttributeError:
        root_elec = cluster.root_elecs[0]
    
   
    amplitudes = []
    for spike in cluster.spike_train:
        frame = round(spike * SAMP_FREQ)
        window = TRACES[root_elec, max(0, frame-n_before):frame+n_after]
        # amp = np.abs(window[n_before] / np.sqrt(np.mean(np.square(window))))
        amp = np.abs(window[n_before] / np.median(np.abs(window)/0.6745))
        amplitudes.append(amp)
    
    return amplitudes 
# endregion


# region Merge
class Merge:
    # Represent a CocCluster merge
    def __init__(self, cluster_i, cluster_j) -> None:
        self.cluster_i = cluster_i
        self.cluster_j = cluster_j
        self.closest_elecs = cluster_i.elecs  # Should not really matter whose elecs since clusters should be close together
        
        i_probs = cluster_i.elec_weight_probs
        j_probs = cluster_j.elec_weight_probs
        # self.elec_probs = (i_probs + j_probs) / 2  # /2 to find average between two elecs
        self.elec_probs = np.max(np.vstack((i_probs, j_probs)), axis=0)  # Max between two elecs
        
    def get_elec_weights(self, elecs, min_r=0.03):
        """
        Get elec weights for weighted average using :param elecs:
        
        Params:
        min_r:
            Weights below min_r are set to 0
        """
        # Get elec weights for weighted average using :param elecs:
        elec_probs = self.elec_probs[elecs]
        elec_prob_weights = elec_probs / np.sum(elec_probs)
        elec_prob_weights[elec_prob_weights < min_r] = 0
        
        mask = elec_prob_weights >= min_r
        new_elec_probs = elec_probs[mask]
        new_elec_probs = new_elec_probs / np.sum(new_elec_probs)
        elec_prob_weights[mask] = new_elec_probs
        return elec_prob_weights
        
        # return elec_prob_weights
        
    def score_latencies(self):
        elecs = self.closest_elecs[1:]  # Ignore root electrode since latency=0 always
        elec_weights = self.get_elec_weights(elecs)
        
        latencies_i = self.cluster_i.latencies[elecs]
        # Set latencies_j relative to root on latencies_i
        latencies_j = self.cluster_j.latencies[self.closest_elecs]
        latencies_j = latencies_j[1:] - latencies_j[0]
        
        # for e, (i, j) in enumerate(zip(latencies_i, latencies_j)):
        #     print(elecs[e], i, j, round(elec_weights[e]*100, 1), round(np.abs(i -j)*elec_weights[e], 2))

        latency_diff = np.abs(latencies_i - latencies_j)
        latency_diff = elec_weights * latency_diff
        latency_diff = np.sum(latency_diff)
        
        return latency_diff
        
    def score_rel_amps(self):
        assert False, "Consider using amp/median"
        
        elecs = self.closest_elecs
        elec_weights = self.get_elec_weights(elecs)
        
        # Clusters' amps relative to different electrodes
        # i_amps = self.cluster_i.mean_amps[elecs]
        # i_rel_amps = i_amps / np.mean(-np.sort(-i_amps)[:3])
        # j_amps = self.cluster_j.mean_amps[elecs]
        # j_rel_amps = j_amps / np.mean(-np.sort(-j_amps)[:3])
            
        # To the same electrodes
        i_amps = self.cluster_i.mean_amps[elecs]
        i_rel_amps = i_amps / np.mean(i_amps[:3])
        j_amps = self.cluster_j.mean_amps[elecs]
        j_rel_amps = j_amps / np.mean(j_amps[:3])
        
        # rel_amp_div = np.min(np.vstack((i_rel_amps, j_rel_amps)), axis=0)
        rel_amp_div = np.mean(np.vstack((i_rel_amps, j_rel_amps)), axis=0)
        
        rel_amp_diff = np.abs(i_rel_amps - j_rel_amps) / rel_amp_div
        rel_amp_diff = elec_weights * rel_amp_diff
        rel_amp_diff = np.sum(rel_amp_diff)
        return rel_amp_diff
    
    def score_amp_dist(self):
        """
        Return p-value of Hartigan's dip test on amplitude distribution
        """
        # all_amps = get_amp_dist(self.cluster_i) + get_amp_dist(self.cluster_j)
        
        assert False, "Amplitudes need to be extracted on the same electrode"
        
        all_amps = self.cluster_i.amplitudes + self.cluster_j.amplitudes
        # Calculate the dip statistic and p-value
        dip, pval = diptest.diptest(np.array(all_amps))
        return pval
        
    def can_merge(self, max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
        return (self.score_latencies() <= max_latency_diff) and (self.score_rel_amps() <= max_rel_amp_diff) and (self.score_amp_dist() >= min_amp_dist_p)
        
        # Incorporate % spike overlap to determine whether or not to merge
        # if not ((self.score_latencies() <= max_latency_diff) and (self.score_rel_amps() <= max_rel_amp_diff)):
        #     return False        
        
        # num_i = len(self.cluster_i.spike_train)
        # num_j = len(self.cluster_j.spike_train)
        # num_overlaps = len(set(self.cluster_i.spike_train).intersection(self.cluster_j.spike_train))
        # return num_overlaps / (num_i + num_j - num_overlaps) >= 0.3

    def merge(self):
        # Combine spike trains, but if both clusters detect same spike, only add once
        spike_train_i = self.cluster_i.spike_train
        spike_train_j = self.cluster_j.spike_train
        
        amplitudes_i = self.cluster_i.amplitudes
        amplitudes_j = self.cluster_j.amplitudes
        
        spike_train = [spike_train_i[0]]
        amplitudes = [amplitudes_i[0]]
        i, j = 1, 0
        while i < len(spike_train_i) and j < len(spike_train_j):
            spike_i, spike_j = spike_train_i[i], spike_train_j[j]
            if spike_i < spike_j:  # i is next to be added
                if np.abs(spike_train[-1] - spike_i) >= 0.1: # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_i)
                    amplitudes.append(amplitudes_i[i])
                i += 1
            else:  # j is next to be added
                if np.abs(spike_train[-1] - spike_j) >= 0.1: # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_j)
                    amplitudes.append(amplitudes_j[j])
                j += 1

        # Append remaning elements (only one cluster's spike train can be appended due to while loop)
        if i < len(spike_train_i):
            spike_train.extend(spike_train_i[i:])
            amplitudes.extend(amplitudes_i[i:])
        else:
            spike_train.extend(spike_train_j[j:])
            amplitudes.extend(amplitudes_j[j:])
        
        # Actually do merge
        try:
            self.cluster_i._spike_train = spike_train
        except AttributeError:
            self.cluster_i.spike_train = spike_train
        self.cluster_i.amplitudes = amplitudes
        
        # Update root elecs
        cluster_i_elecs = set(self.cluster_i.root_elecs)
        for elec in self.cluster_j.root_elecs:
            if elec not in cluster_i_elecs:
                self.cluster_i.root_elecs.append(elec)
            
        setup_cluster(self.cluster_i)  # Update stats
        
        return self.cluster_j  # Return to update dead_clusters

    def is_better(self, other, max_latency_diff=1, max_rel_amp_diff=1):
        """
        Determine if self is a better merge than other
        
        Parameters
        ----------
        max_latency_diff:
            Scale latency diff by this to normalize it, so it can be compared to rel amp
        max_rel_amp_diff:
            Scale rel amp diff by this to normalize it, so it can be compared to latency 
        """
        
        self_diff = self.score_latencies() / max_latency_diff + self.score_rel_amps() / max_rel_amp_diff# + (1-self.score_amp_dist())
        other_diff = other.score_latencies() / max_latency_diff + other.score_rel_amps() / max_rel_amp_diff #+ (1-other.score_amp_dist())
        return self_diff < other_diff

    def summarize(self):
        """
        Print merge metrics
        """
        print(f"Latency diff: {self.score_latencies():.2f}. Rel amp diff: {self.score_rel_amps():.2f}")
        print(f"Amp dist p-value {self.score_amp_dist():.4f}")

        

def merge_verbose(merge, update_history=True):
    """
    Verbose for merge_coc_clusters
    
    Params:
    update_history:
        If True, history of clusters will be updated
        False is for when no merge is found, but still want verbose
    """
    cluster_i, cluster_j = merge.cluster_i, merge.cluster_j
    
    if hasattr(cluster_i, "merge_history"):
        message = f"\nMerged {cluster_i.merge_history} with "
        if update_history:
            cluster_i.merge_history.append(cluster_j.idx)                
    else:
        message = f"\nMerged {cluster_i.idx} with "
        if update_history:
            cluster_i.merge_history = [cluster_i.idx, cluster_j.idx]
    
    if hasattr(cluster_j, "merge_history"):
        message += str(cluster_j.merge_history)
        if update_history:
            cluster_i.merge_history += cluster_j.merge_history[1:]
    else:
        message += f"{cluster_j.idx}"
    print(message)
    print(f"Latency diff: {merge.score_latencies():.2f}. Rel amp diff: {merge.score_rel_amps():.2f}")
    print(f"Amp dist p-value {merge.score_amp_dist():.4f}")

    print(f"#spikes:")
    # num_overlaps = Comparison.count_matching_events(cluster_i.spike_train, cluster_j.spike_train)
    num_overlaps = len(set(cluster_i.spike_train).intersection(cluster_j.spike_train))
    print(f"Merge base: {len(cluster_i.spike_train)}, Add: {len(cluster_j.spike_train)}, Overlaps: {num_overlaps}")
    
    # Find ISI violations after merging
    # cat_spikes = np.sort(np.concatenate((cluster_i.spike_train, cluster_j.spike_train)))
    # diff = np.diff(cat_spikes)
    # num_viols = np.sum(diff <= 1.5)
    # print(f"ISI viols: {num_viols}")
    
    # # Plot footprints
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster_i, idx=cluster_i.idx)
    # plt.show()
    # plot_elec_probs(cluster_j, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs, idx=cluster_j.idx)
    # plt.show()   
    
    # # Plot amp distribution
    # all_amps = get_amp_dist(cluster_i) + get_amp_dist(cluster_j)
    # plot_amp_dist(np.array(all_amps))
    # plt.show()

def merge_coc_clusters(coc_clusters,
                       max_root_elec_dist=1, 
                       max_latency_diff=2.51, max_rel_amp_diff=0.4, min_amp_dist_p=0.075,
                       auto_setup_coc_clusters=True, verbose=False):
    """
    Parameters
    ----------
    max_root_elec_dist:
        For merge to happen, root elecs have to be within max_root_elec_dist
    max_merge_elec_dist:
        Only compare electrodes within max_merge_elec_dist micrometers of root elec
    
    """
    
    if auto_setup_coc_clusters:
        setup_coc_clusters(coc_clusters)

    for idx, cluster in enumerate(coc_clusters):
        cluster.idx = idx
        
    # region study merge scores
    # elec_overlap_matrix = np.full((len(coc_clusters), len(coc_clusters)), -1.0)
    # rel_amp_diff_matrix = np.full((len(coc_clusters), len(coc_clusters)), np.nan)
    # latency_diff_matrix = np.full((len(coc_clusters), len(coc_clusters)), np.nan)
    # isi_viol_pass_matrix = np.full((len(coc_clusters), len(coc_clusters)), np.nan)
    # for i in range(len(coc_clusters)):
    #     cluster_i = coc_clusters[i]
    #     for j in range(i+1, len(coc_clusters)):            
    #         cluster_j = coc_clusters[j]
    #         merge = Merge(cluster_i, cluster_j, elecs)
            
    #         # elecs = list(set(cluster_i.elecs).intersection(cluster_j.elecs))
            
    #         # For calculating weighted mean
    #         # max_probs = (np.max(cluster_i.elec_probs,axis=1) + np.max(cluster_j.elec_probs,axis=1)) / 2 # /2 to find average between two elecs
            
    #         # Rel amps
    #         # max_probs_elecs = max_probs[elecs]
    #         # elec_weights = max_probs_elecs / np.sum(max_probs_elecs)
            
    #         # i_amps = cluster_i.mean_amps[elecs]
    #         # i_rel_amps = i_amps / np.mean(-np.sort(-i_amps)[:3])
    #         # j_amps = cluster_j.mean_amps[elecs]
    #         # j_rel_amps = j_amps / np.mean(-np.sort(-j_amps)[:3])
            
    #         # rel_amp_diff = np.abs(i_rel_amps - j_rel_amps)
    #         # rel_amp_diff = elec_weights * rel_amp_diff
    #         # rel_amp_diff = np.sum(rel_amp_diff)
    #         # rel_amp_diff_matrix[i, j] = rel_amp_diff
    #         rel_amp_diff_matrix[i, j] = merge.score_rel_amps()
            
    #         # Latencies. Ignore root_elec since latency is guaranteed to be 0 
    #         # elecs = [e for e in elecs if e != root_elec]
    #         # max_probs_elecs = max_probs[elecs]
    #         # elec_weights = max_probs_elecs / np.sum(max_probs_elecs)
            
    #         # latency_diff = np.abs(cluster_i.latencies[elecs] - cluster_j.latencies[elecs])
    #         # latency_diff = elec_weights * latency_diff
    #         # latency_diff = np.sum(latency_diff)
    #         # latency_diff_matrix[i, j] = latency_diff
    #         latency_diff_matrix[i, j] = merge.score_latencies()
            
    #         # ISI violations
    #         # i_train = cluster_i.spike_train
    #         # i_viol_p = calc_isi_viol_p(i_train)
    #         # j_train = cluster_j.spike_train
    #         # j_viol_p = calc_isi_viol_p(j_train)
    #         # ij_viol = calc_isi_viol_p(np.sort(np.concatenate((i_train, j_train))))
    #         # isi_viol_pass_matrix[i, j] = ij_viol - min(i_viol_p, j_viol_p)
            
    # df = DataFrame(rel_amp_diff_matrix)
    # display(df)
        
    # df = DataFrame(latency_diff_matrix)
    # display(df)
    
    # # df = DataFrame(isi_viol_pass_matrix)
    # # display(df)

    # # print("use isi violation")
    # endregion
        
    # region Merge
    # Find best merge
    
    dead_clusters = set()
    while True:
        # Find best merge
        best_merge = None
        best_unmerge = None  # Best merge that cannot merge (for final verbose)
        for i in range(len(coc_clusters)):   
            # Load cluster i     
            cluster_i = coc_clusters[i]
            if cluster_i in dead_clusters:
                continue
            
            for j in range(i+1, len(coc_clusters)):    
                # Load cluster j        
                cluster_j = coc_clusters[j]
                if cluster_j in dead_clusters:
                    continue
                
                # Check if root elecs are close enough
                if calc_elec_dist(cluster_i.root_elecs[0], cluster_j.root_elecs[0]) > max_root_elec_dist:
                    continue
                
                # Calculate quality of merge
                cur_merge = Merge(cluster_i, cluster_j)
                if not cur_merge.can_merge(max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
                    if verbose and (best_unmerge is None or cur_merge.is_better(best_unmerge)):
                        best_unmerge = cur_merge
                    continue
                if best_merge is None or cur_merge.is_better(best_merge, max_latency_diff, max_rel_amp_diff):
                    best_merge = cur_merge
                    
        # if len(dead_clusters) == 1:
        #     break
                    
        # If no merges are good enough
        if best_merge is None:
        # if not best_merge.can_merge(max_latency_diff, max_rel_amp_diff):
            if verbose:
                print(f"\nNo merge found. Next best merge:")
                merge_verbose(best_unmerge, update_history=False)
            break
        
        # Merge best merge
        if verbose:
            merge_verbose(best_merge)
        
        dead_clusters.add(best_merge.merge())
        if verbose:
            print(f"After merging: {len(best_merge.cluster_i.spike_train)}")
            
    merged_clusters = [cluster for cluster in coc_clusters if cluster not in dead_clusters]
    
    if verbose:       
        print(f"\nFormed {len(merged_clusters)} merged clusters:")
        for m, cluster in enumerate(merged_clusters):
            # message = f"cluster {m}: {cluster.idx}"
            # if hasattr(cluster, "merge_history"):
            #     message += f",{cluster.merge_history}"
            # print(message)
            
            # Without +[]
            if hasattr(cluster, "merge_history"):
                print(f"cluster {m}: {cluster.merge_history}")
            else:
                print(f"cluster {m}: {cluster.idx}")
        # print(f"Formed {len(merged_clusters)} merged clusters")  # Reprint this because jupyter notebook cuts of middle of long outputs
    return merged_clusters
    #endregion
# endregion


# region Assign spikes
def get_isi_viol_p(cluster, isi_viol=1.5):
    spike_train = cluster.spike_train
    diff = np.diff(spike_train)
    viols = np.sum(diff <= isi_viol)
    return viols / len(spike_train) * 100


def get_spike_match(cluster, root_time,
                    elec_prob_thresh=0.1, elec_prob_mask=0.03,
                    rel_to_closest_elecs=3,
                    min_coacs_r=0.5, max_latency_diff=2.51, max_rel_amp_diff=0.40):
    """
    Return how well spike at :param time: matches with :param unit:
    
    Params:
    elec_prob_thresh:
        Prob on elec needs to cross this to count as coactivation
    max_latency_diff
        Used for determining size of extraction window 
        
        
    Returns:
    ratio of elecs that have detection, latency diff, rel amp diff
    """
    
    # Load elecs
    main_elecs = cluster.main_elecs
    elecs = cluster.elecs
    
    # Load cluster stats
    cluster_latencies = cluster.latencies_elecs 
    cluster_rel_amps = cluster.rel_amps_elecs  
    
    # Calculate extraction window n_before and n_after
    #+1 for good measure
    n_before = round(np.min(cluster_latencies)) + ceil(max_latency_diff) + 1  # Use these signs so n_before is positive
    n_before = max(1, n_before)  # Ensure at least one frame n_before
    n_after = round(np.max(cluster_latencies)) + ceil(max_latency_diff) + 1
    n_after = max(1, n_after)  

    # Extract latencies
    output_frame = rec_ms_to_output_frame(root_time)
    output_window = OUTPUTS[elecs, max(0, output_frame-n_before):output_frame+n_after+1]
    latencies = np.argmax(output_window, axis=1) 
        
    # Extract probabilities    
    elec_probs = sigmoid(output_window[range(latencies.size), latencies])
    
    # Calculate coacs ratio
    coacs_r = sum(elec_probs >= elec_prob_thresh)/len(main_elecs)
    if coacs_r < min_coacs_r:
        return coacs_r, np.inf, np.inf
    
    # Calculate elec weights
    elec_weights = (cluster.elec_weights + elec_probs) / 2
    above_mask = elec_weights >= elec_prob_mask  # 11/19/23  This should be after finding weighted values
    
    new_elec_weights = elec_weights[above_mask]
    elec_weights[above_mask] = new_elec_weights / np.sum(new_elec_weights)
    elec_weights[~above_mask] = 0
    
    # Calculate latency diff
    latency_diff = np.sum(np.abs(cluster_latencies - (latencies - n_before)) * elec_weights)
    if latency_diff > max_latency_diff:
        return coacs_r, latency_diff, np.inf
    
    # Extract rel amps
    rec_frame = round(root_time * SAMP_FREQ)
    rec_window = TRACES[elecs, max(0, rec_frame-n_before):rec_frame+n_after+1]  # Not need max for end of slice since numpy ok with it
    amps = np.abs(rec_window[range(len(latencies)), latencies])
    rel_amps = amps / np.mean(amps[:rel_to_closest_elecs])     
    
    # Calculate rel amp diff
    rel_amp_diff = np.abs(cluster_rel_amps - rel_amps) / cluster_rel_amps
    rel_amp_diff = np.sum((rel_amp_diff * elec_weights))
    
    return coacs_r, latency_diff, rel_amp_diff

# As a first pass, only detect spikes with DL detection with crossing on first elec
# (since DL detects so many spikes, this is probably fine)
def assign_spikes(all_units, time_frame, interelec=False,
                  min_coacs_r=0.5, max_latency_diff=2.51, max_rel_amp_diff=0.4,  # 2.01, 0.35
                  overlap_time=0.1, overlap_dist=50,  # overlap_time=0.04
                  verbose=False):
    """
    Pseudocode:
    1. Have units watch root_elecs for DL detections
    2. When detection on elec, assign spike to unit with best footprint match
    If len(root_elecs)>1: unit watches multiple elecs, but ignores other elecs
    if detects a spike within Xms of another elec (prevent counting same spike multiple times)
    
    Leaves spikes in a buffer zone. Only split spikes within Xms and Xum
    Remove all in front of spike_buffer that occur too before new spike, leave the rest
    
    Attempted method: Fails. Hard to account for when a unit belongs to multiple split groups
        Rationale: Tries to account for edge case of spikes 0.1, 0.3, 0.45 when overlap_time=0.4. 
        Naive method would split 0.1 and 0.3, but perhaps 0.3 should be split with 0.45
        
    OLD - Spike splitting pseudocode:
        Before running through recording:
    1. Determine which units are in the same split group
    2. Create dict[unit] = split group id
        While running through recording:
    1. When unit detects spike, fill in dict[split group id] = [(unit, spike)]
        a. If new spike is not within overlap_time, split spikes
    
        Actually split spikes:
    1. Form two spike clusters: 
        1) spikes closest to earliest spike
        2) spikes closest to latest spike (only if spikes are within overlap_time of latest spike)
    2. In cluster 1, assign spike to unit with best match
    3. Remove spikes in cluster 1 from dict[split group id]    
    
    At end, split again 
    
    Units can belong to more than split groups. unit.spike_train is a set, so if a unit
    is the best match for a spike in multiple groups, the spike is only added once
    
    NEW - Spike splitting pseudocode
    1. Add detected spikes to buffer
    2. If newest spike added is farther than overlap_time with oldest spike, start splitting:
        a. Select spikes that are not within overlap_time of newest spike or closer to oldest spike 
            (Unselected spikes remain in buffer)
        b. Find which spike has highest overlap score and assign to unit
        c. Remove all spikes that are within overlap_dist of best spike
        d. Repeat step b until no spikes remain
        
    Params:
    inter_elec:
        If True, assume all_units contains different root elecs (slower, but needed for interelec spike splitting)
    """
    
    for unit in all_units:
        unit._spike_train = []
        unit._elec_train = []  # Indicates which root elec's detection led to spike in unit._spike_train
        
    # For ith elec, which units are watching
    elec_watchers = {}
    for unit in all_units:
        for elec in unit.root_elecs:
            if elec not in elec_watchers:
                elec_watchers[elec] = [unit]
            else:
                elec_watchers[elec].append(unit)        
                      
    # Start watching for spikes
    spike_buffer = []  # Store spikes before they have been assigned
    
    all_crossings_times = [c[1] for c in ALL_CROSSINGS]
    start_idx = np.searchsorted(all_crossings_times, time_frame[0], side="left")
    end_idx = np.searchsorted(all_crossings_times, time_frame[1], side="right")
    crossings = ALL_CROSSINGS[start_idx:end_idx]
    if verbose:
        crossings = tqdm(crossings)
    
    for elec, time, amp in crossings:
        if elec not in elec_watchers:  # No units are watching elec
            continue
        
        # Intraelectrode spike splitting
        best_unit = None
        best_score = np.inf
        for unit in elec_watchers[elec]:
            elecs_r, latency_diff, rel_amp_diff = get_spike_match(unit, time,
                                                                  max_latency_diff=max_latency_diff, min_coacs_r=min_coacs_r)
            
            if elecs_r >= min_coacs_r and latency_diff <= max_latency_diff and rel_amp_diff <= max_rel_amp_diff:
                # Score spike match with footprint (lower score is better)
                match_score = (1-elecs_r) + latency_diff / max_latency_diff + rel_amp_diff / max_rel_amp_diff  # Need to normalize different metrics
                if match_score < best_score:
                    best_unit = unit
                    best_score = match_score

        if best_unit is None:
            continue
        
        if interelec:
            spike_buffer.append((best_unit, time, elec, best_score))
            if len(spike_buffer) > 1 and time - spike_buffer[0][1] > overlap_time:
                split_interelec_spike(spike_buffer, time, overlap_time, overlap_dist)
        else:
            best_unit._spike_train.append(time)
            
    if interelec:     
        if len(spike_buffer) > 1:
            split_interelec_spike(spike_buffer, time, overlap_time, overlap_dist)
        elif len(spike_buffer) == 1:
            unit, time, elec, score = spike_buffer[0]
            unit._spike_train.append(time)
            unit._elec_train.append(elec)
     
def split_interelec_spike(spike_buffer, time,
                          overlap_time, overlap_dist):      
    # Find which spikes overlap more with earliest spike than latest, split these
    first_time = spike_buffer[0][1]
    
    overlapping_spikes = []
    while len(spike_buffer) > 0:
        old_time = spike_buffer[0][1]
        if (old_time - first_time) > (time - old_time):  # old_time is closer to new time than first_time, so it should be split with new time
            break
        overlapping_spikes.append(spike_buffer.pop(0))
    
    # Split spikes
    while len(overlapping_spikes) > 0:
        # Find best score
        best_data = [None, None, None, np.inf]  # (unit, time, elec, score)
        for spike_data in overlapping_spikes:
            if spike_data[3] < best_data[3]:
                best_data = spike_data
                        
        # Assign spike to unit
        unit, time, elec, score = best_data
        if len(unit._elec_train) == 0 or \
        time - unit._spike_train[-1] > overlap_time or unit._elec_train[-1] == elec:  # If same spike is detected by a different root_elec, do not assign spike
            unit._spike_train.append(time)
            unit._elec_train.append(elec)
        
        # Remove all spikes within overlap_dist of best spike
        for s in range(len(overlapping_spikes)-1, -1, -1):
            if calc_elec_dist(elec, overlapping_spikes[s][2]) <= overlap_dist:
                overlapping_spikes.pop(s)
# endregion


# region Form full sequences
def form_from_root(root_elec, time_frame, min_cocs=50, verbose=False):
    # Form and merge propgations from root_elec
    coc_clusters = form_coc_clusters(root_elec, time_frame, 1/SAMP_FREQ,
                                       min_cocs=min_cocs, verbose=verbose)

    setup_coc_clusters(coc_clusters)

    assign_spikes(coc_clusters, time_frame, verbose=verbose)
    coc_clusters = [c for c in coc_clusters if len(c._spike_train) >= min_cocs]

    merges = merge_coc_clusters(coc_clusters, verbose=verbose)
    
    assign_spikes(merges, time_frame, verbose=verbose)
    merges = [m for m in merges if len(m._spike_train) >= min_cocs]
    
    setup_coc_clusters(merges)
    
    return merges
# endregion


# region Using sorters.py
def clusters_to_clusters(clusters):
    """
    If si_rec6.py is changed, CocCluster objs created in si_rec6.ipynb before the change
    cannot be pickled (which is needed for parallel processing)
    
    This function will initialize new CocCluster objs
    """
    new_clusters = []
    for clust in clusters:
        new_clust = CocCluster(-1, -1, clust.mean_latency, clust.mean_rel_amp)
        new_clust.root_elecs = clust.root_elecs
        new_clust._spike_train = clust._spike_train
        new_clusters.append(new_clust)
    return new_clusters


def clusters_to_units(clusters):
    # Convert CocCluster objs to Unit objs
    all_units = []
    for c, clust in enumerate(clusters):
        unit = Unit(c, clust.spike_train, clust.root_elecs[0], recording=None)  # recording=None for parallel processing
        unit.root_elecs = clust.root_elecs
        unit.mean_amps = clust.mean_amps
        all_units.append(unit)
    return all_units


def clusters_to_sorter(clusters):
    return SpikeSorter(RECORDING, "RT-Sort", units=clusters_to_units(clusters))
# endregion


# region Kilosort comparison
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


if __name__ == "__main__":
    RECORDING = utils.rec_si()
    CHANS_RMS = utils.chans_rms_si()

    SAMP_FREQ = RECORDING.get_sampling_frequency()
    NUM_ELECS = RECORDING.get_num_channels()
    ELEC_LOCS = RECORDING.get_channel_locations()

    COC_DICT_ROOT = Path("/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/prop_signal/coc_dicts")

    ALL_CLOSEST_ELECS = []
    for elec in range(NUM_ELECS):
        elec_ind = []
        dists = []
        x1, y1 = ELEC_LOCS[elec]
        for elec2 in range(RECORDING.get_num_channels()):
            if elec == elec2:
                continue
            x2, y2 = ELEC_LOCS[elec2]
            dists.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            elec_ind.append(elec2)
        order = np.argsort(dists)
        ALL_CLOSEST_ELECS.append(np.array(elec_ind)[order])   
        
    ALL_CROSSINGS = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/prop_signal/all_crossings.npy", allow_pickle=True)
    ALL_CROSSINGS = [tuple(cross) for cross in ALL_CROSSINGS]
    
    ELEC_CROSSINGS_IND = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/prop_signal/elec_crossings_ind.npy", allow_pickle=True)
    ELEC_CROSSINGS_IND = [tuple(ind) for ind in ELEC_CROSSINGS_IND]  # [(elec's cross times ind in all_crossings)]
    
    
    TRACES_FILT_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/traces_filt.npy"
    MODEL_OUTPUTS_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/outputs.npy"
    FRONT_BUFFER = 40  # Model's front sample buffer
    
    TRACES = np.load(TRACES_FILT_PATH, mmap_mode="r")
    OUTPUTS = np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")
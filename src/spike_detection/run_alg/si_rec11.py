from pathlib import Path
import pickle
from math import ceil
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture

# from diptest import diptest
from tqdm import tqdm

from src import utils
# from src.comparison import Comparison
# from src.sorters.base import Unit, SpikeSorter


# region Utils
def pickle_dump(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        
def pickle_load(path):
    with open(path, "rb") as file:
        return pickle.load(file)
# endregion


# region Plotting
class TracesFiltWrapper:
    """
    Wrapper for RECORDING so it can be used for TRACES_FILT
    """
    def __init__(self, recording):
        self.recording = recording
        self.chans = range(recording.get_num_channels())
        # self.frames = range(recording.get_total_samples()) 
        self.shape = (recording.get_num_channels(), recording.get_total_samples())
        
    def __getitem__(self, idx):
        chans, frames = idx  # unpacking like this is okay since TRACES_FILT[:] is never called
        chans = self.chans[chans]
        return self.recording.get_traces_filt(frames.start, frames.stop, chans)

def plot_elec_probs(seq, idx=None,
                    stringent_prob_thresh=17.5, loose_prob_thresh=10,
                    amp_kwargs=None, prob_kwargs=None):    
    # Convert thresholds from percent to decimal
    stringent_prob_thresh /= 100
    loose_prob_thresh /= 100
    
    # if not isinstance(unit, Unit):
    if hasattr(seq, "root_elecs"):
        channel_idx = seq.root_elecs[0]
    else:
        channel_idx = seq.chan
    
    if idx is None:
        idx = getattr(seq, "idx", 0)     
    
    unit = Unit(idx=idx, spike_train=seq.spike_train, channel_idx=channel_idx, recording=RECORDING)
    # unit.set_templates()
       
    all_elec_probs = extract_detection_probs(unit)  # (num_spikes, num_elecs, num_samples)
    elec_probs = np.mean(all_elec_probs, axis=0)  # for footprint
    # mid = elec_probs.shape[1]//2
    # elec_probs = elec_probs[:, mid-6:mid+6]
    
    # Plot elec probs on top of each other
    # plt.plot(elec_probs[[0, 2, 4, 1, 3], :].T)
    # plt.show()
    
    # CHANS_RMS = np.full(NUM_ELECS, 1)  # For when templates are np.mean(amp/median)

    # For neuropixels
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(3.2*2, 4.8))
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(3.2*2, 5.5))

    # For MEA
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 6))

    # Width of each elec's waveforms and detections (pass refers to whether detection crosses loose_prob_threshold)
    pass_width = 1.5
    not_pass_width = 0.85
    wf_widths = []  # Base
    
    # Color of each elec's waveforms (based latencies)
    wf_colors = []
    # For elec's waveform thickness (Mean/median of individual spikes is different than mean of footprint)
    for c in range(NUM_ELECS):
        # Widths (median probability)
        probs = all_elec_probs[:, c, :]  # (num_spikes, num_samples)
        peaks = np.max(probs, axis=1)  # (num_spikes,)
        median = np.median(peaks)
        if median >= loose_prob_thresh:
            wf_widths.append(pass_width)
        else:
            wf_widths.append(not_pass_width)
            
        # Colors (mean latency)
        peaks = np.argmax(probs, axis=1)  # (num_spikes,)
        mean = np.mean(peaks) - probs.shape[1]//2  # Centered around spike on root elec
        wf_colors.append(mean)

    # Plot waveforms footprint
    amp_kwargs = {"chans_rms": CHANS_RMS} if amp_kwargs is None else amp_kwargs
    
    wfs = extract_waveforms(unit, use_filt=True)
    temps = np.mean(wfs, axis=0)
    amp_kwargs = unit.plot(axis=ax0, wf=temps, 
                           wf_widths=wf_widths, wf_colors=wf_colors,
                           fig=fig,
                           **amp_kwargs)
    
    # Plot detections footprint
    if prob_kwargs is None:
        prob_chans_rms = np.array([-loose_prob_thresh/5]*NUM_ELECS)  # Extra modifiers to loc_prob_thresh make thresh line visible in plot
        add_lines = [[stringent_prob_thresh]*NUM_ELECS]
        prob_kwargs = {
            "chans_rms": prob_chans_rms,
            "add_lines": add_lines
        }
    prob_kwargs = unit.plot(axis=ax1, wf=elec_probs, 
                            wf_widths=wf_widths, 
                            **prob_kwargs)
    
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

def plot_gmm(gmm, data):    
    plt.hist(data, bins=30, density=True, alpha=0.5, color="gray")
    from scipy.stats import norm
    x_range = np.linspace(data.min(), data.max(), 1000)
    for i, (mean, cov, weight) in enumerate(zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)):
        plt.plot(x_range, weight * norm.pdf(x_range, mean, np.sqrt(cov)), label=f'Component {i+1}')

def _save_sequences_plots(args, n_spikes=5):
    sequence, root_path = args
    path = root_path / str(sequence.idx)
    path.mkdir(exist_ok=True, parents=True)

    unit = Unit(sequence.idx, sequence.formation_spike_train, sequence.root_elecs[0], None)  # Plot sequence's spike train in TRAINING_FRAMES
    amp_kwargs, prob_kwargs = plot_elec_probs(unit, idx=sequence.idx)
    # plot_elec_probs(sequence, idx=sequence.idx)
    plt.savefig(path / "average_footprint.jpg", format="jpg")
    plt.close()

    # Plot individual spikes 
    spike_train = sequence.spike_train
    np.random.seed(1)
    for idx in np.random.choice(spike_train.size, n_spikes):
        time = spike_train[idx]
        plot_seq_spike_overlap(sequence, time, idx, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
 
        plt.savefig(path / f"spike{idx}.jpg", format="jpg")
        plt.close()
    
    # Plot amp median distribution
    amp_medians = get_amp_medians(sequence, use_formation_spike_train=True)
    plt.hist(amp_medians, bins=30)
    plt.ylabel("#spikes")
    plt.xlabel("amplitude/median on root electrode")
    plt.savefig(path / "root_amp_median_dist.jpg", format="jpg")
    plt.close()
    
    # Plot individual clusters from merge
    # if hasattr(sequence, "history"):
    #     for idx, cluster in enumerate(sequence.history):
    #         plot_elec_probs(cluster, idx=idx)
    #         plt.savefig(path / f"sequence{idx}.jpg", format="jpg")
    #         plt.close()

def save_sequences_plots(sequences, root_path):
    """
    For each sequence, save a folder in root_path/{idx} containing:
        1. average_footprint.png
        2. n_spikes .png of randomly selected spikes in the sequence
    """
        
    tasks = [(seq, root_path) for seq in sequences]    
    with Pool(processes=20) as pool:
        for _ in tqdm(pool.imap_unordered(_save_sequences_plots, tasks), total=len(tasks)):
            pass
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


# region Model output utils
def rec_ms_to_output_frame(ms):
    # Convert time (in ms) in recording to frame in model's outputs
    return round(ms * SAMP_FREQ) - FRONT_BUFFER

def sigmoid(x):
    # return np.where(x>=0,
    #                 1 / (1 + np.exp(-x)),
    #                 np.exp(x) / (1+np.exp(x))
    #                 )
    x = np.clip(x, a_min=-9, a_max=10)  # Prevent overflow error
    return np.exp(x) / (1+np.exp(x))  # Positive overflow is not an issue because DL does not output large positive values (only large negative)

def sigmoid_inverse(y):
    return -np.log(1 / y - 1)

# endregion


# region Extract recording/output data
def extract_waveforms(prop, num_cocs=300, ms_before=2, ms_after=2,
                      use_filt=False):
    """
    Parameters
    ----------
    num_cocs: int
        Number of cocs to sample to extract detection probabilities
    ms_before and ms_after: float
        Window for extracting probabilities
    use_filt:
        Whether to use TRACES or TRACES_FILT
    """
    np.random.seed(231)

    # Load outputs 
    if use_filt:
        outputs = TRACES_FILT
    else:
        outputs = TRACES
        assert False, "If not used for plotting, need to account that these waveforms have not been divided by median"
    num_chans, total_num_frames = outputs.shape

    # Load spike train
    spike_train = prop.spike_train
    if num_cocs is not None and len(spike_train) > num_cocs:
        spike_train = np.random.choice(spike_train, size=num_cocs, replace=False)

    # Extract waveforms
    n_before = round(ms_before * SAMP_FREQ)
    n_after = round(ms_after * SAMP_FREQ)
    all_waveforms = np.zeros((len(spike_train), num_chans, n_before+n_after+1), dtype="float32")  # (n_spikes, n_chans, n_samples)
    for i, time_ms in enumerate(spike_train):
        time_frame = round(time_ms * SAMP_FREQ)
        if time_frame-n_before < 0 or time_frame+n_after+1 > total_num_frames :  # Easier and faster to ignore edge cases than to handle them
            continue
        
        window = outputs[:, time_frame-n_before:time_frame+n_after+1]    
            
        if not use_filt:
            large_window = outputs[:, max(0, time_frame-500):time_frame+501]
            means = np.mean(large_window, axis=1, keepdims=True)
            # large_window = large_window - means
            # medians = np.median(np.abs(large_window), axis=1, keepdims=True) / 0.6745
            # window = (window - means) / medians
            window = window - means
        
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
    outputs = OUTPUTS # np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")  # Load each time to allow for multiprocessing
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

# def get_amp_median(time, elec, n_before=500, n_after=500):
#     frame = round(time * SAMP_FREQ)
#     n_before = n_before if frame - n_before >= 0 else frame  # Prevent indexing problems
#     window = np.abs(TRACES[elec, frame-n_before:frame+n_after+1])
#     median = np.median(window) / 0.6745
#     return window[n_before] / median

# def calc_window_medians(time, elecs, n_before=500, n_after=500):
#     frame = round(time * SAMP_FREQ)
#     window = TRACES[elecs, max(0, frame-n_before):frame+n_after+1]
#     return np.median(np.abs(window), axis=1) / 0.6745

def calc_pre_mean_median(frame, elecs=slice(None), pre_ms=50):
    """
    Get the preceeding mean and median of traces before :time:
    
    Says "mean" but is actually "median" (because it was changed to well after it was first implemented as mean)
    
    1/13/24 - median has already been extracted from window
    """    
    # frame = round(time * SAMP_FREQ)
    pre_frames = round(pre_ms * SAMP_FREQ)
    window = TRACES[elecs, max(0, frame-pre_frames):frame]
    
    # means = np.median(window, axis=1) # np.mean(window, axis=1)
    
    # median = np.median(np.abs(window - means[:, None]), axis=1) 
    median = np.median(np.abs(window), axis=1) 
    
    # median = np.abs(np.median(np.abs(window), axis=1) - means)  # Faster to subtract means after finding median, but this is inaccurate due to taking abs val of window
    # return means, np.clip(median / 0.6745, a_min=0.5, a_max=None) 
    return 0, np.clip(median / 0.6745, a_min=0.5, a_max=None) 

# endregion


# region Form clusters
class Patience:
    def __init__(self, root_elec, patience_end):
        self.counter = 0
        self.root_elec = root_elec
        
        self.last_dist = 0
        
        self.patience_end = patience_end
        
    def reset(self):
        self.counter = 0
    
    def end(self, comp_elec) -> bool:
        """
        Increment counter and check if to end
        """
        
        dist = calc_elec_dist(self.root_elec, comp_elec)
        if dist != self.last_dist:
            self.counter += 1        
            if self.counter >= self.patience_end:
                return True
                
        self.last_dist = dist
        return False
        
    def verbose(self):
        print(f"Patience: {self.counter}/{self.patience_end}")
        
class CocCluster:
    def __init__(self, root_elec, split_elecs, spike_train):
        # Form 1 initial cluster on root elec
        self.root_elec = root_elec
        self.root_elecs = [root_elec]
        self.split_elecs = split_elecs
        self._spike_train = spike_train
        # self.latencies = []
        
    def split(self, split_elec, spike_train):
        # Split cluster using split_elec
        return CocCluster(self.root_elec, self.split_elecs.union({split_elec}), spike_train)
        
    @property
    def spike_train(self):
        return np.sort(self._spike_train)


def branch_coc_cluster(root_cluster, comp_elecs,
                       coc_dict, allowed_root_times,
                       
                       min_unimodal_p, max_n_components,
                       
                       max_latency_diff, 
                       min_coc_n, min_coc_p,
                       
                       min_extend_comp_p,
                       
                       patience,
                       
                       verbose=False):
    """
    Recursive function, first called in form_coc_clusters

    Params:
    allowed_root_times
        The root_times in the cluster being branched, i.e. new clusters can only be formed from the times in allowed_root_times
    max_latency_diff
        For coc to join cluster, the latency difference on the comp elec has to be at most max_latency_diff
        (Keep value as float (even if wanting value to be 0) to account for floating point rounding error)
    min_coc_p
        If #cluster_cocs/#total_cocs_on_root_elec_comp_pair < min_coc_p/100, cluster is discarded  
    """
    
    comp_elec = comp_elecs[0]
    
    if verbose:
        print(f"Comparing to elec {comp_elec}, loc: {ELEC_LOCS[comp_elec]}")
        
    # region Using absolute latency difference for finding clusters
    # # Form new elec clusters
    # elec_clusters = []
    # # Iterate through allowed root times
        
    # # all_latencies = []  # For plotting all latency distribution
    # for root_time in allowed_root_times:
    #     cocs = coc_dict[root_time]
        
    #     # Check each electrode that cooccurs
    #     for tar_elec, tar_time, tar_latency in cocs:
    #         if tar_elec == comp_elec:  # Comp elec found
    #             # Form new cluster for coc or add to existing cluster
    #             closest_cluster = None
    #             min_diff = max_latency_diff
    #             for cluster in elec_clusters:
    #                 diff = np.abs(cluster.mean_latency - tar_latency)
    #                 if diff <= min_diff:
    #                     closest_cluster = cluster
    #                     min_diff = diff
    #             if closest_cluster is not None:  # Add to existing cluster
    #                 closest_cluster.add_coc(tar_time, tar_latency, -1)
    #             else:  # Form new cluster
    #                 elec_clusters.append(CocCluster(root_elec, tar_time, tar_latency, -1))

    #             # all_latencies.append(tar_latency)  # For plotting all latency distribution 
                
    # # Due to moving averages with adding cocs to cluster, CocClusters may be within max_latency_diff, so they need to be merged
    # dead_clusters = set()
    # while True:
        # Find best merge
        # merge = None
        # min_diff = max_latency_diff
        # for i in range(len(elec_clusters)):        
        #     cluster_i = elec_clusters[i]
        #     if cluster_i in dead_clusters:
        #         continue
        #     for j in range(i+1, len(elec_clusters)):            
        #         cluster_j = elec_clusters[j]
        #         if cluster_j in dead_clusters:
        #             continue
        #         diff = np.abs(cluster_i.mean_latency - cluster_j.mean_latency) 
        #         if diff <= min_diff:
        #             merge = [cluster_i, cluster_j]
        #             min_diff = diff
                    
        # # If no merges are found, end loop
        # if merge is None:
        #     break
        
        # merge[0].merge(merge[1])
        # dead_clusters.add(merge[1])
    # endregion
    
    # Using GMM for finding clusters
    """
    Pseudocode:
        1. Find latencies on comparison electrode
        2. Split cluster based on latencise
            a. Fit GMM
            b. Form clusters based on GMM
            c. Add electrode to clusters's group of splitting elecs
        3. Pick next electrode
        4. Determine if group of upcoming comparison electrodes need to be extended
    """
    
    # 1.
    all_times = []
    all_latencies = []
    # all_amps = []
    for root_time in allowed_root_times:
        cocs = coc_dict[root_time]
        # Check each electrode that cooccurs
        for tar_elec, tar_time, tar_latency in cocs:
            if tar_elec == comp_elec:  # Comp elec found
                all_latencies.append(tar_latency)  # For plotting all latency distribution 
                all_times.append(root_time)
                
                # rec_frame = round(tar_time * SAMP_FREQ)
                # pre_means, pre_medians = calc_pre_mean_median(max(0, rec_frame-N_BEFORE), [comp_elec])        
                # amps = np.abs(TRACES[comp_elec, rec_frame] - pre_means)
                # amp_medians = amps / pre_medians
                # all_amps.append(amp_medians[0])
                
                break
            
    # 2.
    min_cocs = max(min_coc_n, len(allowed_root_times) * min_coc_p/100)
    
    if len(all_times) <= min_cocs:  # Not enough codetections on comp_elec to split, so allowed_root_times stay together        
        coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]      
        
        if patience.end(comp_elec):
            return coc_clusters
    else:          
        all_latencies = np.array(all_latencies)        
        # dip, pval = diptest(all_latencies)
        # if pval >= min_unimodal_p:  # latencies are unimodal
        #     coc_clusters = [root_cluster.split(comp_elec, all_times)]  # TODO: If unimodal latency and enough cocs to be cluster, not sure if cocs with and without coc on comp_elec should be split
        #     # coc_clusters = [root_cluster.split(comp_elec, all_times)]
        
        # gmm_latencies = np.vstack([all_latencies, all_amps]).T
        
        gmm_latencies = all_latencies.reshape(-1, 1)  # Reshape to (n_samples, n_features) for GMM
        best_score = np.inf
        best_gmm = None
        max_n_components = min(max_n_components, len(set(all_latencies)))  # Prevents warning "ConvergenceWarning: Number of distinct clusters (3) found smaller than n_clusters (4). Possibly due to duplicate points in X."
        for n_components in range(1, max_n_components+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            try:
                gmm.fit(gmm_latencies)
                score = gmm.bic(gmm_latencies)      
            except FloatingPointError:  # FloatingPointError: underflow encountered in exp, 1/8/24: guess that this is caused by probably of fitting being so low that it causes underflow error
                continue
        
            # print(f"{n_components} components, {score}")
                 
            if score < 0: # If negative, it fits data too well
                continue
                 
            if score < best_score:
                best_score = score
                best_gmm = gmm
            
            if score < 0:
                continue
                
        if best_gmm is None or best_gmm.n_components == 1:  # If all GMM are negative, use n_components=1
            coc_clusters = [root_cluster.split(comp_elec, all_times)]
            patience.reset()  # Can just reset since "len(all_times) <= min_cocs" ensures enough coc if n_components=1
        else:
            predictions = best_gmm.predict(gmm_latencies)
            coc_clusters = [root_cluster.split(comp_elec, []) for _ in range(best_gmm.n_components)]
            for cluster, time in zip(predictions, all_times):
                coc_clusters[cluster]._spike_train.append(time)  
            # for cluster, time, latency in zip(predictions, all_times, all_latencies):
            #     coc_clusters[cluster]._spike_train.append(time)  
            #     coc_clusters[cluster].latencies.append(latency)           
            
            # region For plotting
            # for i, (mean, cov, weight) in enumerate(zip(best_gmm.means_.flatten(), best_gmm.covariances_.flatten(), best_gmm.weights_)):
            #     coc_clusters[i].plot_data = (mean, cov, weight)
            # endregion
            
            coc_clusters = [c for c in coc_clusters if len(c._spike_train) >= min_cocs]
            
            if len(coc_clusters) == 0:  # If allowed_root_times were split into clusters with not enough spikes, allow original cluster to continue branching
                coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]
                if patience.end(comp_elec):
                    return coc_clusters
            else:
                patience.reset()
                
        # region Naive Plot clusters
        # bins = {}
        # cmap = plt.get_cmap("tab10")
        # test = set()
        # for latency in all_latencies:
        #     if latency not in bins:
        #         bins[latency] = 1
        #     else:
        #         bins[latency] += 1
        # for latency, count in bins.items():
        #     prediction = best_gmm.predict([[latency]])
        #     test.add(prediction[0])
        #     plt.scatter([latency]*count, range(count), color=cmap(prediction[0]))
        # plt.show()
        # endregion

        # region Plot GMM
        # plt.hist(all_latencies, bins=range(-15, 16), density=True, alpha=0.5, color='gray')

        # # Plot the PDFs of individual components
        # from scipy.stats import norm
        # x_range = np.linspace(all_latencies.min(), all_latencies.max(), 1000)
        # for i, cluster in enumerate(coc_clusters):
        #     if hasattr(cluster, "plot_data"):
        #         mean, cov, weight = cluster.plot_data
        #         print(f"cluster {i}, mean: {mean}, #cocs: {len(cluster._spike_train)}")
        #         plt.plot(x_range, weight * norm.pdf(x_range, mean, np.sqrt(cov)), label=f'Component {i+1}')     

        # # Plot the combined PDF (sum of individual components)
        # # pdf_sum = np.sum([weight * norm.pdf(x_range, mean, np.sqrt(cov)) for mean, cov, weight in zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)], axis=0)
        # # plt.plot(x_range, pdf_sum, color='red', linewidth=2, label='Combined PDF')

        # plt.show()
        # endregion

        # region More plotting
        # print()
    
        # print(f"total #coc_clusters: {len(coc_clusters)}")
        # for c in coc_clusters:
        #     s = len(c._spike_train)
        #     print(f"{s} cocs", f"{round(s/len(allowed_root_times)*100)}%")
            
        # predictions = best_gmm.predict(gmm_latencies)
        # clusters = [[] for _ in range(best_gmm.n_components)]
        # for label, time in zip(predictions, all_times):
        #     clusters[label].append(time)
        
        # for i, cluster in enumerate(clusters):
        #     unit = Unit(i, cluster, root_cluster.root_elec, None)
        #     plot_elec_probs(unit)
        #     plt.show()
        
        # # Plot distribution of all latencies
        # # latency_min = min(all_latencies)
        # # latency_max = max(all_latencies)
        # plt.hist(all_latencies, bins=range(-15, 16, 1)) #, bins=range(round(latency_min), round(latency_max)+1))
        # plt.xlim(-15, 15)
        # plt.xticks(range(-15, 16, 2))
        # plt.show()
        # assert False  
        # endregion


    if len(comp_elecs) == 1:  # If no more elecs to compare to/branch
        return coc_clusters
    
    # Recursion branching
    min_extend_comp = len(allowed_root_times) * min_extend_comp_p/100
    comp_elecs_set = set(comp_elecs)
    new_coc_clusters = []
    for cluster in coc_clusters:
        # Check if enough cocs for further splitting       
        if len(cluster._spike_train) <= min_coc_n: 
            new_coc_clusters.append(cluster)
            continue 

        # Check whether to add more electrodes to comp_elecs
        if len(cluster._spike_train) >= min_extend_comp:
            cluster_comp_elecs = comp_elecs[1:]
            for elec in ALL_CLOSEST_ELECS[comp_elec]:
                if calc_elec_dist(comp_elec, elec) > INNER_RADIUS:
                    break
                if elec not in comp_elecs_set and elec not in cluster.split_elecs:  # Prevent 1) double counting elecs 2) splitting on an elec that has already been used for splitting
                    cluster_comp_elecs.append(elec)
        else:
            cluster_comp_elecs = comp_elecs[1:]      

        # Actually do recursion
        branches = branch_coc_cluster(
            cluster, cluster_comp_elecs,
            coc_dict, allowed_root_times=set(cluster._spike_train),
            min_unimodal_p=min_unimodal_p, max_n_components=max_n_components,
            max_latency_diff=max_latency_diff, 
            min_coc_n=min_coc_n, min_coc_p=min_coc_p,
            min_extend_comp_p=min_extend_comp_p,
            patience=patience,
            verbose=verbose
        )
    
        new_coc_clusters += branches
        
    return new_coc_clusters


def form_coc_clusters(root_elec, time_frame, 
                      max_latency_diff=2.5, 
                      min_coc_n=10, min_coc_p=10, 
                      min_extend_comp_p=50,
                      min_unimodal_p=0.100001, max_n_components_amp=4,
                      max_n_components_latency=4,
                      min_coc_prob=0.175, 
                      max_amp_elec_dist=100, comp_elec_dist=50,
                      elec_patience=6,
                      verbose=False):    
    """
    Params
        min_unimodal_p
            If diptest p-value is >= min_unimodal_p, the distribution (amp/median) is considered unimodal
        max_n_components
            If amp/median is multimodal, use BIC to determine best n_components to fit amp/median dist, ranging from 2 to max_n_components
        max_amp_elec_dist:
            Which elecs to compare to see if root_elec has max (largest) amp/median
        comp_elec_dist:
            Which cocs to grow tree on 
        
        n_before and n_after
            Coacs need to be within :n_before: and :n_after: frames of root spike
    
    """
    
    n_before = N_BEFORE
    n_after = N_AFTER
    
    min_unimodal_p = MIN_AMP_DIST_P
    min_coc_prob = STRINGENT_THRESH
    
    # Setup
    comp_elecs = get_nearby_elecs(root_elec, comp_elec_dist)
    max_amp_elecs = [root_elec] + get_nearby_elecs(root_elec, max_amp_elec_dist)
    min_coc_prob = sigmoid_inverse(min_coc_prob)
    
    # Extract root_times and amp/median
    start_time, end_time = time_frame
    all_times = []
    for root_idx in ELEC_CROSSINGS_IND[root_elec]:
        time = ALL_CROSSINGS[root_idx][1]
        if time < start_time:
            continue
        elif time > end_time:
            break
        else:
            all_times.append(time)
    
    if verbose:
        print(f"Starting with elec {root_elec}, loc: {ELEC_LOCS[root_elec]}")
        print("\nFinding coocurrences")
        all_times = tqdm(all_times)
        
    allowed_root_times = set() 
    root_amp_medians = []
    coc_dict = {}  # root time to cocs [(elec, latency)]
    root_time_to_amp_median = {}
    num_activity_cocs = 0  # To see if elec contains activity
    for time in all_times:  # Two loops for this so tqdm is accurate
        # Check if time is largest amp/median NOTE: Amp here is defined as max value in traces. Amp in other areas is defined by location of DL prediction. (Probably doesn't make a difference since DL is pretty accurate. Also doing it differently here might be better since the max-amp threshold is slightly more stringent this way, which is better for forming sequence backbones)
        rec_frame = round(time * SAMP_FREQ)
        
        # # Filtered traces and median of surrounding window (named old for debugging)
        # old_traces = TRACES[max_amp_elecs, max(0, rec_frame-n_before):rec_frame+n_after+1]
        # old_amps = np.abs(np.min(old_traces, axis=1))
        # window_medians = calc_window_medians(time, max_amp_elecs)
        # old_amp_medians = old_amps / window_medians
        # amp_medians = old_amp_medians
        
        # Raw traces and mean and median of preceeding window
        this_n_before = n_before if rec_frame - n_before >= 0 else rec_frame  # Prevents indexing problems
        start_frame = rec_frame - this_n_before
        pre_means, pre_medians = calc_pre_mean_median(start_frame, max_amp_elecs)   
             
        traces = TRACES[max_amp_elecs, start_frame:rec_frame+n_after+1]  # Use rec_frame here so its rec_frame-this_n_before:rec_frame+n_after+1
        amps = np.abs(np.min(traces, axis=1) - pre_means)
        amp_medians = amps / pre_medians
        
        if np.argmax(amp_medians) != 0:
            continue
        
        # Check if time has enough a coac with comp_elecs
        output_frame = rec_ms_to_output_frame(time)
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        cocs = []
        for elec in comp_elecs:
            # Check if elec coactivates
            output_window = OUTPUTS[elec, output_frame-this_n_before:output_frame+n_after+1]
            prob = np.max(output_window)
            if prob < min_coc_prob:
                continue
            
            # Add to coc_dict
            latency = np.argmax(output_window) - this_n_before # (np.argmax(output_window) - this_n_before) / SAMP_FREQ  # root_elec detects spike at this_n_before
            cocs.append((elec, time, latency))
        if len(cocs) > 0:
            allowed_root_times.add(time)
            coc_dict[time] = cocs
            root_amp_medians.append(amp_medians[0])    
            root_time_to_amp_median[time] = amp_medians[0]
        if len(cocs) >= MIN_ACTIVITY_ROOT_COCS:
            num_activity_cocs += 1
    
    if len(root_amp_medians) < min_coc_n or num_activity_cocs < MIN_ACTIVITY:
        return []

    # Determine whether root times need to be split into different roots based on amp_median modality
    root_amp_medians = np.array(root_amp_medians)
    dip, pval = diptest(root_amp_medians)
    if pval >= min_unimodal_p:  # amp_medians are unimodal
        amps_allowed_root_times = [allowed_root_times]
    else:  # amp_medians are not unimodal
        root_amp_medians = root_amp_medians.reshape(-1, 1)  # Reshape to (n_samples, n_features)
        best_score = np.inf
        best_gmm = None
        for n_components in range(2, max_n_components_amp+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            gmm.fit(root_amp_medians)
            score = gmm.bic(root_amp_medians)
            if score < best_score:
                best_score = score
                best_gmm = gmm
            # print(n_components, score)
        amps_allowed_root_times = [[] for _ in range(best_gmm.n_components)]
        amps_allowed_root_times_activities = [0] * best_gmm.n_components
        for label, time in zip(best_gmm.predict(root_amp_medians), coc_dict.keys()):
            amps_allowed_root_times[label].append(time)
            if len(coc_dict[time]) >= MIN_ACTIVITY_ROOT_COCS:
                amps_allowed_root_times_activities[label] += 1
        amps_allowed_root_times = [times for times, count in zip(amps_allowed_root_times, amps_allowed_root_times_activities) if count >= MIN_ACTIVITY]
        if len(amps_allowed_root_times) == 0:
            return []
            
    if verbose:
        print(f"{len(allowed_root_times)} cocs total")
        
        if len(amps_allowed_root_times) > 1:
            print(f"\nMultimodal amp/median with p-value: {pval:.3f}")
            print(f"Dividing root cocs into amp/median groups with #cocs:")
            print(f"{[len(t) for t in amps_allowed_root_times]}")
            # plt.hist(root_amp_medians, bins=30)
            # plt.show()
            # assert False
        else:
            print(f"\nUnimodal amp/median with p-value: {pval:.3f}")

    # print(f"root amps:")
    # plt.hist(root_amp_medians, bins=30)
    # plt.show()

    # # For faster testing
    # utils.pickle_dump(amps_allowed_root_times, "/data/MEAprojects/PropSignal/MAX_DELETE_ME/amps_allowed_root_times.pickle") 
    # utils.pickle_dump(coc_dict, "/data/MEAprojects/PropSignal/MAX_DELETE_ME/coc_dict.pickle")
    # amps_allowed_root_times = utils.pickle_load("/data/MEAprojects/PropSignal/MAX_DELETE_ME/amps_allowed_root_times.pickle")
    # coc_dict = utils.pickle_load("/data/MEAprojects/PropSignal/MAX_DELETE_ME/coc_dict.pickle")

    all_coc_clusters = []
    root_cluster = CocCluster(root_elec, {root_elec}, [])
    for allowed_root_times in amps_allowed_root_times:
        if verbose and len(amps_allowed_root_times) > 1:       
            print(f"-"*50)
            print(f"Starting on amp/median group with {len(allowed_root_times)} cocs")
        allowed_root_times = set(allowed_root_times)

        # patience_counter = 0
        # Compare root to each comp elec
        for c in range(len(comp_elecs)):    
            if verbose: 
                print(f"\nComparing to elec {comp_elecs[c]}, loc: {ELEC_LOCS[comp_elecs[c]]}")
                
            # Grow tree on root-comp elec pair
            coc_clusters = branch_coc_cluster(root_cluster, comp_elecs[c:],  # Elecs before c would have already been compared to root-comp elec pair
                                              coc_dict, allowed_root_times=allowed_root_times,
                                              min_unimodal_p=min_unimodal_p, max_n_components=max_n_components_latency,
                                              max_latency_diff=max_latency_diff, 
                                              min_coc_n=min_coc_n, min_coc_p=min_coc_p,
                                              min_extend_comp_p=min_extend_comp_p,
                                              patience=Patience(root_elec, elec_patience),
                                              verbose=False)            
            for cluster in coc_clusters:
                allowed_root_times.difference_update(cluster._spike_train)
                all_coc_clusters.append(cluster)
            
            if verbose:
                print(f"Found {len(coc_clusters)} clusters")
                print(f"{len(allowed_root_times)} cocs remaining")
                                
            if len(allowed_root_times) < min_coc_n:
                if verbose:
                    print(f"\nEnding early because too few cocs remaining")
                break
            
            # if len(coc_clusters) == 0:
            #     patience_counter += 1
            # else:
            #     patience_counter = 0

            # if verbose:
            #     print(f"Patience counter: {patience_counter}/{elec_patience}")
                
            # if patience_counter == elec_patience:
            #     if verbose:
            #         print(f"\nStopping early due to patience")
            #     break
                
        if verbose:
            print(f"\nTotal: {len(all_coc_clusters)} clusters")
        
    # region Split coc_clusters based on root amp medians
    if not SPLIT_ROOT_AMPS_AGAIN:
        return all_coc_clusters
    
    all_split_coc_clusters = []
    for i, cluster in enumerate(all_coc_clusters):
        root_amp_medians = np.array([root_time_to_amp_median[time] for time in cluster._spike_train])
        dip, pval = diptest(root_amp_medians)
        if pval >= min_unimodal_p:  # root_amp_medians are unimodal
            all_split_coc_clusters.append(cluster)
            continue
        if verbose:
            print(f"Cluster {i}: p-val={pval:.4f}")
        
        root_amp_medians = root_amp_medians.reshape(-1, 1)  # Reshape to (n_samples, n_features)
        best_score = np.inf
        best_gmm = None
        for n_components in range(2, max_n_components_amp+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            gmm.fit(root_amp_medians)
            score = gmm.bic(root_amp_medians)
            if score < best_score:
                best_score = score
                best_gmm = gmm
            # print(n_components, score)
        split_coc_clusters = [cluster.split(cluster.root_elec, []) for _ in range(best_gmm.n_components)]  # split() using cluster.root_elec so that no elec is added to splitting elecs
        for label, time in zip(best_gmm.predict(root_amp_medians), cluster._spike_train):
            split_coc_clusters[label]._spike_train.append(time)
        for cluster in split_coc_clusters:
            if len(cluster._spike_train) >= min_coc_n:
                all_split_coc_clusters.append(cluster)
        if verbose:
            print(f"Split cluster {i} into {len(split_coc_clusters)} clusters")
    # endregion
        
    return all_split_coc_clusters

    # Show all root times and remaining root times after forming clusters
    # unit = Unit(0, list({root_time for root_time, root_amp in coc_dict.keys()}), root_elec, RECORDING)
    # amp_kwargs, prob_kwargs = plot_elec_probs(unit)
    # plt.show()

    # unit = Unit(0, list(allowed_root_times), root_elec, RECORDING)
    # plot_elec_probs(unit, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
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

def setup_coc_clusters(coc_clusters, verbose=False):
    # Set important data needed for merging and other analyses
    if verbose:
        coc_clusters = tqdm(coc_clusters)
    
    for cluster in coc_clusters:        
        setup_cluster(cluster)

def setup_cluster(cluster,
                  n_cocs=None): 
    """    
    Parameters:
    n_cocs:
        If not None, setup cluster using randomly selected n_cocs
    
    Previous version
    elec_prob_thresh=0.1:
        Prob on elec needs to cross this to count as part of trunk
    rel_to_closest_elecs=3:
        Set relative amplitudes relative to mean amp of rel_to_closet_elecs elecs
    """
    n_before = N_BEFORE
    n_after = N_AFTER
    
    # root_elec = cluster.root_elecs[0]
    # all_elecs = [root_elec] + get_nearby_elecs(root_elec, max_elec_dist)
    array_elecs = range(NUM_ELECS)
    
    # Select random cocs
    spike_train = cluster.spike_train
    if n_cocs is not None and n_cocs < len(spike_train):
        spike_train = np.random.choice(spike_train, n_cocs, replace=False)

    # Start extracting stats
    # sum_elec_probs = np.zeros(NUM_ELECS, "float32")  # using mean # (n_elecs,)
    all_elec_probs = []  # (n_cocs, n_elecs)
    all_latencies = []  # (n_cocs, n_elecs)
    all_amp_medians = []  # (n_cocs, n_elecs)
    for time in spike_train:
        # Get elec probs
        output_frame = rec_ms_to_output_frame(time)
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
        # sum_elec_probs += sigmoid(np.max(output_window, axis=1))
        all_elec_probs.append(sigmoid(np.max(output_window, axis=1)))
        
        # Get latencies
        latencies = np.argmax(output_window, axis=1) - this_n_before
        all_latencies.append(latencies)
        
        # Get amp/medians
        rec_frame = round(time * SAMP_FREQ)
        pre_means, pre_medians = calc_pre_mean_median(max(0, rec_frame-n_before))        
        amps = np.abs(TRACES[array_elecs, rec_frame + latencies] - pre_means)
        amp_medians = amps / pre_medians
        all_amp_medians.append(amp_medians)
    
    # Set stats (all for all electroes in array, but store values for self.comp_elecs for fast comparison for assigning spikes)    
    
    # When elecs are only based on inner and outer radius
    # elecs = [root_elec] + get_nearby_elecs(root_elec, max_elec_dist)  # Store sliced array for fast comparision with intraelec merging and assigning spikes
    # cluster.elecs = elecs
    
    # all_elec_probs = sum_elec_probs / len(spike_train) 
    cluster.every_elec_prob = np.array(all_elec_probs).T  # (n_elecs, n_cocs)
    all_elec_probs = np.median(all_elec_probs, axis=0)
    all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
    cluster.all_elec_probs = all_elec_probs  # (n_elecs)
    
    cluster.every_latency = np.array(all_latencies).T  # (n_elecs, n_cocs)
    cluster.all_latencies = np.mean(all_latencies, axis=0)
    # cluster.latencies = cluster.all_latencies[comp_elecs[1:]]  # Don't include root elec since always 0

    cluster.every_amp_median = np.array(all_amp_medians).T  # (n_elecs, n_cocs)
    cluster.all_amp_medians = np.mean(all_amp_medians, axis=0)
    cluster.root_to_amp_median_std = {root: np.std(cluster.every_amp_median[root, :], ddof=1) for root in cluster.root_elecs}
    cluster.root_amp_median = cluster.all_amp_medians[cluster.root_elec]
    # cluster.amp_medians = cluster.all_amp_medians[comp_elecs]
    
    cluster.formation_spike_train = cluster.spike_train
    
    setup_elec_stats(cluster)
    
    # region Previous version (stats are based on mean footprint, not mean of individual spikes)
    # # Set important data needed for merging, assigning spikes, and other analyses
    # all_elec_probs = extract_detection_probs(cluster)  # (n_cocs, n_chans, n_samples)
    # elec_probs = np.mean(all_elec_probs, axis=0)  # (n_chans, n_samples)
    
    # # Find probabilities used for elec weighting
    # elec_weight_probs = []
    # for probs in elec_probs:  # 1it for each elec. probs: (n_samples)
    #     peak = np.argmax(probs)
    #     elec_weight_probs.append(np.sum(probs[peak-1:peak+2]))
    
    # # Needed for latencies and amplitudes
    # waveforms = extract_waveforms(cluster)
    
    # latencies = np.argmax(elec_probs, axis=1)
    # # latencies = np.argmin(np.mean(waveforms, axis=0), axis=1)
    # cluster.latencies = latencies - elec_probs.shape[1] // 2 # in frames
    
    # # Save for plotting
    # cluster.all_elec_probs = all_elec_probs  
    # cluster.elec_probs = elec_probs
    
    # # Save for merging
    # cluster.elec_weight_probs = np.array(elec_weight_probs)  # (n_chans,)
    # cluster.amplitudes = get_amp_dist(cluster)
    
    # # Save for assigning spikes
    # elecs = get_merge_elecs(cluster.root_elecs[0])
    # elec_weight_probs = cluster.elec_weight_probs[elecs]
    # cluster.elec_weights = elec_weight_probs / np.sum(elec_weight_probs)
    # # cluster.main_elecs = np.flatnonzero(elec_weight_probs >= elec_prob_thresh)
    # cluster.main_elecs = np.flatnonzero(np.max(cluster.elec_probs[elecs], axis=1) >= elec_prob_thresh)
    # cluster.elecs = elecs  # Elecs to watch for comparing latencies and rel amps
    
    # # cluster.elecs = np.flatnonzero(np.max(elec_probs, axis=1)>=prob_thresh)  # No longer needed
    
    # wf_amps = waveforms[:, range(waveforms.shape[1]), (latencies).astype(int)]  # (n_wfs, n_chans)
    # mean_amps = np.abs(np.mean(wf_amps, axis=0))
    
    # cluster.waveforms = waveforms
    # cluster.mean_amps = mean_amps

    # # Save for assigning spikes to increase speed
    # cluster.rel_amps = mean_amps / np.mean(mean_amps[cluster.elecs[:rel_to_closest_elecs]])
    # cluster.latencies_elecs = cluster.latencies[cluster.elecs]
    # cluster.rel_amps_elecs = cluster.rel_amps[cluster.elecs]
    # endregion


def setup_elec_stats(cluster):
    """
    Set cluster.loose_elecs, cluster.inner_loose_elecs, cluster.root_to_stats
    and the root elec
    
    cluster.root_to_stats = {root_elec: [comp_elecs, elec_probs, latencies, amp_medians]}
        Stats are from all_stat[comp_elecs]. Each root_elec has same comp_elecs, but comp_elecs[0] = root_elec
    """
    # root_elec_amp_medians = cluster.all_amp_medians[cluster.root_elecs]
    # root_elec_idx = np.argmax(root_elec_amp_medians)
    # cluster.root_elec = cluster.root_elecs[root_elec_idx]
    # cluster.root_amp_median = root_elec_amp_medians[root_elec_idx]
    
    
    # Find elecs
    cluster.loose_elecs = []
    for elec in np.flatnonzero(cluster.all_elec_probs >= LOOSE_THRESH):
        for split_elec in cluster.split_elecs:
            if calc_elec_dist(elec, split_elec) <= INNER_RADIUS:
                cluster.loose_elecs.append(elec)
                break
    
    cluster.inner_loose_elecs = []
    comp_elecs = set(cluster.loose_elecs)  # set() to prevent an elec being added more than once
    # Find inner_loose_elecs and comp_elecs
    for loose_elec in cluster.loose_elecs:
        # Check if loose elec within INNER_RADIUS of any inner elec to be a inner_loose_elec
        for root_elec in cluster.root_elecs:
            if calc_elec_dist(root_elec, loose_elec) <= INNER_RADIUS:
                cluster.inner_loose_elecs.append(loose_elec)
                break  # Add loose_elec only once
        # Add elec's inner elecs to comp_elecs
        for elec in ALL_CLOSEST_ELECS[loose_elec]:
            if calc_elec_dist(elec, loose_elec) <= INNER_RADIUS:
                comp_elecs.add(elec)
    cluster.min_loose_detections = max(MIN_LOOSE_DETECTIONS_N, MIN_LOOSE_DETECTIONS_R * len(cluster.loose_elecs))
                
    # For each root elec, make separate comp_elecs so that first elec is root_elec (needed for fast access to compare latencies since root elec should not be considered)
    # This is for fast indexing for assigning spikes
    cluster.root_to_stats = {}
    for root_elec in cluster.root_elecs[::-1]:  # Do it in reverse order so that comp_elecs will be set for root_elecs[0] for rest of function
        comp_elecs = [root_elec] + [elec for elec in comp_elecs if elec != root_elec]        
        latencies = cluster.all_latencies[comp_elecs]
        cluster.root_to_stats[root_elec] = (
            comp_elecs, 
            cluster.all_elec_probs[comp_elecs],
            latencies[1:] - latencies[0],  # For individual spikes due to variations in latency, offsetting latency like this may not be accurate. But averaged over hundreds of spikes, it should be fine
            cluster.all_amp_medians[comp_elecs],
            cluster.root_to_amp_median_std[root_elec]
        )
    cluster.comp_elecs = comp_elecs

def get_amp_medians(cluster, root_elec=None, n_cocs=None, use_formation_spike_train=False):
    """
    Only on root elec
    """
    if root_elec is None:
        root_elec = cluster.root_elecs[0]
        
    if use_formation_spike_train:
        spike_train = cluster.formation_spike_train
    else:
        spike_train = cluster.spike_train
    
    all_amp_medians = []
    if n_cocs is not None and n_cocs < len(spike_train):
        spike_train = np.random.choice(spike_train, n_cocs, replace=False)
    for time in spike_train:
        rec_frame = round(time * SAMP_FREQ)
        amp = TRACES[root_elec, rec_frame]
        # window_medians = calc_window_medians(time, [root_elec])[0]
        pre_mean, pre_median = calc_pre_mean_median(rec_frame, [root_elec])
        all_amp_medians.append((np.abs(amp-pre_mean) / pre_median))
    return np.array(all_amp_medians)
# endregion


# region Merge
def combine_means(x, n, y, m):
    """
    Params:
    x: 
        Mean of sample 1
    n:
        #elements in sample 1
    y:
        Mean of sample 2
    m:
        #elements in sample 2
    
    Returns:
    Mean of combining two samples
    """
    return (x*n + y*m)/(n+m)

class Merge:
    # Represent a CocCluster merge
    def __init__(self, cluster_i, cluster_j) -> None:
        self.cluster_i = cluster_i
        self.cluster_j = cluster_j
        # self.closest_elecs = cluster_i.elecs  # Should not really matter whose elecs since clusters should be close together
        
        # i_probs = cluster_i.elec_weight_probs
        # j_probs = cluster_j.elec_weight_probs
        # # self.elec_probs = (i_probs + j_probs) / 2  # /2 to find average between two elecs
        # self.elec_probs = np.max(np.vstack((i_probs, j_probs)), axis=0)  # Max between two elecs
        
    # def score_latencies(self):

        
    #     return latency_diff
        
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

    # def OLD_merge(self):
    #     # region Combine spike trains, but if both clusters detect same spike, only add once
    #     # spike_train_i = self.cluster_i.spike_train
    #     # spike_train_j = self.cluster_j.spike_train
        
    #     # # all_root_amp_medians_i = self.cluster_i.all_root_amp_medians
    #     # # all_root_amp_medians_j = self.cluster_j.all_root_amp_medians
        
    #     # spike_train = [spike_train_i[0]]
    #     # # all_root_amp_medians = [all_root_amp_medians_i[0]]
    #     # i, j = 1, 0
    #     # while i < len(spike_train_i) and j < len(spike_train_j):
    #     #     spike_i, spike_j = spike_train_i[i], spike_train_j[j]
    #     #     if spike_i < spike_j:  # i is next to be added
    #     #         if np.abs(spike_train[-1] - spike_i) >= 0.1: # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
    #     #             spike_train.append(spike_i)
    #     #             # all_root_amp_medians.append(all_root_amp_medians_i[i])
    #     #         i += 1
    #     #     else:  # j is next to be added
    #     #         if np.abs(spike_train[-1] - spike_j) >= 0.1: # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
    #     #             spike_train.append(spike_j)
    #     #             # all_root_amp_medians.append(all_root_amp_medians_j[j])
    #     #         j += 1

    #     # # Append remaning elements (only one cluster's spike train can be appended due to while loop)
    #     # if i < len(spike_train_i):
    #     #     spike_train.extend(spike_train_i[i:])
    #     #     # all_root_amp_medians.extend(all_root_amp_medians_i[i:])
    #     # else:
    #     #     spike_train.extend(spike_train_j[j:])
    #     #     # all_root_amp_medians.extend(all_root_amp_medians_j[j:])
        
    #     # # Set new spike train
    #     # try:
    #     #     self.cluster_i._spike_train = spike_train
    #     # except AttributeError:
    #     #     self.cluster_i.spike_train = spike_train
    #     # self.cluster_i.all_root_amp_medians = all_root_amp_medians
    #     # endregion
        
    #     cluster_i = self.cluster_i
    #     cluster_j = self.cluster_j
        
    #     # region Update stats 
    #     n = len(cluster_i._spike_train)
    #     m = len(cluster_j._spike_train) 
        
    #     # Elec probs
    #     # all_elec_probs = combine_means(cluster_i.all_elec_probs, n, cluster_j.all_elec_probs, m)
    #     cluster_i.every_elec_prob = np.concatenate((cluster_i.every_elec_prob, cluster_j.every_elec_prob), axis=1)
    #     all_elec_probs = np.median(cluster_i.every_elec_prob, axis=1)
    #     all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
    #     cluster_i.all_elec_probs = all_elec_probs  # (n_elecs)
    
    #     # cluster_i.elec_probs = cluster_i.all_elec_probs[elecs]
        
    #     # Latencies 
    #     every_latency = cluster_j.every_latency
    #     # all_latencies = cluster_j.all_latencies 
    #     if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i)
    #         every_latency -= every_latency[cluster_i.root_elecs[0], :]
    #         # all_latencies = np.mean(every_latency, axis=1)
    #     cluster_i.every_latency = np.concatenate((cluster_i.every_latency, every_latency), axis=1)
    #     # cluster_i.all_latencies = combine_means(cluster_i.all_latencies, n, all_latencies, m)
    #     cluster_i.all_latencies = np.median(cluster_i.every_latency, axis=1)
    #     # cluster_i.all_latencies = np.mean(cluster_i.every_latency, axis=1)
    #     # cluster_i.latencies = cluster_i.all_latencies[elecs[1:]]
        
    #     # Amp/medians
    #     cluster_i.every_amp_median = np.concatenate((cluster_i.every_amp_median, cluster_j.every_amp_median), axis=1)
    #     cluster_i.all_amp_medians = np.median(cluster_i.every_amp_median, axis=1)
    #     # cluster_i.all_amp_medians = combine_means(cluster_i.all_amp_medians, n, cluster_j.all_amp_medians, m)
    #     # luster_i.amp_medians = cluster_i.all_amp_medians[elecs]
    #     # endregion
        
    #     # try:
    #     cluster_i._spike_train.extend(cluster_j._spike_train)
    #     # except AttributeError:
    #     #     self.cluster_i.spike_train.extend(self.cluster_j.spike_train)
    #     #     self.cluster_i.spike_train = np.sort(self.cluster_i.spike_train)  # If accessing spike train this way, keep it sorted

    #     # Update root elecs
    #     cluster_i_elecs = set(cluster_i.root_elecs)
    #     for elec in cluster_j.root_elecs:
    #         if elec not in cluster_i_elecs:
    #             cluster_i.root_elecs.append(elec)
            
    #     setup_elec_stats(cluster_i)
    #     # setup_cluster(self.cluster_i)  # Update stats
        
    #     return cluster_j  # Return to update dead_clusters

    def merge(self):              
        # region Combine spike trains, but if both clusters detect same spike, only add once
        if self.cluster_i.root_amp_median >= self.cluster_j.root_amp_median:
            cluster_i = self.cluster_i
            cluster_j = self.cluster_j
        else:
            cluster_i = self.cluster_j
            cluster_j = self.cluster_i
            self.cluster_i = cluster_i
            self.cluster_j = cluster_j

        spike_train_i = cluster_i.spike_train       
        spike_train_j = cluster_j.spike_train
        
        spike_train = [spike_train_i[0]]        
        every_elec_prob = [cluster_i.every_elec_prob[:, 0]]
        every_latency = [cluster_i.every_latency[:, 0]]
        every_amp_median = [cluster_i.every_amp_median[:, 0]]
        
        i, j = 1, 0
        while i < len(spike_train_i) and j < len(spike_train_j):
            spike_i, spike_j = spike_train_i[i], spike_train_j[j]
            if spike_i < spike_j:  # i is next to be added
                if spike_i - spike_train[-1] > OVERLAP_TIME: # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_i)
                    every_elec_prob.append(cluster_i.every_elec_prob[:, i])
                    every_latency.append(cluster_i.every_latency[:, i])
                    every_amp_median.append(cluster_i.every_amp_median[:, i])
                i += 1
            else:  # j is next to be added
                if spike_j - spike_train[-1] > OVERLAP_TIME: # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_j)
                    every_elec_prob.append(cluster_j.every_elec_prob[:, j])
                    
                    latency = cluster_j.every_latency[:, j]
                    if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i
                        latency = latency - latency[cluster_i.root_elecs[0]]
                    every_latency.append(latency)
                    
                    every_amp_median.append(cluster_j.every_amp_median[:, j])
                j += 1

        # Append remaning elements (only one cluster's spike train can be appended due to while loop)
        if i < len(spike_train_i):
            spike_train.extend(spike_train_i[i:])
            every_elec_prob.extend(cluster_i.every_elec_prob[:, i:].T)
            every_latency.extend(cluster_i.every_latency[:, i:].T)
            every_amp_median.extend(cluster_i.every_amp_median[:, i:].T)
        else:
            spike_train.extend(spike_train_j[j:])
            every_elec_prob.extend(cluster_j.every_elec_prob[:, j:].T)
            every_latency.extend(cluster_j.every_latency[:, j:].T)
            every_amp_median.extend(cluster_j.every_amp_median[:, j:].T)
        
        # Set new spike train
        # try:
        #     self.cluster_i._spike_train = spike_train
        # except AttributeError:
        #     self.cluster_i.spike_train = spike_train
        cluster_i._spike_train = spike_train
        # endregion
        
        # region Update stats 
        # n = len(cluster_i._spike_train)
        # m = len(cluster_j._spike_train) 
        
        # Update root elecs
        cluster_i_elecs = set(cluster_i.root_elecs)
        for elec in cluster_j.root_elecs:
            if elec not in cluster_i_elecs:
                cluster_i.root_elecs.append(elec)
                
        # Elec probs
        # all_elec_probs = combine_means(cluster_i.all_elec_probs, n, cluster_j.all_elec_probs, m)
        # cluster_i.every_elec_prob = np.concatenate((cluster_i.every_elec_prob, cluster_j.every_elec_prob), axis=1)
        cluster_i.every_elec_prob = np.vstack(every_elec_prob).T
        all_elec_probs = np.median(cluster_i.every_elec_prob, axis=1)
        all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
        cluster_i.all_elec_probs = all_elec_probs  # (n_elecs)
    
        # cluster_i.elec_probs = cluster_i.all_elec_probs[elecs]
        
        # Latencies 
        # every_latency = cluster_j.every_latency
        # # all_latencies = cluster_j.all_latencies 
        # if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i)
        #     every_latency -= every_latency[cluster_i.root_elecs[0], :]
        #     # all_latencies = np.mean(every_latency, axis=1)
        # cluster_i.every_latency = np.concatenate((cluster_i.every_latency, every_latency), axis=1)
        # cluster_i.all_latencies = combine_means(cluster_i.all_latencies, n, all_latencies, m)
        cluster_i.every_latency = np.vstack(every_latency).T
        # cluster_i.all_latencies = np.median(cluster_i.every_latency, axis=1)
        cluster_i.all_latencies = np.mean(cluster_i.every_latency, axis=1)
        # cluster_i.latencies = cluster_i.all_latencies[elecs[1:]]
        
        # Amp/medians
        # cluster_i.every_amp_median = np.concatenate((cluster_i.every_amp_median, cluster_j.every_amp_median), axis=1)
        cluster_i.every_amp_median = np.vstack(every_amp_median).T
        cluster_i.all_amp_medians = np.mean(cluster_i.every_amp_median, axis=1)
        cluster_i.root_to_amp_median_std = {root: np.std(cluster_i.every_amp_median[root, :], ddof=1) for root in cluster_i.root_elecs}
        cluster_i.root_amp_median = cluster_i.all_amp_medians[cluster_i.root_elec]
        # cluster_i.all_amp_medians = combine_means(cluster_i.all_amp_medians, n, cluster_j.all_amp_medians, m)
        # luster_i.amp_medians = cluster_i.all_amp_medians[elecs]
        # endregion
                
        # try:
        # cluster_i._spike_train.extend(cluster_j._spike_train)
        # except AttributeError:
        #     self.cluster_i.spike_train.extend(self.cluster_j.spike_train)
        #     self.cluster_i.spike_train = np.sort(self.cluster_i.spike_train)  # If accessing spike train this way, keep it sorted
                
            
        setup_elec_stats(cluster_i)
        # setup_cluster(self.cluster_i)  # Update stats
        
        return cluster_j  # Return to update dead_clusters

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
    print(f"Latency diff: {merge.latency_diff:.2f}. Amp median diff: {merge.amp_median_diff:.2f}")
    print(f"Amp dist p-value {merge.dip_p:.4f}")

    print(f"#spikes:")
    num_overlaps = Comparison.count_matching_events(cluster_i.spike_train, cluster_j.spike_train, delta=OVERLAP_TIME)
    # num_overlaps = len(set(cluster_i.spike_train).intersection(cluster_j.spike_train))
    print(f"Merge base: {len(cluster_i.spike_train)}, Add: {len(cluster_j.spike_train)}, Overlaps: {num_overlaps}")
    
    # Find ISI violations after merging
    # cat_spikes = np.sort(np.concatenate((cluster_i.spike_train, cluster_j.spike_train)))
    # diff = np.diff(cat_spikes)
    # num_viols = np.sum(diff <= 1.5)
    # print(f"ISI viols: {num_viols}")
    
    # Plot footprints
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster_i, idx=cluster_i.idx)
    # plt.show()
    # plot_elec_probs(cluster_j, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs, idx=cluster_j.idx)
    # plt.show()   
    
    # # Plot amp distribution
    # all_amps = get_amp_dist(cluster_i) + get_amp_dist(cluster_j)
    # plot_amp_dist(np.array(all_amps))
    # plt.show()

def merge_coc_clusters(coc_clusters,
                       max_latency_diff=2.51, max_amp_median_diff=0.35, min_amp_dist_p=0.075,
                       
                       auto_setup_coc_clusters=True, verbose=False):
    """
    Parameters
    ----------
    max_root_elec_dist:
        For merge to happen, root elecs have to be within max_root_elec_dist
    max_merge_elec_dist:
        Only compare electrodes within max_merge_elec_dist micrometers of root elec
    
    """
    
    max_latency_diff = MAX_LATENCY_DIFF_SEQUENCES
    max_amp_median_diff = MAX_AMP_MEDIAN_DIFF_SEQUENCES
    min_amp_dist_p = MIN_AMP_DIST_P
    
    if auto_setup_coc_clusters:
        setup_coc_clusters(coc_clusters, verbose=verbose)

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
        
    # Find best merge
    
    # for cluster in coc_clusters:
    #     cluster.root_amps = get_amp_medians(cluster, root_elec=cluster.root_elecs[0])
    
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
                
                # Check if root elecs are close enough (find max dist between i_root_elecs and j_root_elecs)
                max_dist = 0
                for root_i in cluster_i.root_elecs:
                    for root_j in cluster_j.root_elecs:
                        max_dist = max(max_dist, calc_elec_dist(root_i, root_j))
                        if max_dist >= INNER_RADIUS:
                            break
                    else:
                        continue
                    break
                if max_dist >= INNER_RADIUS:
                    continue
                    
                # Check if enough overlapping loose electrodes
                total_loose = len(set(cluster_i.loose_elecs).union(cluster_j.loose_elecs))
                num_loose_overlaps = len(set(cluster_i.loose_elecs).intersection(cluster_j.loose_elecs))
                # print(num_loose_overlaps, total_loose, num_loose_overlaps/total_loose)
                if num_loose_overlaps < MIN_LOOSE_DETECTIONS_N or num_loose_overlaps/total_loose < MIN_LOOSE_DETECTIONS_R:
                    continue
                num_inner_loose_overlaps = len(set(cluster_i.inner_loose_elecs).intersection(cluster_j.inner_loose_elecs))
                if num_inner_loose_overlaps < MIN_INNER_LOOSE_DETECTIONS:
                    continue
                    
                # if calc_elec_dist(cluster_i.root_elecs[0], cluster_j.root_elecs[0]) > max_root_elec_dist:
                #     continue
        
                # Get elecs for comparison (do it this way so comp_elecs[0] is root elec)
                if cluster_i.root_amp_median >= cluster_j.root_amp_median: ## Find which cluster's root amp to use (use one with higher amplitude)
                    i_comp_elecs = cluster_i.comp_elecs
                    i_comp_elecs_set = set(i_comp_elecs)
                    comp_elecs = i_comp_elecs + [elec for elec in cluster_j.comp_elecs if elec not in i_comp_elecs_set]
                else:
                    j_comp_elecs = cluster_j.comp_elecs
                    j_comp_elecs_set = set(j_comp_elecs)
                    comp_elecs = j_comp_elecs + [elec for elec in cluster_i.comp_elecs if elec not in j_comp_elecs_set]
                                
                # Get elec probs
                i_elec_probs = cluster_i.all_elec_probs[comp_elecs]
                j_elec_probs = cluster_j.all_elec_probs[comp_elecs]
                
                # Compare latencies                 
                i_latencies = cluster_i.all_latencies[comp_elecs][1:]
                j_latencies = cluster_j.all_latencies[comp_elecs][1:] - cluster_j.all_latencies[comp_elecs[0]]  # Relative to same electrode as cluster_i
                elec_weights = get_elec_weights(i_elec_probs, j_elec_probs, for_latencies=True)
                latency_diff = np.abs(i_latencies - j_latencies)
                latency_diff = np.clip(latency_diff, a_min=None, a_max=CLIP_LATENCY_DIFF)
                latency_diff = np.sum(latency_diff * elec_weights)
                # latency_diff = np.sum(np.abs(i_latencies - j_latencies) * elec_weights)
                if latency_diff > max_latency_diff:
                    continue
                
                # Compare amp/medians
                i_amp_medians = cluster_i.all_amp_medians[comp_elecs]
                j_amp_medians = cluster_j.all_amp_medians[comp_elecs]                
                elec_weights = get_elec_weights(i_elec_probs, j_elec_probs, for_latencies=False)
                amp_median_div = (i_amp_medians + j_amp_medians) / 2
                amp_median_diff = np.abs((i_amp_medians - j_amp_medians)) / amp_median_div
                amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=CLIP_AMP_MEDIAN_DIFF)
                amp_median_diff = np.sum(amp_median_diff * elec_weights)
                if amp_median_diff > max_amp_median_diff:
                    continue                
                
                # Test if merge is bimodal
                are_unimodal = True
                for root_elec in set(cluster_i.root_elecs + cluster_j.root_elecs):
                    root_amps_i = cluster_i.every_amp_median[root_elec, :] # get_amp_medians(cluster_i, root_elec=root_elec)
                    root_amps_j = cluster_j.every_amp_median[root_elec, :] # get_amp_medians(cluster_j, root_elec=root_elec)
                    dip, pval = diptest(np.concatenate([root_amps_i, root_amps_j]))
                    if pval < min_amp_dist_p:
                        are_unimodal = False
                        break
                if not are_unimodal:
                    continue
                                
                # Calculate quality of merge
                cur_merge = Merge(cluster_i, cluster_j)
                score = latency_diff / max_latency_diff + amp_median_diff / max_amp_median_diff 
                if best_merge is None or score < best_merge.score:
                    best_merge = cur_merge
                    best_merge.score = score
                    best_merge.latency_diff = latency_diff
                    best_merge.amp_median_diff = amp_median_diff
                    best_merge.dip_p = pval
                
                # if not cur_merge.can_merge(max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
                #     if verbose and (best_unmerge is None or cur_merge.is_better(best_unmerge)):
                #         best_unmerge = cur_merge
                #     continue
                # if best_merge is None or cur_merge.is_better(best_merge, max_latency_diff, max_rel_amp_diff):
                #     best_merge = cur_merge
                    
        # If no merges are good enough
        if best_merge is None:
        # if not best_merge.can_merge(max_latency_diff, max_rel_amp_diff):
            # if verbose:
            #     print(f"\nNo merge found. Next best merge:")
            #     merge_verbose(best_unmerge, update_history=False)
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

def get_elec_weights(elec_probs_i, elec_probs_j, for_latencies=False, min_prob=0.03):
    """
    Need to do this before scoring match with :other:
    
    for_latencies:
        If True, only include elec_probs[1:] 
        Else (if for amp medians), multiply elec_probs[0] by 2 to increase weight on root elec
    min_r:
        Elec probabilities below min_r are set to 0
    """
    if for_latencies:
        elec_probs = (elec_probs_i[1:] + elec_probs_j[1:]) / 2
    else:
        elec_probs = (elec_probs_i + elec_probs_j) / 2
        # elec_probs[0] *= 2  # To weight root elec more
    # elec_probs[elec_probs < min_prob] = 0  
    return elec_probs / np.sum(elec_probs)

def assign_spikes(all_units, time_frame, interelec=False,
                  max_latency_diff=2.51, max_amp_median_diff=0.35,                    
                  overlap_dist=50, 
                  
                  only_max_amps=True,                  
                  verbose=False):
    """
    OLD 
        Spike splitting pseudocode:
        1. Have units watch root_elecs for DL detections
        2. When detection on elec, assign spike to unit with best footprint match
        If len(root_elecs)>1: unit watches multiple elecs, but ignores other elecs
        if detects a spike within Xms of another elec (prevent counting same spike multiple times)
        
        Leaves spikes in a buffer zone. Only split spikes within Xms and Xum
        Remove all in front of spike_buffer that occur too before new spike, leave the rest
        
        Attempted method: Fails. Hard to account for when a unit belongs to multiple split groups
            Rationale: Tries to account for edge case of spikes 0.1, 0.3, 0.45 when overlap_time=0.4. 
            Naive method would split 0.1 and 0.3, but perhaps 0.3 should be split with 0.45
        
    OLD
        Spike splitting pseudocode:
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
        
    CURRENT 
        Spike splitting pseudocode:
        1. Add detected spikes to buffer
        2. If newest spike added is farther than overlap_time with oldest spike, start splitting (when implemented in real time, this would be after 0.1ms pass):
            a. Select spikes that are not within overlap_time of newest spike or closer to oldest spike 
                (Unselected spikes remain in buffer)
            b. Find which spike has highest overlap score 
            c. Calculate the number of units (X) that are allowed to detect spike by extracting window of DL output probs around this spike (see code for details)
            d. For spikes within overlap_dist of best spike, assign spike to top X units with highest overlap scores and demove other units
            e. Repeat step b until no spikes remain
        
    Params:
    inter_elec:
        If True, assume all_units contains different root elecs (slower, but needed for interelec spike splitting)
    """    
    n_before = N_BEFORE
    n_after = N_AFTER
    
    max_latency_diff = MAX_LATENCY_DIFF_SPIKES
    max_amp_median_diff = MAX_AMP_MEDIAN_DIFF
    
    clip_latency_diff = CLIP_LATENCY_DIFF
    clip_amp_median_diff = CLIP_AMP_MEDIAN_DIFF
    
           
    for unit in all_units:        
        unit._spike_train = []
        # unit._spike_scores = []  # For spike splitting after assigning spikes
        unit._elec_train = []  # Indicates which root elec's detection led to spike in unit._spike_train
        
    # For ith elec, which units are watching
    elec_watchers = {}
    elec_to_seq_elecs = {}  # For each elec if stringent threshold crossing, only access data for electrodes that contain a sequence (seq_elecs)
    for unit in all_units:
        # unit.elec_to_stats = {}  # Now done in set_elec_stats - Original: Cache so don't have to keep slicing for spike match
        for elec in unit.root_elecs:
            if elec not in elec_watchers:
                elec_watchers[elec] = [unit]
            else:
                elec_watchers[elec].append(unit)                  
            
            if elec not in elec_to_seq_elecs:
                elec_to_seq_elecs[elec] = set(unit.comp_elecs)
            else:
                elec_to_seq_elecs[elec].update(unit.comp_elecs)
                
        unit.time_to_spike_match = {} # For testing with time_to_spike_match, {spike: [root_elec, num_inner_loose_elecs, num_loose_elecs, latency diff, amplitude diff, match score]}
                
    elec_to_seq_elecs_dict = {}  # For fast indexing elec_to_seq_elecs to form root_to_assign_spikes_stats for each unit
    for elec, seq_elecs in elec_to_seq_elecs.items():
        seq_elecs = list(seq_elecs)
        elec_to_seq_elecs[elec] = seq_elecs
        elec_to_seq_elecs_dict[elec] = {e: idx for idx, e in enumerate(seq_elecs)} 

    # For each unit, convert elec ind for entire array to elec ind in to elec_to_seq_elecs (comp_elecs)
    for unit in all_units:
        unit.assign_spikes_root_to_stats = {}  # root_to_stats for assigning spikes faster
        for root_elec in unit.root_elecs:
            elec_to_seq_idx = elec_to_seq_elecs_dict[root_elec]
            
            comp_elecs, elec_probs, latencies, amp_medians, amp_median_std = unit.root_to_stats[root_elec]
            comp_elecs = [elec_to_seq_idx[root_elec]] + [elec_to_seq_idx[e] for e in comp_elecs if e != root_elec]  # root_elec is first elec
            
            loose_elecs = [elec_to_seq_idx[e] for e in unit.loose_elecs]
            inner_loose_elecs = [elec_to_seq_idx[e] for e in unit.inner_loose_elecs]
            
            latency_elec_weights = elec_probs[1:] / np.sum(elec_probs[1:])
            amp_elec_weights = elec_probs / np.sum(elec_probs)
            
            # For max amp requirement            
            inner_loose_elec_to_comp_elec_idx = {e: idx for idx, e in enumerate(comp_elecs)}
            inner_loose_elecs_set = {inner_loose_elec_to_comp_elec_idx[e] for e in inner_loose_elecs}  # inner_loose_elecs relative to comp_elecs ind
            
            unit.assign_spikes_root_to_stats[root_elec] = (
                loose_elecs, inner_loose_elecs, comp_elecs, 
                inner_loose_elecs_set,
                latency_elec_weights, amp_elec_weights, 
                latencies, amp_medians, amp_median_std
            )
    
    elec_to_outer_elecs = {} 
    for elec in elec_watchers:        
        elec_to_outer_elecs[elec] = [elec] + get_nearby_elecs(elec, OUTER_RADIUS)

    # Start watching for spikes
    spike_buffer = []  # Store spikes before they have been assigned
    
    all_crossings_times = [c[1] for c in ALL_CROSSINGS]
    start_idx = np.searchsorted(all_crossings_times, time_frame[0], side="left")
    end_idx = np.searchsorted(all_crossings_times, time_frame[1], side="right")
    crossings = ALL_CROSSINGS[start_idx:end_idx]
    if verbose:
        crossings = tqdm(crossings)
    
    # Consider recalculating pre amps and medians every thresh ms (actually seems to be slower)
    # thresh = 1
    # prev_time = -np.inf
    
    for elec, time, amp in crossings:
        # if time not in time_to_spike_match:  # Handle multiple elecs detecting a spike at the same time
        #     time_to_spike_match[time] = []
        
        if elec not in elec_watchers:  # No units are watching elec
            continue
        
        # if time - prev_time >= thresh:
        #     rec_frame = round(time * SAMP_FREQ)
        #     pre_means, pre_medians = calc_pre_mean_median(rec_frame-n_before)
        #     prev_time = time
        
        # inner_elecs, outer_elecs, elecs = elec_to_nearby_elecs[elec]
        seq_elecs = elec_to_seq_elecs[elec]   
        
        output_frame = rec_ms_to_output_frame(time)
        
        # region Check for all electrode array getting detection (noise spike)
        noise_probs = OUTPUTS[:, output_frame]
        if np.sum(noise_probs >= LOOSE_THRESH_LOGIT) >= MIN_ELECS_FOR_ARRAY_NOISE:
            continue
        # endregion
        
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems. TODO: Consider removing for speed and just not detecting spikes in first N_BEFORE frames
        # output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
        output_window = OUTPUTS[seq_elecs, output_frame-this_n_before:output_frame+n_after+1]
        all_elec_probs = sigmoid(np.max(output_window, axis=1))
         
        spike_num_elecs = np.sum(all_elec_probs >= LOOSE_THRESH)
         
        # Check if enough codetections
        # if np.sum(all_elec_probs[inner_elecs] >= min_coc_prob) < min_inner_cocs:
        #     continue
        
        # if np.sum(all_elec_probs[outer_elecs] >= min_coc_prob) < min_outer_cocs:
        #     continue
        
        # Intraelectrode spike splitting (all have same root_elec and therefore nearby elecs, so only extract data for this spike once)
        # elecs = elec_watchers[elec][0].elecs
        
        # Get elec probs
        # output_frame = rec_ms_to_output_frame(time)
        # this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        # output_window = OUTPUTS[elecs, output_frame-this_n_before:output_frame+n_after+1]
        # output_window = output_window[:]
        # elec_probs = sigmoid(np.max(output_window, axis=1))
        
        # Get latencies
        all_latencies = np.argmax(output_window, axis=1) - this_n_before 
        
        # Get amp/medians
        rec_frame = round(time * SAMP_FREQ)
        # pre_means, pre_medians = calc_pre_mean_median(rec_frame-n_before)
        # amps = np.abs(TRACES[array_elecs, rec_frame + all_latencies] - pre_means)
        pre_means, pre_medians = calc_pre_mean_median(rec_frame-n_before, seq_elecs)
        amps = np.abs(TRACES[seq_elecs, rec_frame + all_latencies] - pre_means)
        all_amp_medians = amps / pre_medians
        
        if PRE_INTERELEC_ROOT_MAX_AMP_ONLY: # only_max_amps:  # 1/6/24 max-amp requirement is now unique to each sequence and is always required
            outer_elecs = elec_to_outer_elecs[elec]
            pre_means, pre_medians = calc_pre_mean_median(rec_frame-n_before, outer_elecs)
            outer_elecs_traces = TRACES[outer_elecs, rec_frame-this_n_before:rec_frame+n_after+1]
            amps = np.min(outer_elecs_traces, axis=1)
            amps = (amps - pre_means) / pre_medians
            if np.argmin(amps) != 0:
                continue
        
        # Now done separately for each seq :  Don't include root elec in latencies since always 0  (Needed it earlier for getting amp/medians)
        # latencies = latencies[1:]
        
        best_unit = None
        best_score = np.inf
        for unit in elec_watchers[elec]:
            # region Previous code (more modular but slower)
            # elecs_r, latency_diff, rel_amp_diff = get_spike_match(unit, time,
            #                                                       max_latency_diff=max_latency_diff, min_coacs_r=min_coacs_r)
            
            # if elecs_r >= min_coacs_r and latency_diff <= max_latency_diff and rel_amp_diff <= max_rel_amp_diff:
            #     # Score spike match with footprint (lower score is better)
            #     match_score = (1-elecs_r) + latency_diff / max_latency_diff + rel_amp_diff / max_rel_amp_diff  # Need to normalize different metrics
            #     if match_score < best_score:
            #         best_unit = unit
            #         best_score = match_score
            # endregion

            # region unit.root_to_stats
            # Check if enough codetections for unit
            
            unit_loose_elecs, unit_inner_loose_elecs, unit_comp_elecs, \
            unit_inner_loose_elecs_set, \
            latency_elec_weights, amp_elec_weights, \
            unit_latencies, unit_amp_medians, unit_amp_median_std = unit.assign_spikes_root_to_stats[elec]
            
            num_inner_loose_elecs = np.sum(all_elec_probs[unit_inner_loose_elecs] >= LOOSE_THRESH)
            # if num_inner_loose_elecs < MIN_INNER_LOOSE_DETECTIONS:
            #     continue
            
            # max-amp requirement
            amp_medians = all_amp_medians[unit_comp_elecs]
            if np.argmax(amp_medians) not in unit_inner_loose_elecs_set:
                continue
        
            num_loose_elecs = np.sum(all_elec_probs[unit_loose_elecs] >= LOOSE_THRESH)
            # if num_loose_elecs < unit.min_loose_detections:
            if num_loose_elecs < 2:
                # time_to_spike_match[time].append((
                #     unit, elec, num_inner_loose_elecs, num_loose_elecs, np.nan, np.nan, np.nan
                # ))
                continue

            # region Faster method, TODO: return to this implementation in final version
            # if np.sum(all_elec_probs[unit_inner_loose_elecs] >= LOOSE_THRESH) < MIN_INNER_LOOSE_DETECTIONS:
            #     continue
        
            # num_loose_elecs = np.sum(all_elec_probs[unit_loose_elecs] >= LOOSE_THRESH)
            # if num_loose_elecs < unit.min_loose_detections:
            #     continue
            # endregion
                        
            # Get unit's stats
            # unit_comp_elecs, unit_elec_probs, unit_latencies, unit_amp_medians = unit.root_to_stats[elec]
            
            # Set spike's elec_probs
            # elec_probs = all_elec_probs[unit_comp_elecs]
            
            # Compare latencies
            latencies = all_latencies[unit_comp_elecs[1:]]
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=True)
            latency_diff = np.abs(unit_latencies - latencies)
            latency_diff = np.clip(latency_diff, a_min=None, a_max=clip_latency_diff)
            latency_diff = np.sum(latency_diff * latency_elec_weights)
            # latency_diff = np.sum(np.abs(unit_latencies - latencies) * elec_weights)
            if latency_diff > max_latency_diff:
                continue
            
            # Compare amp/medians
            # amp_medians = all_amp_medians[unit_comp_elecs]
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=False)
            # amp_median_div = (unit_amp_medians + amp_medians) / 2
            amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / unit_amp_medians # / amp_median_div
            amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=clip_amp_median_diff)
            amp_median_diff = np.sum(amp_median_diff * amp_elec_weights)
            if amp_median_diff > max_amp_median_diff:
                continue
            
            amp_median_z = np.abs(amp_medians[0] - unit_amp_medians[0]) / unit_amp_median_std
            if amp_median_z > MAX_ROOT_AMP_MEDIAN_STD:
                continue
            # endregion
            
            # region unit.all_elec_probs[elecs]
            # # Compare latencies
            # unit_elec_probs = unit.all_elec_probs[elecs]
            
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=True)
            # latency_diff = np.sum(np.abs(unit.all_latencies[elecs[1:]] - latencies) * elec_weights)
            # if latency_diff > max_latency_diff:
            #     continue
            
            # # Compare amp/medians
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=False)
            # unit_amp_medians = unit.all_amp_medians[elecs]
            # amp_median_div = (unit_amp_medians + amp_medians) / 2
            # amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / amp_median_div
            # amp_median_diff = np.sum(amp_median_diff * elec_weights)
            # if amp_median_diff > max_amp_median_diff:
            #     continue
            # endregion
            
            # Calc match score
            score = (latency_diff / max_latency_diff) + (amp_median_diff / max_amp_median_diff) + (1 - num_loose_elecs / spike_num_elecs)
            if num_inner_loose_elecs >= MIN_INNER_LOOSE_DETECTIONS and num_loose_elecs >= unit.min_loose_detections:
                if latency_diff <= max_latency_diff and amp_median_diff <= max_amp_median_diff:
                    if score < best_score:
                        best_unit = unit
                        best_score = score
                
            # spike_match = (elec, num_inner_loose_elecs, num_loose_elecs, num_loose_elecs/len(unit_loose_elecs), latency_diff, amp_median_diff, score)
            # if time not in unit.time_to_spike_match:  # Need to account for if unit has more than one root elec and detects spike on each root elec
            #     unit.time_to_spike_match[time] = spike_match
            # else:
            #     # Each root elec has same num_inner_loose_elecs, num_loose_elecs
            #     if score < unit.time_to_spike_match[time][-1]:
            #         unit.time_to_spike_match[time] = spike_match
                    
            # time_to_spike_match[time].append((
            #     unit, elec, num_inner_loose_elecs, num_loose_elecs, latency_diff, amp_median_diff, score
            #     ))
            
        if best_unit is None:
            continue
        
        if interelec:
            spike_buffer.append((best_unit, time, elec, best_score))
            if len(spike_buffer) > 1 and time - spike_buffer[0][1] > OVERLAP_TIME:
                split_interelec_spike(spike_buffer, time, overlap_dist, elec_to_outer_elecs)
        else:
            best_unit._spike_train.append(time)
            # best_unit._spike_scores.append(best_score)
            
    if interelec:     
        if len(spike_buffer) > 1:
            split_interelec_spike(spike_buffer, time, overlap_dist, elec_to_outer_elecs)
        elif len(spike_buffer) == 1:
            unit, time, elec, score = spike_buffer[0]
            unit._spike_train.append(time)
            unit._elec_train.append(elec)
          
def split_interelec_spike(spike_buffer, time,
                          overlap_dist, 
                          elec_to_outer_elecs):      
    """
    spike_buffer[i] = [unit, time, elec, score]
    """
    
    # Find which spikes overlap more with earliest spike than latest, split these
    first_time = spike_buffer[0][1]
    
    overlapping_spikes = []
    while len(spike_buffer) > 0:
        old_time = spike_buffer[0][1]
        if (old_time - first_time) > (time - old_time):  # old_time is closer to new time than first_time, so it should be split with new time
            break
        spike_data = spike_buffer.pop(0)
        overlapping_spikes.append(spike_data)
    
    overlapping_spikes = sorted(overlapping_spikes, key=lambda spike_data: spike_data[3])    
    
    # Split spikes
    while len(overlapping_spikes) > 0:
        # Find best score
        # best_data = [None, None, None, np.inf]  # (unit, time, elec, score)
        # for spike_data in overlapping_spikes:
        #     if spike_data[3] < best_data[3]:
        #         best_data = spike_data
                        
        best_unit, best_time, best_elec, best_score = overlapping_spikes.pop(0)
        # if len(unit._elec_train) == 0 or \
        # time - unit._spike_train[-1] > overlap_time or unit._elec_train[-1] == elec:  # If same spike is detected by a different root_elec, do not assign spike
        #     unit._spike_train.append(time)
        #     unit._elec_train.append(elec)
        add_spike_to_unit(best_unit, best_time, best_elec, OVERLAP_TIME)
        
        # Find number of spikes that can detect spike
        output_frame = rec_ms_to_output_frame(best_time)
        this_n_before = N_BEFORE if output_frame - N_BEFORE >= 0 else output_frame  # Prevents indexing problems. TODO: Consider removing for speed and just not detecting spikes in first N_BEFORE frames
        output_window = OUTPUTS[elec_to_outer_elecs[best_elec], output_frame-this_n_before:output_frame+N_AFTER+1]
        above_thresh = output_window >= LOOSE_THRESH_LOGIT
        # Actually do computation
        peaks = above_thresh[:, :-1] & ~above_thresh[:, 1:]
        num_spikes = np.sum(peaks, axis=1) + above_thresh[:, -1]  # above_thresh[:, -1] to account for a peak not going below LOOSE_THRESH due to window ending 
        num_spikes = np.max(num_spikes) - 1  # -1 since spike already assigned to best unit
        
        # Remove all spikes within overlap_dist of best spike
        # for s in range(len(overlapping_spikes)-1, -1, -1):
        #     if calc_elec_dist(elec, overlapping_spikes[s][2]) <= overlap_dist:
        #         overlapping_spikes.pop(s)
        
        # Add spikes to top units
        for s in range(len(overlapping_spikes)-1, -1, -1):
            if calc_elec_dist(best_elec, overlapping_spikes[s][2]) <= overlap_dist:
                cur_unit, cur_time, cur_elec, cur_score = overlapping_spikes.pop(s)
                if num_spikes > 0:
                    add_spike_to_unit(cur_unit, cur_time, cur_elec, OVERLAP_TIME)
                    num_spikes -= 1          

def add_spike_to_unit(unit, time, elec, overlap_time):
    if len(unit._elec_train) == 0 or \
    time - unit._spike_train[-1] > overlap_time or unit._elec_train[-1] == elec:  # If same spike is detected by a different root_elec, do not assign spike
        unit._spike_train.append(time)
        unit._elec_train.append(elec)
        
def get_seq_spike_overlap(seq, time):
    """
    Find overlap between seq and spike
        Used fror testing and _save_sequences_plots()
    """   
    
    # Get stats for all elecs
    n_before = N_BEFORE
    n_after = N_AFTER
        
    output_frame = rec_ms_to_output_frame(time)
    this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
    output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
    all_elec_probs = sigmoid(np.max(output_window, axis=1))
    
    all_latencies = np.argmax(output_window, axis=1) - this_n_before 

    rec_frame = round(time * SAMP_FREQ)
    pre_means, pre_medians = calc_pre_mean_median(rec_frame-n_before)
    amps = np.abs(TRACES[np.arange(all_latencies.size), rec_frame + all_latencies] - pre_means)
    all_amp_medians = amps / pre_medians
    
    all_latencies -= all_latencies[seq.root_elec]  # This is necessary since kilosort's time (:param time:) may not align with RT-Sort's predicted time (even if RT-Sort's and kilosorts times are offset by 1 frame, it amounts to big difference)
    
    # Get stats for seq elecs     
    unit_comp_elecs, unit_elec_probs, unit_latencies, unit_amp_medians = seq.root_to_stats[seq.root_elec]
    
    num_inner_loose_elecs = np.sum(all_elec_probs[seq.inner_loose_elecs] >= LOOSE_THRESH)
    num_loose_elecs = np.sum(all_elec_probs[seq.loose_elecs] >= LOOSE_THRESH)
    
    latency_elec_weights = unit_elec_probs[1:] / np.sum(unit_elec_probs[1:])
    latencies = all_latencies[unit_comp_elecs[1:]]
    latency_diff = np.abs(unit_latencies - latencies)
    latency_diff = np.clip(latency_diff, a_min=None, a_max=CLIP_LATENCY_DIFF)
    latency_diff = np.sum(latency_diff * latency_elec_weights)
    
    amp_elec_weights = unit_elec_probs / np.sum(unit_elec_probs)
    amp_medians = all_amp_medians[unit_comp_elecs]
    amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / unit_amp_medians # / amp_median_div
    amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=CLIP_AMP_MEDIAN_DIFF)
    amp_median_diff = np.sum(amp_median_diff * amp_elec_weights)
    
    return num_inner_loose_elecs, num_loose_elecs, num_loose_elecs / len(seq.loose_elecs), latency_diff, amp_median_diff
    
def plot_seq_spike_overlap(seq, time, idx=0,
                           amp_kwargs=None, prob_kwargs=None):    
    num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff = get_seq_spike_overlap(seq, time)
    unit = Unit(idx, [time], seq.root_elecs[0], RECORDING)
    amp_kwargs, prob_kwargs = plot_elec_probs(unit, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.suptitle(f"Inner: {num_inner_loose_elecs}/{MIN_INNER_LOOSE_DETECTIONS}. Loose: {num_loose_elecs}/{seq.min_loose_detections:.1f}. Latency: {latency_diff:.2f} frames. Amp: {amp_median_diff*100:.1f}%")
    return amp_kwargs, prob_kwargs

# endregion


# region Form full sequences
def form_from_root(root_elec, time_frame, 
                   max_latency_diff, max_amp_median_diff,
                   verbose=False):
    # Form and merge propgations from root_elec
    coc_clusters = form_coc_clusters(root_elec, time_frame, verbose=verbose)

    if len(coc_clusters) == 0:
        return []

    setup_coc_clusters(coc_clusters)

    # Below curation is because if allowed_root_times stay together, it is possible to only have 1 loose elec (the root elec)
    coc_clusters = [cluster for cluster in coc_clusters if len(cluster.inner_loose_elecs) >= MIN_INNER_LOOSE_DETECTIONS and len(cluster.loose_elecs) >= MIN_LOOSE_DETECTIONS_N]

    assign_spikes(coc_clusters, time_frame, only_max_amps=False, #True,
                  max_latency_diff=max_latency_diff, max_amp_median_diff=max_amp_median_diff,
                  verbose=verbose)

    coc_clusters = [cluster for cluster in coc_clusters if len(cluster._spike_train) > 3]  # Need more than 3 spikes for dip test

    if len(coc_clusters) == 0:
        return []

    merges = merge_coc_clusters(coc_clusters, 
                                max_latency_diff=max_latency_diff, max_amp_median_diff=max_amp_median_diff,
                                verbose=verbose)
    
    # assign_spikes(coc_clusters, time_frame, only_max_amps=True,
    #               max_latency_diff=max_latency_diff, max_amp_median_diff=max_amp_median_diff,
    #               min_inner_cocs=min_inner_coacs, min_outer_cocs=min_outer_cocs,
    #               verbose=verbose)
    
    # setup_coc_clusters(merges)
    
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
        new_clust.idx = getattr(clust, "idx", -1)
        new_clusters.append(new_clust)
    return new_clusters


def clusters_to_units(clusters):
    # Convert CocCluster objs to Unit objs
    all_units = []
    for c, clust in enumerate(clusters):
        unit = Unit(c, clust.spike_train, clust.root_elecs[0], recording=None)  # recording=None for parallel processing
        unit.root_elecs = clust.root_elecs
        unit.mean_amps = clust.all_amp_medians
        all_units.append(unit)
    return all_units


def clusters_to_sorter(clusters):
    return SpikeSorter(RECORDING, "RT-Sort", units=clusters_to_units(clusters))
# endregion


# region Kilosort comparison
def select_prop_spikes_within_kilosort_spikes(prop_units, ks_units,
                                              max_ms_dist=0.4, max_micron_dist=100,
                                              return_ks_only_units=False):
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
    Returns:
        within_prop_units
            Prop units whose spike trains are detected by kilosort
        outside_prop_units
            Prop units whose spike trains are not detected by kilosort
    
        if return_ks_units: return ks units whose spikes trains are not detected by prop
    """
    
    # Get sorted spike times and corresponding xy-positions
    chan_locs = ELEC_LOCS
    
    all_ks_spike_times, all_ks_spike_locs, all_ks_ids = [], [], []
    for idx, unit in enumerate(ks_units):
        all_ks_spike_times.extend(unit.spike_train)
        all_ks_spike_locs += [chan_locs[unit.chan]] * len(unit.spike_train)
        all_ks_ids += [idx] * len(unit.spike_train)
        
    order = np.argsort(all_ks_spike_times)
    all_ks_spike_times, all_ks_spike_locs, all_ks_ids = np.array(all_ks_spike_times)[order], np.array(all_ks_spike_locs)[order], np.array(all_ks_ids)[order]
    matched_ks_spikes = set()  # (ks_id, ks_spike_time) in all_ks_spike_times that have already been matched to a prop spike
    
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
                ks_spike = (all_ks_spike_times[idx], all_ks_ids[idx])
                if ks_spike not in matched_ks_spikes and utils.calc_dist(*all_ks_spike_locs[idx], *loc) <= max_micron_dist:
                    within_spikes.append(spike)
                    matched_ks_spikes.add(ks_spike)
                    break
                idx += 1
            else:
                outside_spikes.append(spike)
        
        within_prop_units.append(Unit(unit.idx, np.array(within_spikes), unit.chan, unit.recording))
        outside_prop_units.append(Unit(unit.idx, np.array(outside_spikes), unit.chan, unit.recording))
        # within_prop_units.append(PropUnit(unit.df, unit.idx, np.array(within_spikes), unit.recording))
        # outside_prop_units.append(PropUnit(unit.df, unit.idx, np.array(outside_spikes), unit.recording))

    if not return_ks_only_units:
        return within_prop_units, outside_prop_units
    
    ks_only_spikes = []
    for idx, unit in enumerate(ks_units):
        spike_train = []
        for spike in unit.spike_train:
            if (spike, idx) not in matched_ks_spikes:
                spike_train.append(spike)
        ks_only_spikes.append(Unit(idx, np.array(spike_train), unit.chan, unit.recording))
    
    return within_prop_units, outside_prop_units, ks_only_spikes
# endregion


# region Kilosort as sequences (see end of si_rec9.ipynb)
def set_ks_only_spike_match_scores(ks_only_units, all_ks_sequences
                                   ):
    undetectable_seqs = []  # No inner_loose_eles
    
    for unit, seq in zip(tqdm(ks_only_units), all_ks_sequences):
        elec = unit.chan
        unit.time_to_spike_match = {}
        
        loose_elecs = seq.loose_elecs
        inner_loose_elecs = seq.inner_loose_elecs
        
        if len(inner_loose_elecs) == 0:
            undetectable_seqs.append((seq))
            for time in unit.spike_train:   
                unit.time_to_spike_match[time] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            continue
        
        for time in unit.spike_train:          
            score = np.nan # latency_diff / max_latency_diff + amp_median_diff / max_amp_median_diff + new elec overlap score
            num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff = get_seq_spike_overlap(seq, time)
            if num_loose_elecs < 2:
                unit.time_to_spike_match[time] = (elec, num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, np.nan, np.nan, np.nan)
            else:
                unit.time_to_spike_match[time] = (elec, num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff, score)

    print(f"Undetectable sequences: {[seq.idx for seq in undetectable_seqs]}")
    
def get_tp_fp_fn(metric_idx, all_ks_sequences, prop_and_ks_units, prop_only_units, ks_only_units):
    """
    Params
        include_nan
            Whether to include nan values in returned arrays
            (Needed when creating scatter plot when one metric on x-axis and another on y-axis)
    """
       
    true_positives, false_positives, false_negatives = [], [], []
    for spike_type, all_units, all_seqs in (
        (true_positives, prop_and_ks_units, all_ks_sequences),
        (false_positives, prop_only_units, all_ks_sequences),
        (false_negatives, ks_only_units, ks_only_units)
        ):
        for unit, seq in zip(all_units, all_seqs):
            for time in unit.spike_train:
                metric = seq.time_to_spike_match[time][metric_idx]
                if not np.isnan(metric):
                    spike_type.append(metric) 
    return true_positives, false_positives, false_negatives  

def get_spike_metrics(all_ks_sequences, prop_and_ks_units, prop_only_units, ks_only_units):
    """
    Same as get_tp_fp_fn but get all metrics for each spike instead of only 1
    
    Params
        include_nan
            Whether to include nan values in returned arrays
            (Needed when creating scatter plot when one metric on x-axis and another on y-axis)
    """
       
    true_positives, false_positives, false_negatives = [], [], []
    for spike_type, all_units, all_seqs in (
        (true_positives, prop_and_ks_units, all_ks_sequences),
        (false_positives, prop_only_units, all_ks_sequences),
        (false_negatives, ks_only_units, ks_only_units)
        ):
        for unit, seq in zip(all_units, all_seqs):
            for time in unit.spike_train:
                spike_type.append(seq.time_to_spike_match[time]) 
    return true_positives, false_positives, false_negatives  
# endregion



if __name__ == "__main__":
    # Parameters for recording
    RECORDING = utils.rec_si()
    CHANS_RMS = utils.chans_rms_si()

    SAMP_FREQ = RECORDING.get_sampling_frequency()
    NUM_ELECS = RECORDING.get_num_channels()
    ELEC_LOCS = RECORDING.get_channel_locations()

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
    
    TRACES_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/traces.npy"
    TRACES_FILT_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/traces_filt.npy"
    MODEL_OUTPUTS_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/outputs.npy"
    FRONT_BUFFER = 40  # Model's front sample buffer
    
    TRACES = np.load(TRACES_PATH, mmap_mode="r")
    TRACES_FILT = np.load(TRACES_FILT_PATH, mmap_mode="r")
    OUTPUTS = np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")
    
    # RT-Sort hyperparameters
    # (N_BEFORE and N_AFTER need to be set in .ipynb because they rely on SAMP_FREQ)
    N_BEFORE = N_AFTER = round(0.5 * SAMP_FREQ)  # Window for looking for electrode codetections
    
    MIN_ELECS_FOR_ARRAY_NOISE = round(0.5 * NUM_ELECS)
MIN_ACTIVITY_ROOT_COCS = 2
MIN_ACTIVITY = 30
SPLIT_ROOT_AMPS_AGAIN = True

STRINGENT_THRESH = 0.15
LOOSE_THRESH = 0.1  # Decimal form (not percent)
LOOSE_THRESH_LOGIT = sigmoid_inverse(LOOSE_THRESH)  # -2.1972245773362196  # For faster computation in spike splitting
INNER_RADIUS = 50
OUTER_RADIUS = 100

PRE_INTERELEC_ROOT_MAX_AMP_ONLY = True

MIN_AMP_DIST_P = 0.1  # I think 0.05 might work better

# #electrode codetections overlap for spike assignment and merging
MIN_LOOSE_DETECTIONS_N = 3 
MIN_LOOSE_DETECTIONS_R = 2/3
MIN_INNER_LOOSE_DETECTIONS = 2

MAX_LATENCY_DIFF_SPIKES = 2.51
MAX_AMP_MEDIAN_DIFF = 0.35

MAX_LATENCY_DIFF_SEQUENCES = 2.51
MAX_AMP_MEDIAN_DIFF_SEQUENCES = 0.35
    
MIN_ELEC_PROB = 0.03  # If an elec's mean prob is less than MIN_ELEC_PROB, it is set to 0

CLIP_LATENCY_DIFF = 5
CLIP_AMP_MEDIAN_DIFF = 1# 1000# 0.7

OVERLAP_TIME = 0.2  # For spike splitting and merging clusters with overlapping spikes

MAX_ROOT_AMP_MEDIAN_STD = 2
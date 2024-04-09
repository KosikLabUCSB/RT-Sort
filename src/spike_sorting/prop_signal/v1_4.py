"""
5/12/23 updating latency matrix
"""

import numpy as np
from pandas import DataFrame
from multiprocessing import Pool
from tqdm import tqdm
from scipy.cluster.hierarchy import fcluster, linkage


def merge_propagations(propagations: list, min_elec_overlap_p: float, min_seq_order_p: float,
                       return_clusters=False):
    """
    If the electrodes of two or more propagations overlap at least elec_overlap_p
    AND their sequential order overlaps ,
    then combine them into one propagation with multiple paths

    :param propagations:
        Output of get_propagation with P elements.
        Each element is a pandas.DataFrame of electrode cohorts for each propagation
        in a recording. Each DataFrame provides a list of candidate
        electrodes along with the latency between each electrode
        with the reference electrode, the number of co-occurrences,
        and the n1/n2 ratio.
    :param min_elec_overlap_p:
        Overlap amount between two propagations (A, B) is determined by
        (overlap(A, B) / (num(A) + num(B) - overlap(A,B)) where overlap(A, B) is the
        number of electrodes shared by both A and B, and num(X) is the number of electrodes in X.
        For two propagations to merge, their overlap score has to be at least elec_overlap_p / 100
    :param min_seq_order_p:
        Sequential order overlap between two propagations (A, B) is determined by
        creating a binary latency matrix. For two propagations to merge, the
         sequential order overlap has to be at least seq_order_p / 100.


    :return propagations_merged: list
        Each element is a list containing the merged propagations
        (len(list)==1 if propagation was not merged with another)
    """

    ind_a, ind_b = np.triu_indices(len(propagations), k=1)
    pdist = []
    for i_a, i_b in zip(ind_a, ind_b):
        prop_a = propagations[i_a]
        prop_b = propagations[i_b]
        overlap_score = get_elec_overlap_score(prop_a, prop_b)
        if overlap_score >= min_elec_overlap_p / 100:
            merge_score = get_seq_order_score(prop_a, prop_b)
        else:
            merge_score = 0
        pdist.append(1 - merge_score)
    pdist = np.asarray(pdist)
    linkage_m = linkage(pdist, method="average")
    clusters = fcluster(linkage_m, 1-min_seq_order_p, criterion="distance")
    propagations_merged = {}
    for i, c in enumerate(clusters):
        if c not in propagations_merged:
            propagations_merged[c] = [propagations[i]]
        else:
            propagations_merged[c].append(propagations[i])

    propagations_merged = list(propagations_merged.values())
    if return_clusters:
        return propagations_merged, clusters
    else:
        return propagations_merged


def get_elec_overlap(prop_a, prop_b):
    elec_a = prop_a.ID.values
    elec_b = prop_b.ID.values
    _, overlap_a, overlap_b = np.intersect1d(elec_a, elec_b, return_indices=True, assume_unique=True)
    return overlap_a, overlap_b


def get_elec_overlap_score(prop_a, prop_b):
    elec_a = prop_a.ID.values
    elec_b = prop_b.ID.values

    overlap_a, overlap_b = get_elec_overlap(prop_a, prop_b)
    num_overlap = len(overlap_a)
    overlap_score = num_overlap / (len(elec_a) + len(elec_b) - num_overlap)
    return overlap_score


def get_matrix_latency(prop, overlap_ind):
    latencies = prop.latency.values[overlap_ind]
    return latencies - latencies[:, None]
    # elecs = [prop.ID.values[e] for e in range(len(prop)) if e in overlap_ind]
    # elecs = np.argsort(elecs)
    # return np.clip(elecs - elecs[:, None], -1, 1)


def get_latency_pairs(latencies: np.ndarray):
    if latencies.size == 1:
        return latencies

    pairs = []
    for i in range(len(latencies)):
        for j in range(i+1, len(latencies)):
            pairs.append(latencies[i] - latencies[j])
    return np.asarray(pairs)


def get_seq_order_score(prop_a, prop_b):
    overlap_a, overlap_b = get_elec_overlap(prop_a, prop_b)
    if overlap_a.size == 0:
        return 10

    latencies_a = prop_a.latency.values[overlap_a]
    latencies_b = prop_b.latency.values[overlap_b]
    sums = np.sum(latencies_a) + np.sum(latencies_b)
    if sums < 0.00001:
        return 0

    pairs_a = get_latency_pairs(latencies_a)
    pairs_b = get_latency_pairs(latencies_b)
    seq_score = np.sum(np.abs(pairs_a - pairs_b))
    return seq_score / sums


# def get_seq_order_score(prop_a, prop_b):
#     # Construct sequential order matrix. Columns = reference. Rows = what reference is compared to. I.e. column rel. position - row rel. position
#     overlap_a, overlap_b = get_elec_overlap(prop_a, prop_b)
#     num_overlap = len(overlap_a)
#
#     matrix_a = get_latency_matrix(prop_a, overlap_a)
#     matrix_b = get_latency_matrix(prop_b, overlap_b)
#
#     # div = num_overlap * num_overlap - num_overlap if num_overlap > 1 else 1
#     # seq_order_score = np.sum(matrix_a * matrix_b) / div
#     seq_order_score = np.sum(matrix_a * matrix_b) / (np.sum(prop_a.latency) + np.sum(prop_b.latency))
#     return seq_order_score


def get_propagations(electrode_cohorts, min_duration):
    """
    This function generates a collection of cohort electrodes, each
    representing an eAP propagation in each recording

    Inputs:
        electrode_cohorts: list
            Output of rescan_candidate_cohorts.
            Each element is a np.array that contains the candidate electrodes
            along with the latency between each electrode with the reference electrode,
            the number of cooccurrences, and the n1/n2 ratio.
        min_duration: float
            Propagations must have a duration of at least min_duration

    Output:
        propagations: list
            Each element is a pandas.DataFrame of electrode cohorts for each propagation
            in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
    """

    propagations = []
    for cohort in electrode_cohorts:
        # Skip cohorts that are empty or only have 1 electrode
        if cohort.size <= 1 or cohort.shape[1] <= 1: continue
        # Skip cohorts that have duration less than min_duration
        if np.max(cohort[1, :]) < min_duration: continue

        # Sort propagation
        cohort_structured = [(cohort[1, j],  # latency
                             -cohort[2, j],  # small window cooccurrences, - to sort in descending order
                             -cohort[3, j])  # n1_n2_ratio, - to sort in descending order
                             for j in range(cohort.shape[1])]
        cs_dtype = [("latency", float), ("cooccs", float), ("ratio", float)]
        cohort_structured = np.array(cohort_structured, dtype=cs_dtype)
        order = np.argsort(cohort_structured, order=('latency', 'cooccs', 'ratio'))
        cohort = cohort[:, [0] + [j for j in order if j != 0]]  # Ensure first electrode comes first

        # Format into DataFrame
        table_data = {
            "ID": cohort[0, :],
            "latency": cohort[1, :],
            "small_window_cooccurrences": cohort[2, :],
            "n1_n2_ratio": cohort[3, :],
        }
        table = DataFrame(table_data).astype({"ID": int})
        propagations.append(table)

    return propagations


def rescan_candidate_cohorts(candidate_cohorts, max_latency, min_cocs_n, min_cocs_p, min_cocs_2_p):
    """
    This function rescans each set of candidate electrodes found for each
    reference electrode. First, find the electrode with the maximum number of
    co-occurrences with the reference electrode. Second, scan through all
    other electrodes in the set of candidate electrodes, to identify only
    the electrodes with more than p * the maximum number of co-occurrences
    and more than thres_cooccurrences and more than min_cocs_p*num_occurrences_on_first_electrode
    in the 0.5ms window with maximum number of co-occurrences in the CCG. The electrodes that satisfy
    this criterion are kept in electrode_cohorts

    Inputs:
        candidate_cohorts: list
            Output of scan_reference_electrode.py. Each element is a np.array
            containing a list of candidate constituent electrodes for each reference electrode.
            Each row of the np.arrays provides a list of candidate electrodes along with the
            latency between each electrode with the reference electrode, the number
            of co-occurrences, and the n1/n2 ratio.
        max_latency: float
            Maximum latency an electrode can have. Electrodes with greater than max_latency
            are removed from the cohort.
        min_cocs_n: float
            Lower bound of the number of short latency co-occurrences each
            electrode needs to have.
        min_cocs_p: float
            Each electrode needs to have min_cocs_p * num_occurrences
            to join a cohort where num_occurrences is the number of spikes
        min_cocs_2_p: int or float
            Percentage of the maximum number of co-occurrences required for
            all constituent electrodes. p should be between 0 and 100.

    Output:
        electrode_cohorts: list
            Each element is a np.array that contains the candidate electrodes
            along with the latency between each electrode with the reference electrode,
            the number of cooccurrences, and the n1/n2 ratio.

    """

    electrode_cohorts = [np.array([]) for _ in range(len(candidate_cohorts))]

    for i, cohort in enumerate(candidate_cohorts):
        if cohort.size <= 1: continue

        # Only include electrodes with 0 <= latency <= max_latency
        cohort = cohort[:, (0 <= cohort[1, :]) * (cohort[1, :] <= max_latency)]
        if cohort.shape[1] <= 1: continue

        # Remove electrodes with too few cooccurrences
        thresh_cocs = max(min_cocs_n, min_cocs_p / 100 * cohort[2, 0], min_cocs_2_p / 100 * np.max(cohort[2, 1:]))
        cohort = cohort[:, cohort[2, :] >= thresh_cocs]
        if cohort.shape[1] > 1: electrode_cohorts[i] = cohort

        # current_cohort = candidate_cohorts[i]
        # reference = np.flatnonzero(current_cohort[0, :] == i)
        # thresh_cooccurrences = max(min_cocs_n, min_cocs_p / 100 * current_cohort[2, reference])
        # current_cohort = current_cohort[:, np.flatnonzero(current_cohort[2, :] >= thresh_cooccurrences)]
        #
        # reference = np.flatnonzero(current_cohort[0, :] == i)  # Need to refind index of reference
        #
        # non_zero_electrodes = np.flatnonzero(current_cohort[1, :] != 0)
        # target_electrodes = np.setdiff1d(non_zero_electrodes, reference)
        # if target_electrodes.size > 0:
        #     cooccurrences = min_cocs_2_p / 100 * max(current_cohort[2, target_electrodes])
        #     index = np.flatnonzero(current_cohort[2, :] >= cooccurrences)
        #     index = np.union1d(index, reference)
        #     current_cohort_new = current_cohort[:, index]
        #     electrode_cohorts[i] = current_cohort_new

    return electrode_cohorts


def scan_reference_electrode(spike_times, sampling_freq, min_prop_spikes, min_ccg_ratio,
                             ccg_before=1.5, ccg_after=1.5, ccg_small_window=0.5, ccg_big_window=2):
    """
    This function generates a cell array containing candidate electrode cohorts
    for each electrode. Each cell corresponds to an electrode with the same
    order of your input. If a cell is empty, there's no electrode cohorts
    associated with this electrode. Use rescan_each_reference to find
    constituent electrodes from candidate electrode cohorts.

    Inputs:
        spike_times: list
            Contains N elements, each representing 1 electrode.
            Each element contains a np.array with shape (m,) representing
            the spike times for each electrode.

        sampling_freq: int
            Sampling frequency of recording that gave spike times (in kHz)
        min_prop_spikes: int
            Min number of spikes for an electrode to be considered to have neuronal activity
        min_ccg_ratio: float
            Let n1 denote the largest sum of counts in any 0.5 ms moving
            window in the cross-correlogram (CCG) and n2 denote the sum
            of counts of the 2 ms window with the location of the largest
            sum in the center. If the largest sum is found in the first
            1 ms or the last 1 ms of the CCG, take the sum of the counts
            of the first 2 ms window or the counts of the last 2 ms window
            as n2. This ratio is the lower bound threshold for n1/n2

        ccg_before: int or float
            The time (in ms) before each reference spike to use when creating the CCG
        ccg_after: int or float
            The time (in ms) after each reference spike to use when creating the CCG
        ccg_small_window: int or float
            The size (in ms) of the window of n1 described in min_ccg_ratio
        ccg_big_window: int or float
            The size (in ms) of the window of n2 described in min_ccg_ratio

    Output:
        candidate_cohorts: list
            Each element is a np.array containing a list of candidate constituent
            electrodes for each reference electrode. Each row of the np.arrays
            provides a list of candidate electrodes along with the latency between
            each electrode with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
    """

    init_dict = {
        "spike_times": spike_times,

        "sampling_freq": sampling_freq,
        "min_prop_spikes": min_prop_spikes,
        "min_ccg_ratio": min_ccg_ratio,

        "ccg_before": ccg_before,
        "ccg_after": ccg_after,
        "ccg_small_window": ccg_small_window,
        "ccg_big_window": ccg_big_window,
    }
    num_elecs = len(spike_times)
    candidate_cohorts = [np.array([]) for _ in range(num_elecs)]

    # Used to replicate MATLAB's multiprocessing with parfor
    print(f"Scanning reference electrodes ...")
    with Pool(initializer=_scan_reference_electrode_worker_init, initargs=(init_dict,), processes=20) as pool:
        for i, cohort_data in tqdm(enumerate(pool.imap(_scan_reference_electrode_func, range(num_elecs))), total=num_elecs):
            if cohort_data is not None:
                candidate_cohorts[i] = cohort_data

    return candidate_cohorts


def _scan_reference_electrode_worker_init(init_dict):
    # Initialize variables for parallel processing worker
    # TODO: COULD BE CONSUMING TOO MUCH RAM SINCE "spike_times" IS VERY LARGE AND COPIED FOR EACH WORKER. INVESTIGATE FURTHER. cache in memory?
    global _scan_reference_electrode_worker_dict
    _scan_reference_electrode_worker_dict = init_dict


def _scan_reference_electrode_func(electrode):
    # Function that each parallel processing worker will execute

    spike_times = _scan_reference_electrode_worker_dict["spike_times"]
    sf = _scan_reference_electrode_worker_dict["sampling_freq"]
    min_prop_spikes = _scan_reference_electrode_worker_dict["min_prop_spikes"]
    min_ccg_ratio = _scan_reference_electrode_worker_dict["min_ccg_ratio"]
    ccg_before = _scan_reference_electrode_worker_dict["ccg_before"]
    ccg_after = _scan_reference_electrode_worker_dict["ccg_after"]
    ccg_small_window = _scan_reference_electrode_worker_dict["ccg_small_window"]
    ccg_big_window = _scan_reference_electrode_worker_dict["ccg_big_window"]

    ref_spike_times = spike_times[electrode]
    if len(ref_spike_times) == 0 or len(ref_spike_times) < min_prop_spikes: return None

    elec_cohort = [(electrode, 0, ref_spike_times.size, 1)]

    for electrode2 in range(len(spike_times)):
        if electrode2 == electrode: continue
        tar_spike_times = spike_times[electrode2]
        n_tar = tar_spike_times.size
        if n_tar == 0: continue

        ccg = np.zeros(int((ccg_before + ccg_after) * sf) + 1)
        for ref in ref_spike_times:
            i_tar = np.searchsorted(tar_spike_times, ref - ccg_before)

            # Add tar spikes on left and right of index to CCG
            while 0 <= i_tar < n_tar:
                tar = tar_spike_times[i_tar]
                if tar <= ref + ccg_after:
                    bin = (tar - ref + ccg_before) * sf
                    ccg[round(bin)] += 1  # round because of floating point rounding error
                else:
                    break
                i_tar += 1

        small_ind = round(ccg_small_window * sf) + 1
        loc = 0
        spikes_small_window = np.sum(ccg[:small_ind])
        sum_small_window = spikes_small_window
        for i in range(small_ind, len(ccg)):
            sum_small_window += -ccg[i-small_ind] + ccg[i]
            if sum_small_window > spikes_small_window:
                spikes_small_window = sum_small_window
                loc = i-small_ind+1

        delay = loc + np.argmax(ccg[loc:loc+small_ind])
        big_w = round(ccg_big_window * sf)
        min_ind = delay - big_w//2
        max_ind = delay + big_w//2 + 1

        if min_ind < 0:
            spikes_big_window = ccg[:big_w + 1].sum()
        elif max_ind > ccg.size:
            spikes_big_window = ccg[-big_w-1:].sum()
        else:
            spikes_big_window = ccg[min_ind:max_ind].sum()

        if spikes_small_window >= min_ccg_ratio * spikes_big_window and spikes_big_window >= 1:
            time_delay = delay / sf - ccg_before
            if time_delay >= -ccg_before:
                elec_cohort.append((electrode2, time_delay, spikes_small_window, spikes_small_window / spikes_big_window))
    return np.vstack(elec_cohort).T


# def prop_signal(spike_times,
#                 sampling_freq, min_prop_spikes, min_ccg_ratio,
#                 min_cocs_n, min_cocs_p, min_cocs_2_p):
#     """
#     This function detects eAP propagation in a recording and generates a
#     collection of cohort electrodes, each representing an eAP propagation in
#     each recording. Along with the cohort electrodes, this function also
#     outputs the propagating spike times for each eAP propagation with
#     different number of anchor points.
#
#     Inputs:
#         spike_times: list
#             Contains N elements, each representing 1 electrode.
#             Each element contains a np.array with shape (m,) representing
#             the spike times for each electrode.
#
#         sampling_freq: int
#             Sampling frequency of recording that gave spike times (in kHz)
#         min_prop_spikes: int
#             Min number of spikes for an electrode to be the first electrode in a prop sequence
#         min_ccg_ratio: float
#             Let n1 denote the largest sum of counts in any 0.5 ms moving
#             window in the cross-correlogram (CCG) and n2 denote the sum
#             of counts of the 2 ms window with the location of the largest
#             sum in the center. If the largest sum is found in the first
#             1 ms or the last 1 ms of the CCG, take the sum of the counts
#             of the first 2 ms window or the counts of the last 2 ms window
#             as n2. This ratio is the lower bound threshold for n1/n2
#
#         min_cocs_n: int or float
#             Lower bound of the number of short latency co-occurrences each
#             electrode needs to have.
#         min_cocs_p: float
#             Each electrode needs to have min_cocs_p * num_occurrences
#             to join a cohort where num_occurrences is the number of spikes
#             detected on the first electrode in the cohort
#         min_cocs_2_p: int or float
#             Percentage of (the maximum number of co-occurrences of any constituent electrode)
#             required for all constituent electrodes. p should be between 0 and 100.
#     Outputs:
#         propagations: list
#             Contains P elements, each a pandas.DataFrames (each with 4 columns) of electrode cohorts for
#             each propagation (p) in a recording. Each DataFrame provides a list of candidate
#             electrodes along with the latency between each electrode
#             with the reference electrode, the number of co-occurrences,
#             and the n1/n2 ratio.
#         propagating_times: list
#             Contains P elements, each a 1d np.array with shape (Q,)
#             of spike times in the propagation with different number of anchor points chosen
#             for each propagation in propagations. The pth element in propagating_times
#             contains the spike times for the pth element in propagations.
#
#             The qth element in the inner array with shape (Q,) is an array containing the
#             propagating spike times isolated with q+1 anchor points. I.e. the 0th element
#             contains the propagating spike times isolated with 1 anchor point (an empty array),
#             the 1st element contains propagating spike times isolated with 2 anchor points,
#             the 2nd element contains propagating spike times isolated with 3 anchor points,
#             etc., until all constituent electrodes are used as anchor points.
#
#     Possible optimizations:
#         Optimize parallel processing. See TODO
#         numpy operations can speed up calculations
#         Combine scan_reference_electrode and rescan_candidate_cohorts? (1 loop instead of 2)
#     """
#     candidate_cohorts = scan_reference_electrode(spike_times, thresh_freq, seconds_recording, thresh_number_spikes, ratio,
#                                                  small_window=small_window, big_window=big_window,
#                                                  ccg_before=ccg_before, ccg_after=ccg_after, ccg_n_bins=int_round(ccg_n_bins))
#     electrode_cohorts = rescan_candidate_cohorts(candidate_cohorts, min_cocs_n, min_cocs_p, p)
#     propagations = get_propagation(electrode_cohorts)
#     propagating_times = get_propagation_time(propagations, spike_times, prop_after=ccg_after)
#
#     return propagations, propagating_times
#
#
# if __name__ == "__main__":
#     spike_times = np.load("/data/MEAprojects/PropSignal/data/200123_2950/thresh_crossings_5.npy", allow_pickle=True)
#     spike_times = [np.asarray(st) for st in spike_times]
#     scan_reference_electrode(spike_times, 20, 180, 0.5)

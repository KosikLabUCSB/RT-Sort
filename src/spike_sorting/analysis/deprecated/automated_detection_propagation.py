import numpy as np
from pandas import DataFrame
from multiprocessing import Pool


def int_round(n):
    """
    Rounds a float (n) to the nearest integer
    Uses standard math rounding, i.e. 0.5 rounds up to 1
    """

    n_int = int(n)
    if n - n_int >= 0.5:
        n_int += 1
    return n_int


def get_propagation_time(list_of_propagation, spike_times, prop_after):
    """
    This function generates a sequence of eAPs for each
    propagation using different number of anchor points

    Inputs:
        list_of_propagation: list
            Output of get_propagation with P elements.
            Each element is a pandas.DataFrame of electrode cohorts for each propagation
            in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
        spike_times: list
            Contains N elements, each representing 1 electrode.
            Each element contains a np.array with shape (m,) representing
            the spike times for each electrode.
        ccg_after: int or float
            Maximum time after reference spike to classify the target spike as
            a propagation of the reference spike.
            For example, if reference spike is at time 0 and prop_after=1.5,
            any target spike within the interval (0, 1.5) will be classified as a
            propagation of reference spike

    Output:
        propagating_times: list
            Contains P elements where each element contains a 1d np.array with shape (Q,)
            of spike times in the propagation with different number of anchor points chosen
            for each propagation in list_of_propagation. The pth element in propagating_times
            contains the spike times for the pth element in list_of_propagation.

            The qth element in the inner array with shape (Q,) is an array containing the
            propagating spike times isolated with q+1 anchor points. I.e. the 0th element
            contains the propagating spike times isolated with 1 anchor point (an empty array),
            the 1st element contains propagating spike times isolated with 2 anchor points,
            the 2nd element contains propagating spike times isolated with 3 anchor points,
            etc., until all constituent electrodes are used as anchor points.
    """

    n = len(list_of_propagation)
    propagating_times = []
    for i in range(n):
        time_signal = [np.array([])]
        time = []
        current_signal = list_of_propagation[i]

        ind0 = 0
        index = np.flatnonzero(current_signal.loc[:, "latency"] > 0)  # 0.001
        index = np.concatenate(([ind0], index))
        current_signal = current_signal.loc[index, :].copy()
        size_of_signal, _ = current_signal.shape

        current_signal.loc[ind0, "small_window_cooccurrences"] = -888
        ID0 = current_signal.loc[ind0, "ID"]
        reference_spike_times = spike_times[ID0]

        for n_ele in range(1, size_of_signal):  # ele = electrode
            ind1 = current_signal.loc[:, "small_window_cooccurrences"].idxmax()
            current_signal.loc[ind1, "small_window_cooccurrences"] = -888
            ID1 = current_signal.loc[ind1, "ID"]
            target_spike_times = spike_times[ID1]

            ind_all = []
            for j in range(reference_spike_times.size):
                ref = reference_spike_times[j]
                index = np.flatnonzero((target_spike_times > ref) & (target_spike_times < ref+prop_after))
                if index.size > 0:
                    ind_all.append(j)

            time.extend(reference_spike_times[ind_all])
            time_signal.append(np.sort(np.unique(time)))
        propagating_times.append(np.asarray(time_signal, dtype=object))

    return propagating_times


def get_propagation(electrode_cohorts):
    """
    This function generates a collection of cohort electrodes, each
    representing an eAP propagation in each recording

    Inputs:
        electrode_cohorts: list
            Output of rescan_candidate_cohorts.
            Each element is a np.array that contains the candidate electrodes
            along with the latency between each electrode with the reference electrode,
            the number of cooccurrences, and the n1/n2 ratio.

    Output:
        list_of_propagation: list
            Each element is a pandas.DataFrame of electrode cohorts for each propagation
            in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
    """

    list_of_propagation = []
    for i in range(len(electrode_cohorts)):
        if electrode_cohorts[i].size > 0:
            temps = electrode_cohorts[i]
            m1, n1 = temps.shape
            if temps.size > 0 and not np.any(temps[1, :] < 0) and n1 > 1:
                index = np.flatnonzero(temps[0, :] == i)
                if index.size > 0:
                    order = np.argsort(temps[1, :])
                    order = np.delete(order, np.flatnonzero(order == index))
                    order = np.concatenate((index, order))
                    temps = temps[:, order]

                    table_data = {
                        "ID": temps[0, :],
                        "latency": temps[1, :],
                        "small_window_cooccurrences": temps[2, :],
                        "n1_n2_ratio": temps[3, :],
                    }
                    table = DataFrame(table_data).astype({"ID": int})
                    if table.size > 0:
                        list_of_propagation.append(table)

    return list_of_propagation


def rescan_candidate_cohorts(candidate_cohorts, thres_cooccurrences, p):
    """
    This function rescans each set of candidate electrodes found for each
    reference electrode. First, find the electrode with the maximum number of
    co-occurrences with the reference electrode. Second, scan through all
    other electrodes in the set of candidate electrodes, to identify only
    the electrodes with more than p * the maximum number of co-occurrences
    and more than thres_cooccurrences in the 0.5ms window with
    maximum number of co-occurrences in the CCG. The electrodes that satisfy
    this criterion are kept in electrode_cohorts

    Inputs:
        candidate_cohorts: list
            Output of scan_reference_electrode.py. Each element is a np.array
            containing a list of candidate constituent electrodes for each reference electrode.
            Each row of the np.arrays provides a list of candidate electrodes along with the
            latency between each electrode with the reference electrode, the number
            of co-occurrences, and the n1/n2 ratio.
        thres_cooccurrences: int or float
            Lower bound of the number of short latency co-occurrences each
            electrode needs to have.
        p: int or float
            Percentage of the maximum number of co-occurrences required for
            all constituent electrodes. p should be between 0 and 100.

    Output:
        electrode_cohorts: list
            Each element is a np.array that contains the candidate electrodes
            along with the latency between each electrode with the reference electrode,
            the number of cooccurrences, and the n1/n2 ratio.

    """

    numbers = len(candidate_cohorts)
    electrode_cohorts = [np.array([])] * numbers

    for i in range(numbers):
        if candidate_cohorts[i].size > 0:
            current_cohort = candidate_cohorts[i]
            current_cohort = current_cohort[:, np.flatnonzero(current_cohort[2, :] >= thres_cooccurrences)]
            reference = np.flatnonzero(current_cohort[0, :] == i)
            non_zero_electrodes = np.flatnonzero(current_cohort[1, :] != 0)
            target_electrodes = np.setdiff1d(non_zero_electrodes, reference)

            if target_electrodes.size > 0:
                cooccurrences = p/100 * max(current_cohort[2, target_electrodes])
                index = np.flatnonzero(current_cohort[2, :] >= cooccurrences)
                index = np.union1d(index, reference)
                current_cohort_new = current_cohort[:, index]
                electrode_cohorts[i] = current_cohort_new

    return electrode_cohorts


def scan_reference_electrode(spike_times, thres_freq, seconds_recording, thres_number_spikes, ratio,
                             small_window=0.5, big_window=2, ccg_before=1.5, ccg_after=1.5, ccg_n_bins=61):
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
        thres_freq: int or float
            A value representing the frequency lower bound of the spiking
            frequency for all electrodes. Only electrodes that's above the
            threshold will be considered as a reference electrode. For
            example, enter 1 for 1Hz.
        seconds_recording: int or float
            The length of recording in seconds. For example, 120 for
            2 minutes recording.
        thres_number_spikes: int or None
            The threshold for total number of spikes on each reference
            electrode. If is None, use thres_freq for threshold instead.
        ratio: float
            Let n1 denote the largest sum of counts in any 0.5 ms moving
            window in the cross-correlogram (CCG) and n2 denote the sum
            of counts of the 2 ms window with the location of the largest
            sum in the center. If the largest sum is found in the first
            1 ms or the last 1 ms of the CCG, take the sum of the counts
            of the first 2 ms window or the counts of the last 2 ms window
            as n2. This ratio is the lower bound threshold for n1/n2
        small_window: int or float
            The size of the window of n1 described in ratio i
        big_window: int or float
            The size of the window of n2 described in ratio
        ccg_before: int or float
            The time before each reference spike to use when creating the CCG
        ccg_after: int or float
            The time after each reference spike to use when creating the CCG
        ccg_n_bins: int
            The number of bins to use when creating the CCG

    Output:
        candidate_cohorts: list
            Each element is a np.array containing a list of candidate constituent
            electrodes for each reference electrode. Each row of the np.arrays
            provides a list of candidate electrodes along with the latency between
            each electrode with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
    """

    if thres_number_spikes is not None:
        thres = thres_number_spikes
    else:
        thres = thres_freq * seconds_recording

    init_dict = {
        "spike_times": spike_times,
        "small_window": small_window,
        "big_window": big_window,
        "ccg_before": ccg_before,
        "ccg_after": ccg_after,
    }

    ccg_mid_ind = int(ccg_n_bins // 2)
    ccg_ind_factor = (ccg_n_bins - 1 - ccg_mid_ind) / init_dict["ccg_after"]
    init_dict["ccg_mid_ind"] = ccg_mid_ind
    init_dict["ccg_ind_factor"] = ccg_ind_factor
    init_dict["ccg_n_bins"] = ccg_n_bins

    init_dict["t"] = int_round(init_dict["small_window"] * ccg_ind_factor)

    n_e = len(spike_times)
    init_dict["n_e"] = n_e
    candidate_cohorts = [np.array([])] * n_e

    init_dict["thres"] = thres
    init_dict["ratio"] = ratio

    # Used to replicate MATLAB's multiprocessing with parfor
    with Pool(initializer=_scan_reference_electrode_worker_init, initargs=(init_dict,)) as pool:
        for i, cohort_data in enumerate(pool.map(_scan_reference_electrode_func, range(n_e))):
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
    thres = _scan_reference_electrode_worker_dict["thres"]
    ratio = _scan_reference_electrode_worker_dict["ratio"]
    n_e = _scan_reference_electrode_worker_dict["n_e"]
    t = _scan_reference_electrode_worker_dict["t"]

    small_window = _scan_reference_electrode_worker_dict["small_window"]
    big_window = _scan_reference_electrode_worker_dict["big_window"]
    ccg_before = _scan_reference_electrode_worker_dict["ccg_before"]
    ccg_after = _scan_reference_electrode_worker_dict["ccg_after"]
    ccg_mid_ind = _scan_reference_electrode_worker_dict["ccg_mid_ind"]
    ccg_ind_factor = _scan_reference_electrode_worker_dict["ccg_ind_factor"]
    ccg_n_bins = _scan_reference_electrode_worker_dict["ccg_n_bins"]

    ref = spike_times[electrode]
    n = ref.size  # Number of spikes detected by electrode

    if n >= 0 and n >= thres:
        time_delay = np.full(n_e, -999.0)
        small_window_cooccurrences = np.zeros(n_e)
        spikes_ratio = np.full(n_e, -999.0)
        for electrode2 in range(n_e):
            if electrode2 == electrode:
                continue

            ccg = np.zeros(ccg_n_bins)  # ccg = cross correlogram
            tar = spike_times[electrode2]  # tar = target(s)
            for k in range(n):
                ref_value = ref[k]
                index = np.flatnonzero((tar >= ref_value-ccg_before) * (tar <= ref_value + ccg_after))
                if index.size > 0:
                    for i in index:
                        bin_value = tar[i] - ref_value
                        ccg_ind = int_round(bin_value * ccg_ind_factor + ccg_mid_ind)
                        ccg_ind = min(ccg_ind, ccg_n_bins-1)
                        ccg_ind = max(ccg_ind, 0)
                        ccg[ccg_ind] += 1
            spikes_small_window = ccg[:t+1].sum()
            location = 0
            for i in range(ccg_n_bins-t):
                sum_small_window = ccg[i:i+t+1].sum()
                if sum_small_window > spikes_small_window:
                    spikes_small_window = sum_small_window
                    location = i

            delay = location + np.argmax(ccg[location:location+t+1])
            big_window_converted = int_round(big_window * ccg_ind_factor)
            min_ind = int(delay - big_window_converted/2)
            max_ind = int_round(delay + big_window_converted/2) + 1

            if min_ind < 0:
                spikes_big_window = ccg[:big_window_converted + 1].sum()
            elif max_ind > ccg_n_bins:
                spikes_big_window = ccg[int_round(big_window_converted/2):].sum()
            else:
                spikes_big_window = ccg[min_ind:max_ind].sum()

            if spikes_small_window >= ratio * spikes_big_window and spikes_big_window >= 1:
                time_delay[electrode2] = (delay - ccg_mid_ind) / ccg_ind_factor
                small_window_cooccurrences[electrode2] = spikes_small_window
                spikes_ratio[electrode2] = spikes_small_window / spikes_big_window

        # Set values for own electrode
        time_delay[electrode] = 0
        small_window_cooccurrences[electrode] = n
        spikes_ratio[electrode] = 1

        ind = np.flatnonzero(time_delay >= -ccg_before)   # ind = indices which are the electrode ids that are a part of the cohort
        return np.vstack((ind, time_delay[ind], small_window_cooccurrences[ind], spikes_ratio[ind]))


def automated_detection_propagation(spike_times, thres_freq, seconds_recording, thres_number_spikes, ratio, thres_cooccurrences, p,
                                    small_window=0.5, big_window=2, ccg_before=1.5, ccg_after=1.5, ccg_n_bins=61):
    """
    This function detects eAP propagation in a recording and generates a
    collection of cohort electrodes, each representing an eAP propagation in
    each recording. Along with the cohort electrodes, this function also
    outputs the propagating spike times for each eAP propagation with
    different number of anchor points.

    Inputs:
        spike_times: list
            Contains N elements, each representing 1 electrode.
            Each element contains a np.array with shape (m,) representing
            the spike times for each electrode.
        thres_freq: int, float, or None
            A value representing the frequency lower bound of the spiking
            frequency for all electrodes. Only electrodes that's above the
            threshold will be considered as a reference electrode. For
            example, enter 1 for 1Hz.
        seconds_recording: int, float, or None
            The length of recording in seconds. For example, 120 for
            2 minutes recording.
        thres_number_spikes: int, float, or None
            The threshold for total number of spikes on each reference
            electrode. If is None, use thres_freq for threshold instead.
        ratio: float
            Let n1 denote the largest sum of counts in any 0.5 ms moving
            window in the cross-correlogram (CCG) and n2 denote the sum
            of counts of the 2 ms window with the location of the largest
            sum in the center. If the largest sum is found in the first
            1 ms or the last 1 ms of the CCG, take the sum of the counts
            of the first 2 ms window or the counts of the last 2 ms window
            as n2. This ratio is the lower bound threshold for n1/n2
        thres_cooccurrences: int or float
            Lower bound of the number of short latency co-occurrences each
            electrode needs to have.
        p: int or float
            Percentage of the maximum number of co-occurrences required for
            all constituent electrodes. p should be between 0 and 100.
                small_window: int or float
            The size of the window of n1 described in ratio
        small_window: int or float
            The size of the window of n1 described in ratio
        big_window: int or float
            The size of the window of n2 described in ratio
        ccg_before: int or float
            The time before each reference spike to use when creating the CCG
        ccg_after: int or float
            The time after each reference spike to use when creating the CCG
        ccg_n_bins:
            The number of bins to use when creating the CCG
    Outputs:
        list_of_propagation: list
            Contains P elements, each a pandas.DataFrames (each with 4 columns) of electrode cohorts for
            each propagation (p) in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
        propagating_times: list
            Contains P elements, each a 1d np.array with shape (Q,)
            of spike times in the propagation with different number of anchor points chosen
            for each propagation in list_of_propagation. The pth element in propagating_times
            contains the spike times for the pth element in list_of_propagation.

            The qth element in the inner array with shape (Q,) is an array containing the
            propagating spike times isolated with q+1 anchor points. I.e. the 0th element
            contains the propagating spike times isolated with 1 anchor point (an empty array),
            the 1st element contains propagating spike times isolated with 2 anchor points,
            the 2nd element contains propagating spike times isolated with 3 anchor points,
            etc., until all constituent electrodes are used as anchor points.

    Possible optimizations:
        Optimize parallel processing. See TODO
        numpy operations can speed up calculations
        Combine scan_reference_electrode and rescan_candidate_cohorts? (1 loop instead of 2)
    """
    candidate_cohorts = scan_reference_electrode(spike_times, thres_freq, seconds_recording, thres_number_spikes, ratio,
                                                 small_window=small_window, big_window=big_window,
                                                 ccg_before=ccg_before, ccg_after=ccg_after, ccg_n_bins=int_round(ccg_n_bins))
    electrode_cohorts = rescan_candidate_cohorts(candidate_cohorts, thres_cooccurrences, p)
    list_of_propagation = get_propagation(electrode_cohorts)
    propagating_times = get_propagation_time(list_of_propagation, spike_times, prop_after=ccg_after)

    return list_of_propagation, propagating_times

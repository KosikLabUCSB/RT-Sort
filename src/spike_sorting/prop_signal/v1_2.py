"""
Changes since last version
    :func get_propagations: (previously :method get_propagation:)
        Previously get_propagation
        Now orders electrode cohorts by latency then num_cooccurrences, then by ratio

    :func merge_propagations:
        New function to merge propagations

    :func get_propagating_times:
        Gets propagations times with new merged propagations
"""

import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from src.prop_signal import v1_1


def get_propagations(electrode_cohorts):
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
        propagations: list
            Each element is a pandas.DataFrame of electrode cohorts for each propagation
            in a recording. Each DataFrame provides a list of candidate
            electrodes along with the latency between each electrode
            with the reference electrode, the number of co-occurrences,
            and the n1/n2 ratio.
    """

    propagations = []
    for i in range(len(electrode_cohorts)):
        if electrode_cohorts[i].size > 0:
            temps = electrode_cohorts[i]
            m1, n1 = temps.shape
            if temps.size > 0 and not np.any(temps[1, :] < 0) and n1 > 1:
                index_first = np.flatnonzero(temps[0, :] == i)[0]

                temps_structured = [(temps[1, j],  # latency
                                     -temps[2, j],  # small window cooccurrences, - to sort in descending order
                                     -temps[3, j])  # n1_n2_ratio, - to sort in descending order
                                    for j in range(n1)]
                ts_dtype = [("latency", float), ("cooccs", float), ("ratio", float)]
                temps_structured = np.array(temps_structured, dtype=ts_dtype)
                order = np.argsort(temps_structured, order=('latency', 'cooccs', 'ratio'))
                temps = temps[:, [index_first] + [j for j in order if j != index_first]]

                table_data = {
                    "ID": temps[0, :],
                    "latency": temps[1, :],
                    "small_window_cooccurrences": temps[2, :],
                    "n1_n2_ratio": temps[3, :],
                }
                table = DataFrame(table_data).astype({"ID": int})
                propagations.append(table)

    return propagations


def merge_propagations(propagations: list, elec_overlap_p: float, seq_order_p: float):
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
    :param elec_overlap_p:
        Overlap amount between two propagations (A, B) is determined by
        (overlap(A, B) / (num(A) + num(B) - overlap(A,B)) where overlap(A, B) is the
        number of electrodes shared by both A and B, and num(X) is the number of electrodes in X.
        For two propagations to merge, their overlap score has to be at least elec_overlap_p / 100
    :param seq_order_p:
        Sequential order overlap between two propagations (A, B) is determined by
        creating a binary latency matrix. For two propagations to merge, the
         sequential order overlap has to be at least seq_order_p / 100.


    :return propagations_merged: list
        Each element is a list containing the merged propagations
        (len(list)==1 if propagation was not merged with another)
    """
    # Need to create matrix to handle edge case of:
    # Prop A matches with prop B and C, but prop B does not match with prop C.
    # Each individual prop can only be a part of one merged prop.
    # (If one prop joins multiple merged prop, then when that prop detects
    # a spike, multiple merged props will detect a spike because only one
    # prop in a merged prop needs to detect a spike for the merge to detect a spike)
    # Prop A should match with whichever prop (B or C) has higher seq order score
    merge_matrix = np.zeros((len(propagations), len(propagations)))
    for i_a in range(len(propagations)):
        prop_a = propagations[i_a]
        for i_b in range(len(propagations)):
            if i_a == i_b: continue
            prop_b = propagations[i_b]

            # Merge based on electrode overlap
            overlap_score = get_elec_overlap_score(prop_a, prop_b)
            if overlap_score < elec_overlap_p / 100: continue

            # Merge based on sequential order
            seq_order_score = get_seq_order_score(prop_a, prop_b)
            if seq_order_score < seq_order_p / 100: continue

            merge_matrix[i_a, i_b] = seq_order_score

    # Get scores of unique pairs of propagations (i.e. only count (A, B), not (B, A) as well)
    # i.e. everything above (not including) top left to bottom right diagonal
    unique_scores_ind = np.triu_indices_from(merge_matrix, k=1)
    unique_scores = merge_matrix[unique_scores_ind]
    ind_sorted = np.argsort(-unique_scores)

    # Merge propagations starting with pairs with highest score
    propagations_merged = {i: [prop] for i, prop in enumerate(propagations)}
    for score, i_a, i_b in zip(unique_scores[ind_sorted], unique_scores_ind[0][ind_sorted], unique_scores_ind[1][ind_sorted]):
        if score < seq_order_p / 100:
            break

        if i_a not in propagations_merged or i_b not in propagations_merged: continue

        propagations_merged[i_a].extend(propagations_merged[i_b])
        del propagations_merged[i_b]

    # Remove propagations which have been merged
    return list(propagations_merged.values())


def get_elec_overlap(prop_a, prop_b):
    elec_a = prop_a.ID.values
    elec_b = prop_b.ID.values
    _, overlap_a, overlap_b = np.intersect1d(elec_a, elec_b, return_indices=True)
    return overlap_a, overlap_b


def get_elec_overlap_score(prop_a, prop_b):
    elec_a = prop_a.ID.values
    elec_b = prop_b.ID.values

    overlap_a, overlap_b = get_elec_overlap(prop_a, prop_b)
    num_overlap = len(overlap_a)
    overlap_score = num_overlap / (len(elec_a) + len(elec_b) - num_overlap)
    return overlap_score


def get_seq_order_score(prop_a, prop_b):
    # Construct sequential order matrix. Columns = reference. Rows = what reference is compared to. I.e. column rel. position - row rel. position
    elec_a = prop_a.ID.values
    elec_b = prop_b.ID.values

    overlap_a, overlap_b = get_elec_overlap(prop_a, prop_b)
    num_overlap = len(overlap_a)

    matrices = []
    for elec, overlap_ind in [(elec_a, overlap_a), (elec_b, overlap_b)]:
        overlap_ind = set(overlap_ind)
        elec = [elec[e] for e in range(len(elec)) if e in overlap_ind]

        elec_order = np.argsort(elec)
        rel_pos = elec_order - elec_order[:, None]
        matrix = np.clip(rel_pos, -1, 1)
        matrices.append(matrix)

        # Sanity check that calculating matrix works
        # test = np.zeros_like(matrix)
        # for i, elec_a in enumerate(np.sort(elec)):
        #     for j, elec_b in enumerate(np.sort(elec)):
        #         rel_pos = np.flatnonzero(elec == elec_b) - np.flatnonzero(elec == elec_a)
        #         rel_pos = np.clip(rel_pos, -1, 1)
        #         test[i, j] = rel_pos
        # if not np.all(matrix == test):
        #     print(elec)
        #     display(matrix)
        #     display(test)
        #     print()

    seq_order_score = np.sum(matrices[0] * matrices[1]) / (num_overlap * num_overlap - num_overlap)
    return seq_order_score


def get_propagating_times(merged_propagations, spike_times, prop_after, thresh_coactivations):
    """
    This function generates a sequence of eAPs for each
    propagation using different number of anchor points

    Inputs:
        merged_propagations: list
            Output of merge_propagations
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
        thresh_coactivations: int
            Number of electrodes that need to be activated in a sequence within prop_after of
            the first electrode being activated to detect a spike

    Output:
        propagating_times: list
            N elements where N is the number of merged propagations in merged_propagations.
            Each element is a list containing the spike times detected by the propagation
            (at the time it was first detected by one of its electrodes).
    """

    def reset_prop(idx):
        initial_sts[idx] = None
        activated_elecs[idx] = set()

    # Sort spike times
    est_dtype = [('elec', int), ('st', float)]
    elec_spike_times = []
    for i, elec in enumerate(spike_times):
        elec_spike_times.extend((i, st) for st in elec)
    elec_spike_times = np.sort(np.array(elec_spike_times, dtype=est_dtype), order="st")

    propagating_times = [[] for _ in range(len(merged_propagations))]
    # Iterate through every merged propagation
    for prop_m in tqdm(merged_propagations):
        elec_order = [{elec: i for i, elec in enumerate(prop)} for prop in prop_m]  # {electrode : index}
        initial_sts = [None] * len(prop_m)  # Spike time of first activated electrode
        activated_elecs = [set() for _ in range(len(prop_m))]  # Electrode that have already been activated

        # Iterate through every spike time
        for elec, st in elec_spike_times:
            # Iterate through every signal in a merged propagation signal
            for i in range(len(prop_m)):
                # Check if spike's electrode is in signal
                if elec not in prop_m[i]:
                    continue

                # Check if an electrode in signal has been activated yet
                if initial_sts[i] is None:
                    initial_sts[i] = st
                    activated_elecs[i].add(elec)
                    continue

                # Check if time for signal has passed
                "TODO: Compare >= vs >"
                if st - initial_sts[i] >= prop_after:
                    # Reset only 1 prop of merged prop
                    reset_prop(i)
                    continue

                "TODO: Reset instead?"
                # Check if spike's electrode has already been activated
                if elec in activated_elecs[i]:
                    continue

                "TODO: Analysis on effect of forcing first electrode in prop need to be activated first"

                "TODO: Analysis on effect of forcing to follow sequence order. What about 0-latency or elecs with same latency?"
                # Check if spike's electrode follows signal's sequence order
                elec_pos = elec_order[i][elec]
                for e in activated_elecs[i]:
                    if elec_order[i][e] < elec_pos:
                        break
                else:
                    # Check if enough electrodes have been activated for prop to detect a spike
                    if len(activated_elecs) == thresh_coactivations - 1:  # -1 is better than adding elec to activated_elecs, then checking if len == thresh
                        propagating_times[i].append(initial_sts[i])
                        "TODO: Analysis on effect of reseting all individual props"
                        # Reset all props in merged prop because spike occurred
                        for j in range(len(prop_m)):
                            reset_prop(j)
                    else:
                        activated_elecs[i].add(elec)

    return propagating_times


def prop_signal(spike_times,
                thresh_freq, seconds_recording, thresh_number_spikes,
                ratio, thresh_cooccurrences_num, thresh_cooccurrences_p, p,
                elec_overlap_p, seq_order_p,
                small_window=0.5, big_window=2, ccg_before=1.5, ccg_after=1.5, ccg_n_bins=61):
    candidate_cohorts = v1_1.scan_reference_electrode(spike_times, thresh_freq, seconds_recording, thresh_number_spikes, ratio,
                                                      small_window=small_window, big_window=big_window,
                                                      ccg_before=ccg_before, ccg_after=ccg_after, ccg_n_bins=ccg_n_bins)
    electrode_cohorts = v1_1.rescan_candidate_cohorts(candidate_cohorts, thresh_cooccurrences_num, thresh_cooccurrences_p, p)
    propagations = get_propagations(electrode_cohorts)
    propagations_merged = merge_propagations(propagations, elec_overlap_p, seq_order_p)

    return propagations_merged

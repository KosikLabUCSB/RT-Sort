from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache, EcephysSession
import numpy as np


def get_best_session_id(cache: EcephysProjectCache, filter_full_genotype: set, filter_ecephys_structure_acronyms: set):
    # Best session is session that passes all filters and has the most amount of units

    sessions = cache.get_session_table()
    filtered = []
    for i in range(len(sessions)):
        sess = sessions.iloc[i]
        for genotype in filter_full_genotype:
            if genotype == sess.full_genotype:
                break
        else:
            filtered.append(False)
            continue

        for acronym in filter_ecephys_structure_acronyms:
            if acronym not in sess.ecephys_structure_acronyms:
                filtered.append(False)
                break
        else:
            filtered.append(True)

    sessions_filtered = sessions[filtered]
    print(f"Number of filtered sessions: {len(sessions_filtered)}")

    session_id_best = None
    unit_count_max = 0
    for session_id in sessions_filtered.index:
        session = sessions_filtered.loc[session_id]
        unit_count = session.unit_count
        if unit_count > unit_count_max:
            session_id_best = session_id
            unit_count_max = unit_count
        print(f"Session ID: {session_id}. {unit_count} units")


def get_best_probe_id(session: EcephysSession, filter_ecephys_structure_acronyms: set):
    # Best probe is probe that passes through the most of the ecephys structures in FILTER_ECEPHYS_STRUCTURE_ACRONYMS
    probe_id_best = None
    num_ecephys_best = 0
    for probe_id in session.probes.index.values:
        num_ecephys = 0
        for struct in session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique():
            if struct in filter_ecephys_structure_acronyms:
                num_ecephys += 1
        if num_ecephys > num_ecephys_best:
            num_ecephys_best = num_ecephys
            probe_id_best = probe_id
    print(num_ecephys_best)
    print(probe_id_best)
    print(session.channels[session.channels.probe_id == probe_id_best].ecephys_structure_acronym.unique())


def get_channel_map(session: EcephysSession, probe_id: int):
    """
    Get a channel map for the channels in a probe

    :param session:
    :param probe_id: int
        ID of probe to get channels from

    :return: channel_map: dict
        {channel_id: (x, y)}
    """

    channels = session.channels[session.channels.probe_id==probe_id]

    # Only get channels with defined LFP
    # lfp = session.get_lfp(probe_id)
    # channel_map = channels.loc[lfp.channel][["probe_horizontal_position", "probe_vertical_position"]]

    channel_map = {chan_id: channels.loc[chan_id][["probe_horizontal_position", "probe_vertical_position"]].tolist()
                   for chan_id in channels.index}

    return channel_map


def get_units(session: EcephysSession, probe_id: int,
              fr_min=0.05, isi_max=0.3, snr_min=5,
              channel_amp_min=0.12):
    """
    Get units detected by Allen Inst for comparison

    :param session:
        Session to take units from
    :param probe_id:
        Probe ID to take units from
    :param fr_min:
        Minimum firing rate
    :param isi_max:
        Maximum ISI violations
    :param snr_min:
        Minimum SNR
    :param channel_amp_min:
        A unit is detected on a channel and the channel is included on its [units x sequences] list
        if its mean waveform amplitude on that channel is at least channel_amp_min * maximum amplitude
        Default is 0.12 because Allen Inst uses that for waveform_spread metric
        (https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/80/75/8075a100-ca64-429a-b39a-569121b612b2/neuropixels_visual_coding_-_white_paper_v10.pdf)

    :return: [units x sequences]: list
        Each element represents a unit and is a list of channel IDs sorted by amplitude
    :return [units x spike_times]: list
        Each element represents a unit and is a list of spike times (in samples)
    ith unit in [units x sequences] corresponds to ith unit in [units x spike_times]

    Note: amplitude is absolute value of trough
    """
    units = session.units[
        (session.units.probe_id == probe_id) &
        (session.units.firing_rate >= fr_min) &
        (session.units.isi_violations <= isi_max) &
        (session.units.snr >= snr_min)
    ]

    sequences = []
    spike_times = []
    for unit_id in units.index:
        templates = session.mean_waveforms[unit_id]  # (n_channels, n_samples)
        amps = np.abs(np.min(templates, axis=1))
        amp_max = max(amps)

        amps_sorted = amps.sortby(amps, ascending=True)
        amp_min_idx = np.searchsorted(amps_sorted, amp_max * channel_amp_min, side="left")
        # channel_ids = amps_sorted[-1:amp_min_idx - 1:-1].channel_id.values.tolist()

        # Plot mean waveforms on each selected channel
        # channel_ids = amps_sorted[-1:amp_min_idx-1:-1].channel_id.values.tolist()
        # import matplotlib.pyplot as plt
        # for chan_id in channel_ids:
        #     plt.plot(templates.sel(channel_id=chan_id))
        # plt.show()
        # exit()

        sequences.append(
            amps_sorted[-1:amp_min_idx-1:-1].channel_id.values.tolist()  # Weird indexing for returning descending-amplitude-sorted channel IDS
        )
        spike_times.append(
            (session.spike_times[unit_id] * units.loc[unit_id].probe_lfp_sampling_rate).tolist()  # Convert from seconds to samples
        )

    return sequences, spike_times


def main():
    cache = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)
    # get_best_session_id(cache, FILTER_FULL_GENOTYPE, FILTER_ECEPHYS_STRUCTURE_ACRONYMS)
    session = cache.get_session_data(SESSION_ID)
    # get_best_probe_id(session, filter_ecephys_structure_acronyms=FILTER_ECEPHYS_STRUCTURE_ACRONYMS)
    # np.save(CHANNEL_MAP_PATH, get_channel_map(session, PROBE_ID))
    unit_sequences_allen, unit_spike_times_allen = get_units(session, PROBE_ID)
    print(len(unit_sequences_allen))
    np.save(UNIT_SEQUENCES_ALLEN_PATH, unit_sequences_allen)
    np.save(UNIT_SPIKE_TIMES_ALLEN_PATH, unit_spike_times_allen)

    # get_inputs(
    #     save_path=THRESH_CROSSINGS_PATH,
    #     lfp_nwb_path=LFP_NWB_PATH,
    #     freq_min=300,
    #     freq_max=3000,
    #     spike_amp_thresh=5,
    #
    # )

    # Find how many times multiple channels have same threshold crossing time
    # threshold_crossings = np.load(THRESH_CROSSINGS_PATH, allow_pickle=True)
    # unique = set()
    # total = 0
    # for channel in threshold_crossings:
    #     total += len(channel)
    #     unique.update(channel)
    # print(total)
    # print(len(unique))

    # units_of_interest = session.units[(session.units.probe_id == PROBE_ID)
    #                                   & (session.units.isi_violations <= 0.3)
    #                                   & (session.units.snr >= 5)]
    # print(len(units_of_interest))
    # unit_id = units_of_interest.index.values[0]
    # waveforms = session.mean_waveforms[unit_id]
    #
    # import numpy as np
    # chan_max = np.min(waveforms, axis=1).argmin()
    # waveform_max = waveforms[chan_max, :]
    #
    # st = session.spike_times[unit_id][-3]
    # lfp = session.get_lfp(PROBE_ID)
    # buffer = 4/1000
    # lfp = lfp.sel(time=slice(st-buffer, st+buffer))
    # # lfp = lfp.sel(channel=lfp.channel[chan_max])
    #
    # import matplotlib.pyplot as plt
    # plt.plot(waveform_max)
    # plt.show()
    #
    # for c in range(lfp.shape[1]):
    #     plt.plot(lfp[:, c])
    # plt.show()


MANIFEST_PATH = "/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/manifest.json"
FILTER_ECEPHYS_STRUCTURE_ACRONYMS = {"VISp", "LP", "CA1", "CA3"}
FILTER_FULL_GENOTYPE = {"wt/wt"}
SESSION_ID = 757216464
PROBE_ID = 769322753
LFP_NWB_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_lfp.nwb"
THRESH_CROSSINGS_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_thresh_crossings.npy"
UNIT_SEQUENCES_ALLEN_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_unit_sequences_allen.npy"
UNIT_SPIKE_TIMES_ALLEN_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_unit_spike_times_allen.npy"
CHANNEL_MAP_PATH = f"/home/mea/SpikeSorting/prop_signal/allen_inst/cache_manifest/session_{SESSION_ID}/probe_{PROBE_ID}_channel_map.npy"

if __name__ == "__main__":
    main()

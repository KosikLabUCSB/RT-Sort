from automated_detection_propagation import automated_detection_propagation
import numpy as np

if __name__ == "__main__":
    """This example requires scipy.io to read data from example_data.mat"""
    from scipy.io import loadmat

    """
    Load example_data.mat which contains example data from
    https://github.com/max-workspace/Propagation-Signal-and-Synaptic-Coupling-Algorithm-Python/blob/main/example/example_data.mat
    """
    example_data_path = "example_data.mat"
    spike_times_raw = loadmat(example_data_path)["spike_times"][0, :]
    # Data must be reformatted
    spike_times = np.array([a[0, :] if a.size > 0 else np.array([]) for a in spike_times_raw], dtype=object)

    np.save("/home/mea/SpikeSorting/prop_signal/allen_inst/example/example_data.npy", spike_times)

    """
    spike_times is the spike times from 120 electrodes. Each element contains the
    spike time from one electrode. The time unit is ms. This recording is 180s in
    total, and 1Hz is used as the thresholding for the spiking frequency of the
    reference electrode. Put None for thres_number_spikes.
    """
    list_of_propagation, time_all = automated_detection_propagation(spike_times, 1, 180, None, 0.5, 50, p=0)
    print(len(list_of_propagation))
    exit()

    """
    Instead of thresholding for spiking frequency of the reference electrode,
    you can also use an absolute number to threshold the total number of
    spikes needed on each electrode. Put None for thres_freq and seconds_recording.
    Add the number you want to use for thres_number_spikes. Here, 180 is used.
    """
    list_of_propagation, time_all = automated_detection_propagation(spike_times, None, None, 180, 0.5, 50, 50)


    """
    For the following parameters, it is recommended to maintain the original values
    because those were the only values thoroughly tested and ensured to produce good results.
    However, these parameters can be changed as shown below
    
    small_window (default value=0.5)
    big_window (default=2)
    ccg_before (default=1.5)
    ccg_after (default=1.5)
    ccg_n_bins (default=61)
    """
    list_of_propagation, time_all = automated_detection_propagation(spike_times, None, None, 180, 0.5, 50, 50,
                                                                    small_window=1.0, big_window=2.2,
                                                                    ccg_before=2.0, ccg_after=2.0, ccg_n_bins=81)

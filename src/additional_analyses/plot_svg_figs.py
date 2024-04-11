import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

CM = 1/2.54
RT_COLOR = "#ffd343"
OTHER_COLOR1 = "#426eff"
OTHER_COLOR2 = ["#7542ff", "#42ccff"]
OTHER_COLOR3 = ["#816fd5", "#d5b46f", "#6fc3d5"]
MARKERS = ["o", "*", 'v', '^']


def plot_hist(load_paths, LOG_Y, CONVERT_TO_FRAME, hist_color, fig_size, xrange, ytic, save_path=0):

    all_file_cont = []

    # for each load path
    for p in load_paths:

        # load data
        file_cont = np.load(p)

        if CONVERT_TO_FRAME:
            file_cont = file_cont/20

        all_file_cont.append(file_cont)

    # flatten into single list
    all_file_cont = [item for sublist in all_file_cont for item in sublist]

    # print distribution mean and std
    print(np.mean(all_file_cont))
    print(np.std(all_file_cont))

    
    # initiate figure
    fig, ax = plt.subplots()

    # set figuresize
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # plot results
    plt.hist(all_file_cont, log=LOG_Y, bins=10, color=hist_color)

    plt.xlim((xrange[0],xrange[-1]))
    plt.xticks(ticks=xrange, labels=[""]*len(xrange))

    if ytic != 0:
        plt.yticks(ticks=ytic, labels=[""] * len(ytic))

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        if LOG_Y == True:
            save_path = save_path + "_log"
        plt.savefig(save_path + ".svg", format="svg")

    # Show the plot
    plt.show()

# # #

def plot_bar_dur(plot_data, bar_color, fig_size, markers, save_path=0):

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figuresize
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # Plot the first 4 values as a bar plot
    plt.bar(range(len(plot_data)-1), plot_data[:-1], color=bar_color[0])

    # for each point
    for i, p in enumerate(plot_data[:-1]):
        plt.scatter(i, np.max(plot_data[:-1])+0.2, c="k", marker=markers[i])

    # Plot the last value as a separate bar, with a different color
    ax.bar(len(plot_data)-1, plot_data[-1], color=bar_color[1])

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    plt.xticks(ticks=[])
    plt.yticks(ticks=[0,1,2,3], labels=["", "", "", ""])
    
    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    print(np.mean(plot_data[:4]))
    print(np.std(plot_data[:4]))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    # Show the plot
    plt.show()

# # #

def plot_pass_speed(load_path, fig_size, line_colors, x_ticks, y_ticks, save_path=0):

    # load data
    np_speeds = np.load(load_path + "NP.npy")
    mea_speeds = np.load(load_path + "MEA.npy")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figure size
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # print times
    print(mea_speeds[1,np.where(mea_speeds[0] == 1020)[0]])
    print(np_speeds[1,np.where(np_speeds[0] == 384)[0]])

    # plot lines at times
    plt.vlines([384, 1020], ymin=[0,0], ymax=[np_speeds[1,np.where(np_speeds[0] == 384)[0]],
            mea_speeds[1,np.where(mea_speeds[0] == 1020)[0]]], colors="k", linewidth=2)

    # plot results
    plt.plot(mea_speeds[0], mea_speeds[1], color=line_colors[0], linewidth=3)
    plt.plot(np_speeds[0], np_speeds[1], color=line_colors[1], linewidth=3)
    plt.scatter(mea_speeds[0], mea_speeds[1], c=line_colors[0], marker="o")
    plt.scatter(np_speeds[0], np_speeds[1], c=line_colors[1], marker="o")

    # print linear fit results
    print_linear_fit_results(mea_speeds[0], mea_speeds[1])
    print_linear_fit_results(np_speeds[0], np_speeds[1])

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    # set ticks
    plt.xticks(ticks=x_ticks, labels=[""]*len(x_ticks))
    plt.yticks(ticks=y_ticks, labels=[""]*len(y_ticks))

    # set axis labels
    plt.xlim((0, 1.05*np_speeds[0, -1]))
    plt.ylim((0, 1.05*np_speeds[1, -1]))
    
    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    # Show the plot
    plt.show()

# # #

def plot_bar_num_detect(plot_data, sort_names, fig_size, bar_color, yticks, save_path=0):

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figuresize
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # Plot the first 4 values as a bar plot
    plt.bar(range(len(plot_data)), plot_data, color=bar_color)

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    plt.xticks(ticks=range(len(plot_data)), labels=sort_names)
    plt.yticks(ticks=yticks, labels=[""]*len(yticks))

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    print(np.mean(plot_data[:4]))
    print(np.std(plot_data[:4]))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    # Show the plot
    plt.show()


def print_linear_fit_results(x, y):

    # Perform the linear regression
    result = stats.linregress(x, y)

    # Print the p-value and R^2 value
    print(f"p-value: {result.pvalue}")
    print(f"R^2 value: {result.rvalue**2}")

    # Print the fitted function
    print(f"Fitted function: y = {result.slope:.3f} x + {result.intercept:.3f}")

# # #

def plot_loss_epoch(load_path_train, load_path_val, remove_last, fig_size, line_colors, x_ticks, y_ticks, save_path=0):

    # load data
    loss_train = np.load(load_path_train, allow_pickle=True)
    loss_val = np.load(load_path_val, allow_pickle=True)

    if remove_last:
        loss_train = loss_train[:-1]
        loss_val = loss_val[:-1]

    # find the mean and std of the number of epochs
    array_lengths = np.array([len(arr) for arr in loss_val])
    print(array_lengths)
    av_epoch = np.mean(array_lengths)
    std_epoch = np.std(array_lengths)

    print(av_epoch)
    print(std_epoch)

    # find the maximum length
    max_len = np.max([np.max([len(i) for i in loss_train]), np.max([len(i) for i in loss_val])])

    # padd data up to max_len
    padded_loss_train = np.array([np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in loss_train])
    padded_loss_val = np.array([np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in loss_val])

    # Compute the mean and standard deviation for each index over all arrays
    means_train = np.mean(padded_loss_train, axis=0)
    stds_train = np.std(padded_loss_train, axis=0)
    means_val = np.mean(padded_loss_val, axis=0)
    stds_val = np.std(padded_loss_val, axis=0)
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figure size
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # Plot the mean with error bars indicating the standard deviation
    plt.errorbar(range(max_len), means_train, yerr=stds_train, color=line_colors[0], linewidth=3)
    plt.errorbar(range(max_len), means_val, yerr=stds_val, color=line_colors[1], linewidth=3)


    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    # set ticks
    plt.xticks(ticks=x_ticks, labels=[""] * len(x_ticks))
    plt.yticks(ticks=y_ticks, labels=[""] * len(y_ticks))

    # set axis limits
    plt.xlim((0, av_epoch))
    plt.ylim((0, np.max([means_train[0]+2*stds_train[0], means_val[0]+2*stds_val[0]])))

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    plt.show()

# # #

def plot_perf_thresh(load_path, fig_size, line_colors, thresholds, x_ticks, y_ticks, save_path=0):

    # load data
    perf_data = np.load(load_path, allow_pickle=True)

    # compute mean over models
    perf_mean = np.mean(perf_data, axis=1)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figure size
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # mark the loose and stringent thresholds
    plt.vlines(thresholds, ymin=[0, 0], ymax=[100, 100], colors="k", linewidth=2)

    # Plot the mean for the different performance metrics
    plt.errorbar(perf_mean[:, 0], perf_mean[:, 2], color=line_colors[0], linewidth=3)
    plt.errorbar(perf_mean[:, 0], perf_mean[:, 3], color=line_colors[2], linewidth=3)
    plt.errorbar(perf_mean[:, 0], perf_mean[:, 1], color=line_colors[1], linewidth=3)

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    # set ticks
    plt.xticks(ticks=x_ticks, labels=[""] * len(x_ticks))
    plt.yticks(ticks=y_ticks, labels=[""] * len(y_ticks))

    # set axis limits
    plt.xlim((0, 100))
    plt.ylim((0, 105))

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    plt.show()
    
# # #

if __name__ == "__main__":

    # plot deviation dist
    plot_hist(["./data_files/abs_deviations_MEA.npy"], True, True, OTHER_COLOR2[0], [2,2], [0,0.4], 0)
    plot_hist(["./data_files/abs_deviations_MEA.npy"], False, True, OTHER_COLOR2[0], [3,3], [0,0.4], [0,7000])
    plot_hist(["./data_files/abs_deviations_neuropixel.npy"], True, True, OTHER_COLOR2[0], [2,2], [0,0.4], 0)
    plot_hist(["./data_files/abs_deviations_neuropixel.npy"], False, True, OTHER_COLOR2[0], [3,3], [0,0.4], [0,4000])

    # plot overlap scores
    plot_hist(["./data_files/sim_ground_truth_recall.npy"], False, False, OTHER_COLOR2[0], [3,2], [0,1], 0)

    # plot simulated ground truth performance
    plot_hist(["./data_files/rt_sim_ground_truth_precision.npy"], False, False, OTHER_COLOR2[0], [3, 2], [0,1], [0, 30, 60])
    plot_hist(["./data_files/rt_sim_ground_truth_recall.npy"], False, False, OTHER_COLOR2[0], [3,2], [0,1], [0, 10, 20])
    plot_hist(["./data_files/rt_sim_ground_truth_spikeinterface_formula.npy"], False, False, OTHER_COLOR2[0], [3,2], [0,1], [0, 10, 20])

    plot_hist(["./data_files/sim_times_rt_unit_precision.npy"], False, False, OTHER_COLOR2[0], [3, 2], [0, 1], [0, 50, 100])
    plot_hist(["./data_files/sim_times_rt_unit_recall.npy"], False, False, OTHER_COLOR2[0], [3, 2], [0, 1], [0, 10, 20])
    plot_hist(["./data_files/sim_times_rt_unit_spikeinterface_formula.npy"], False, False, OTHER_COLOR2[0], [3, 2], [0, 1], [0, 10, 20])

    # plot sorting only durations
    plot_hist(["./data_files/patch_cell3_sorting_computation_times.npy", "./data_files/patch_cell7_sorting_computation_times.npy"], False, False, OTHER_COLOR2[0], [2,2], [0.4, 1.4], [0,20000,40000])
    plot_hist(["./data_files/sim_sorting_computation_times.npy"], False, False, OTHER_COLOR2[1], [2,2], [0, 4], [0,10000,20000])

    # plot total durations
    plot_hist(["./data_files/patch_cell3_sorting_delays.npy", "./data_files/patch_cell7_sorting_delays.npy"], False, False, OTHER_COLOR2[0], [2, 2], [3, 13], [0, 1000, 2000])
    plot_hist(["./data_files/sim_sorting_delays.npy"], False, False, OTHER_COLOR2[1], [2, 2], [3, 13], [0, 20000, 40000])
    plot_hist(["./data_files/np_sorting_delays.npy"], False, False, OTHER_COLOR2[0], [2, 2], [3, 13], [0, 40000, 80000])

    # plot distribution of hartigans dip test p-values
    plot_hist(["./data_files/amp_split_dip_pvals.npy"], False, False, OTHER_COLOR2[0], [3, 2], [0,0.1, 1], [0, 150])

    # pre-rec processing durations
    plot_bar_dur(np.asarray([315/3.10, 298/3.09, 151/1.21, 97/0.69, 772/5])/60, OTHER_COLOR2, [3,2], MARKERS)

    # # plot computation speed for forward pass
    plot_pass_speed("./data_files/model_speed_", [2.5,2], OTHER_COLOR2, [0,1000,2000], [0,2,4])

    # # plot loss as function of epoch
    plot_loss_epoch("./data_files/all_training_train_losses_MEA.npy", "./data_files/all_training_val_losses_MEA.npy", True, [2.5,2], OTHER_COLOR2, [0,10,20,30], [0,2.5,5])
    plot_loss_epoch("./data_files/all_training_train_losses_NP.npy", "./data_files/all_training_val_losses_NP.npy", False, [2.5,2], OTHER_COLOR2, [0,5,10,15,20,25,30], [0,2.5,5])

    # plot detection peformance as function of threshold
    plot_perf_thresh("./data_files/all_loc_prob_thresh_tuning_MEA.npy", [3,2], OTHER_COLOR3, [27.5, 10], [0,50,100], [0,50,100])
    plot_perf_thresh("./data_files/all_loc_prob_thresh_tuning_NP.npy", [3,2], OTHER_COLOR3, [17.5, 7.5], [0,50,100], [0,50,100])

    # plot number of detected units per sorter
    plot_bar_num_detect([168,210,317,233,446,628,187], ["RT","HS","HDS","IC","KS","SC","TDC"], [3,2], OTHER_COLOR2[0], [0,300,600], 0)
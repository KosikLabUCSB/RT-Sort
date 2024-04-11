import matplotlib.pyplot as plt
import numpy as np

# This script is used to plot the performance of RT-Sort assuming kilosort2 detections as ground truth. The performance
# is plotted as a function of the ith highest amplitude in the footprint of each detected unit.

# specify paths to stored data
DATA_PATHS_MEA = []

DATA_PATHS_NP = []

NUM_ELEC = 6
OTHER_COLOR2 = ["#7542ff", "#42ccff"]


def compute_ks_perf(data_paths):

    # initiate empty result arrays and dictionaries
    all_precision = {}
    all_recall = {}

    av_precision = np.zeros((NUM_ELEC,1))
    av_recall = np.zeros((NUM_ELEC,1))
    std_precision = np.zeros((NUM_ELEC,1))
    std_recall = np.zeros((NUM_ELEC,1))

    # for each dataset
    for i, dp in enumerate(data_paths):

        # load data
        data_cont = np.load("{}/units_highest_elecs_tp_fp_fn.npy".format(dp))

        # for each electrode
        for elec in range(NUM_ELEC):

            # compute precision and recall
            precision_elec = 100 * data_cont[:,elec,0] / (data_cont[:,elec,0] + data_cont[:,elec,1])
            recall_elec = 100 * data_cont[:,elec,0] / (data_cont[:,elec,0] + data_cont[:,elec,2])

            if i == 0:

                all_precision[elec] = precision_elec
                all_recall[elec] = recall_elec

            else:
                
                all_precision[elec] = np.concatenate((all_precision[elec], precision_elec))
                all_recall[elec] = np.concatenate((all_recall[elec], recall_elec))

    # for each dataset
    for dk in all_precision.keys():

        # compute mean and std
        av_precision[dk] = np.nanmean(all_precision[dk])
        std_precision[dk] = np.nanstd(all_precision[dk])
        av_recall[dk] = np.nanmean(all_recall[dk])
        std_recall[dk] = np.nanstd(all_recall[dk])

    # plot line plots
    plot_line_fig([av_precision, av_recall], [std_precision, std_recall], [3,2], OTHER_COLOR2, [1,4,NUM_ELEC], [0,100])

    # plot distribution of recall scores
    plot_hist(all_recall[0], OTHER_COLOR2[0], [3 ,2], [0,100], [0,50,100], [0, 200, 400])
    plot_hist(all_recall[3], OTHER_COLOR2[0], [3, 2], [0, 100], [0, 50, 100], [0, 50, 100])
    
# # #

def plot_line_fig(mean_vals, std_vals, fig_size, line_colors, x_ticks, y_ticks, save_path=0):

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # set figure size
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # Plot the mean with error bars indicating the standard deviation
    plt.errorbar(range(1, len(mean_vals[1]) + 1), np.squeeze(mean_vals[1]), yerr=np.squeeze(std_vals[1]), color=line_colors[1], linewidth=3)
    plt.errorbar(range(1,len(mean_vals[0])+1), np.squeeze(mean_vals[0]), yerr=np.squeeze(std_vals[0]), color=line_colors[0], linewidth=3)

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
    plt.xlim((0.5, x_ticks[-1]+0.5))
    plt.ylim((0.5, y_ticks[-1]+0.5))

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_path != 0:
        plt.savefig(save_path + ".svg", format="svg")

    plt.show()

# # #

def plot_hist(plot_data, hist_color, fig_size, xrange, x_ticks, y_ticks, save_path=0):

    # initiate figure
    fig, ax = plt.subplots()

    # set figuresize
    fig.figure.set_size_inches(fig_size[0], fig_size[1])

    # plot results
    plt.hist(plot_data, bins=20, color=hist_color)

    # adjust axes
    plt.xlim((xrange[0], xrange[1]))
    plt.xticks(ticks=x_ticks, labels=[""] * len(x_ticks))
    plt.yticks(ticks=y_ticks, labels=[""] * len(y_ticks))

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
        plt.savefig(save_path + ".svg", format="svg")

    # Show the plot
    plt.show()

# # #


if __name__ == "__main__":

    # compute results for DATA_PATHS_MEA or DATA_PATHS_NP
    compute_ks_perf(DATA_PATHS_MEA)
    compute_ks_perf(DATA_PATHS_NP)
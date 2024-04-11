import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

CM = 1/2.54
RT_COLOR = "#ffd343"
OTHER_COLOR1 = "#426eff"
OTHER_COLOR2 = ["#7542ff", "#42ccff"]
MARKERS = ["o", "*", 'v', '^']

PATCH_ALL = np.asarray([210, 328, 63, 39])
OFFLINE_FN = np.asarray([12,6,10,5])
OFFLINE_FP = np.asarray([0,0,6,0])
ONLINE_FN = np.asarray([8,2])
ONLINE_FP = np.asarray([0,12])

MODEL_MEA_PRE = [95.1, 94.6, 96.2, 88.4, 92.0, 87.3]
MODEL_MEA_REC = [91.1, 86.8, 92.4, 86.6, 85.3, 100]
MODEL_NP_PRE = [81.1, 77.6, 95.8, 95.9, 77.8, 90.4]
MODEL_NP_REC = [87.6, 73.1, 88.8, 90.9, 85.0, 82.7]
RMS_MEA_PRE = [98.9, 96, 94.6, 94.4, 88.4, 97.7]
RMS_MEA_REC = [69.1, 61.1, 63, 54.1, 57.8, 76.4]
RMS_NP_PRE = [79.8, 84.9, 79, 85.9, 75.2, 73.3]
RMS_NP_REC = [66.6, 60.4, 69.7, 67.3, 65.4, 60.9]

def set_basic_plot_settings(fig):

    # Set the default font family and size
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 5

    ax = fig.get_axes()[0]

    # hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # hide title
    ax.title.set_visible(False)

    # increase thickness of the bottom and left spines
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)

    # increase thickness of tick marks
    ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

    return ax

# # #

def plot_prec_rec(plot_vals, bar_colors, markers, width, height, save_path=0):

    # compute ttest
    print(ttest_rel(plot_vals[0], plot_vals[1]))
          
    # define figure size
    fig = plt.figure(figsize=(width, height))

    # plot bar plot
    plt.bar([0,1], [np.mean(plot_vals[0]), np.mean(plot_vals[1])], color = bar_colors)

    print([np.mean(plot_vals[0]), np.mean(plot_vals[1])])
    print([np.std(plot_vals[0]), np.std(plot_vals[1])])

    offsets = np.linspace(-len(plot_vals[0])/20, len(plot_vals[0])/20, len(plot_vals[0]))

    # for each value to plot
    for i in range(len(plot_vals[0])):

        # plot result
        plt.scatter(0+offsets[i], plot_vals[0][i], c="k", marker=markers[i])
        plt.scatter(1+offsets[i], plot_vals[1][i], c="k", marker=markers[i])

    # set basic plot parameters
    ax = set_basic_plot_settings(fig)

    # adjust ticks
    plt.xticks(ticks=[])
    plt.yticks(ticks=[0,100], labels=["",""])
    
    plt.tight_layout()

    # save results
    if save_path != 0:
        plt.savefig(save_path, format="svg")
        print("Figure saved at: {}".format(save_path))

    plt.show()


# # #

if __name__ == "__main__":

    # compute patch performance
    offline_pre = 100 * (PATCH_ALL - OFFLINE_FN) / (PATCH_ALL - OFFLINE_FN + OFFLINE_FP)
    offline_rec = 100 * (PATCH_ALL - OFFLINE_FN) / PATCH_ALL
    online_pre = 100 * (PATCH_ALL[:len(ONLINE_FP)] - ONLINE_FN) / (PATCH_ALL[:len(ONLINE_FP)] - ONLINE_FN + ONLINE_FP)
    online_rec = 100 * (PATCH_ALL[:len(ONLINE_FP)] - ONLINE_FN) / PATCH_ALL[:len(ONLINE_FP)]

    # plot patch ground truth performance figures
    plot_prec_rec([offline_pre, offline_rec], OTHER_COLOR2, MARKERS, 2 * CM, 4 * CM, 0)
    plot_prec_rec([online_pre, online_rec], OTHER_COLOR2, MARKERS[:2], 2 * CM, 4 * CM, 0)

    # plot performance of model
    plot_prec_rec([MODEL_MEA_PRE, RMS_MEA_PRE], [RT_COLOR, OTHER_COLOR1], MARKERS[0]*len(MODEL_MEA_PRE), 2 * CM, 4 * CM, 0)
    plot_prec_rec([MODEL_MEA_REC, RMS_MEA_REC], [RT_COLOR, OTHER_COLOR1], MARKERS[0]*len(MODEL_MEA_REC), 2 * CM, 4 * CM, 0)
    plot_prec_rec([MODEL_NP_PRE, RMS_NP_PRE], [RT_COLOR, OTHER_COLOR1], MARKERS[0]*len(MODEL_NP_PRE), 2 * CM, 4 * CM, 0)
    plot_prec_rec([MODEL_NP_REC, RMS_NP_REC], [RT_COLOR, OTHER_COLOR1], MARKERS[0]*len(MODEL_NP_REC), 2 * CM, 4 * CM, 0)
    



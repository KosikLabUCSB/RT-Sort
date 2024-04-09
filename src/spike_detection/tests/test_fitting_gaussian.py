import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import norm

output = np.load(r"E:\KosikLab\Deep-Learning-for-Realtime-Spike-Sorting\data\outputs.npy")
loc_scores = output[1:-1]
loc_probs = np.exp(loc_scores) / np.sum(np.exp(loc_scores))
loc_probs *= 100

# loc_probs_x = np.linspace(start=0, stop=200, num=1000)
loc_probs_x = np.arange(loc_probs.size)

# loc_probs = norm.pdf(loc_probs_x, loc=50, scale=10)
# loc_probs += norm.pdf(loc_probs_x, loc=150, scale=5)

gmm = BayesianGaussianMixture(n_components=2, weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=1e5)
gmm = gmm.fit(X=np.concatenate([loc_probs_x[:, None], loc_probs[:, None]]))
colors = ["blue", "green", "orange", "red", "black"]

plt.plot(loc_probs_x, loc_probs, color='black')
for m, c in zip(gmm.means_.ravel(), gmm.covariances_.ravel()):
    x = np.arange(loc_probs_x.size)
    y = np.max(loc_probs) * np.exp(-(x - m) ** 2. / (2. * np.sqrt(c) ** 2.))
    plt.plot(x, y, color=colors.pop(0))


plt.show()
print()

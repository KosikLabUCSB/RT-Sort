"""
Visualize the weights of the model
"""


from src.model import ModelSpikeSorter
import matplotlib.pyplot as plt
from math import ceil
import numpy as np


def visualize_weights(model: ModelSpikeSorter, num_per_row=5):
    for param in model.parameters():
        if len(param.shape) == 3:
            weights = param.data.cpu().numpy()

            print(weights.shape)
            num_weights = len(weights)

            fig, subplots = plt.subplots(ceil(num_weights/num_per_row), min(num_weights, num_per_row), figsize=(10, 10), tight_layout=True)
            subplots = np.atleast_2d(subplots)
            for i in range(num_weights):
                weight = weights[i]
                weight = np.mean(weight, axis=0)

                row = i // num_per_row
                col = i % num_per_row
                subplots[row, col].plot(weight)

            plt.show()


def main():
    MODEL = ModelSpikeSorter.load("/data/MEAprojects/DLSpikeSorter/models/v0_3/2954")
    visualize_weights(model=MODEL)


if __name__ == "__main__":
    main()

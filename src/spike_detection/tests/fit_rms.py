"""
Test training a fully-connected neural network to calculating RMS of sample
"""

import torch.nn as nn
import torch


def main(input_size, batch_size, num_iter):
    model = nn.Sequential(
        nn.Linear(input_size, 200),
        nn.ReLU(),
        nn.Linear(200, 1)
    )
    # model = nn.Sequential(
    #     nn.Conv1d(1, 50, 100, 100),
    #     nn.ReLU(),
    #     nn.Conv1d(50, 1, 2),
    #     nn.Flatten()
    # )

    optim = torch.optim.Adam(model.parameters(), 3e-4)
    loss_fn = nn.L1Loss()

    for it in range(num_iter):
        for param in model.parameters():
            param.grad = None

        inputs = torch.rand(batch_size, input_size)

        labels = torch.sqrt(torch.mean(torch.square(inputs), dim=1, keepdim=True))

        outputs = model(inputs)  # For linear
        # outputs = model(inputs[:, None, :])  # For conv

        loss = loss_fn(outputs, labels)
        loss.backward()

        optim.step()

        diff = torch.mean(torch.abs(outputs-labels)/labels) * 100
        print(f"{it+1}/{num_iter}: Loss = {loss:.3f} | % diff = {diff:.1f}%")

    print(labels)
    print(outputs)


if __name__ == "__main__":
    main(input_size=200, batch_size=1, num_iter=500)

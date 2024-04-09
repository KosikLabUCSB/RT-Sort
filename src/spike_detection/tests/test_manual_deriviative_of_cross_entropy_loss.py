import torch

outputs = torch.tensor([[43, 12, 52, 26],
                        [10, 20, 15, 30],
                        [30, 10, 10, 15]],
                       dtype=torch.float, requires_grad=True)
labels = torch.tensor([1, 3])
wf_samples = [0, 1]

loss = torch.nn.CrossEntropyLoss()
l = loss(outputs[wf_samples], labels)
l.backward()
real = outputs.grad[:2, :]

grads = torch.softmax(outputs.detach(), dim=1)
grads[wf_samples, labels] -= 1
grads = grads[:2, :] / len(wf_samples)
print(outputs)


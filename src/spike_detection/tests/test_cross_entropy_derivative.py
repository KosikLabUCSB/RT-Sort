import torch

y_hat = torch.tensor([[10, 20, 30]], dtype=torch.float, requires_grad=True)
y_pred_prob = torch.tensor([[1, 1, 1]], dtype=torch.float)


# PyTorch derivative
loss = torch.nn.CrossEntropyLoss()
l = loss(y_hat, y_pred_prob)
l.backward()
print(y_hat.grad)

# My derivative
probs = torch.softmax(y_hat.detach(), dim=1)
ind_of_correct_classes = torch.nonzero(y_pred_prob, as_tuple=True)
grads = torch.sum(probs * y_pred_prob[ind_of_correct_classes][:, None], dim=0)
grads[ind_of_correct_classes[1]] -= y_pred_prob[ind_of_correct_classes]
print(grads)


# Test gradient descent
loss = torch.nn.CrossEntropyLoss()
for _ in range(10000):
    y_hat.grad = None
    l = loss(y_hat, y_pred_prob)
    l.backward()
    with torch.no_grad():
        y_hat -= 1 * y_hat.grad
print(torch.softmax(y_hat, dim=1))
print(y_hat)






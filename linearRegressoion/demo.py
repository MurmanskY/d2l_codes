import torch

X = torch.tensor([[1, 2, 3], [1, 2, 3]])

Y = torch.arange(2)

Z = torch.matmul(X, Y)

print(X)
print(Y)
print(Z)
print(type(Z))

'''
pytorch_intro.py
Joe Nguyen | 06 Aug 2020
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

What is PyTorch?
- A replacement for NumPy to use the power of GPUs
- a deep learning research platform that provides maximum flexibility and speed
'''

from __future__ import print_function
import torch
import numpy as np

# declare uninitialised matrix: contains allocated values in memory
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3, 10])
x = torch.tensor([[5.5, 3, 10], [1, 2, 3]])
print(x)

# Create tensor based on existing tensor
x = x.new_ones(5, 3, dtype=torch.double)    # new_* methods take in sizes
print(x)
x = torch.rand_like(x, dtype=torch.double)    # override dtype, inherit size
print(x)
print(x.size())

# Operations
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

# with output
z = torch.add(x, y)
print(z)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Inplace
# Any operation that mutates a tensor in-place is post-fixed with an _.
# For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print(y)

# Resize / reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1)
a = x.view(-1, 8)
print(x)
print(y)
print(z)
print(a)

# Convert one-element tensors to scalar
x = torch.rand(1)
print(x, x.item())

# Tensor to numpy
a = torch.ones(5)
b = a.numpy()
print(a, b)

a.add_(2)
print(a, b)

# Convert numpy to tensor
x = np.ones(5)
y = torch.from_numpy(x)
print(x, y)

np.add(x, 1, out=x)
print(x, y)





'''
pytorch_autograd.py
Joe Nguyen | 06 Aug 2020
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

The autograd package provides automatic differentiation for all operations on
Tensors. It is a define-by-run framework, which means that your backprop is
defined by how your code is run, and that every single iteration can be
different.
'''

from __future__ import print_function
import torch
import numpy as np

# Create tensor and track computation
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# # Update requires_grad inplace
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

# Gradients: backpropagate
print(out)
out.backward()

# Print gradients d(out)/dx
print(x.grad)
# y.backward() # RuntimeError: grad can be implicitly created only for scalar outputs


# vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
y.data.norm()

while y.data.norm() < 1000:
    y *= 2
print(y)

# Gradient
# y.backward()    # RuntimeError: grad can be implicitly created only for scalar outputs

out = y.mean()
out.backward(retain_graph=True)
print(x.grad)

# vector-Jacobian product
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)


# Stop autograd from tracking history on Tensors
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# -> detach to get a new Tensor with the same content
# but doesn't require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

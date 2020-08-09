'''
pytorch_nn.py
Joe Nguyen | 06 Aug 2020
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Neural Network
- `nn.Module` contains layers, and a method `forward(input)` that returns the
`output`
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # 1 input image channel, 6 output channels,
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # affine operation: y = Wx + b
        # hidden layers (120, 84)
        # output layer (10)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6x6 image
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square, you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # resize
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
net.fc1

# NN parameters (activations and weights)
params = list(net.parameters())
print(params)
print(len(params))
print(params[0].size())  # conv1's .weight

for i in range(len(params)):
    print(params[i].size())

print(params[-1])
print(params[-2])

# Input 32x32 image
input = torch.randn(1, 1, 32, 32)
input.size()

# Feedforward
output = net(input)
print(output)

# Zero the gradient buffers of all parameters
net.zero_grad()

# backprop with random gradients
output.backward(torch.randn(1, 10), retain_graph=True)

# Loss function
target = torch.randn(10)  # dummy target
target.shape
target = target.view(1, -1)
output.shape
target.shape
target
target2 = target.view(-1, 1)
target2.shape
target2

criterion = nn.MSELoss()
loss = criterion(output, target)
loss
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0])  # ReLU

# Backprop
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward(retain_graph=True)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update weights: weight -= learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(learning_rate * f.grad.data)

optimiser = optim.SGD(net.parameters(), lr=0.01)

# in training loop:
optimiser.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimiser.step()  # update


# if __name__ == '__main__':
#     net = Net()
#     print(net)

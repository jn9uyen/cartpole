'''
pytorch_cls.py
Joe Nguyen | 06 Aug 2020
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Neural Network Image Classifier
-------------------------------
1. Load and normalize the CIFAR10 training and test datasets using torchvision
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img, unnormalise=False, path_img='./figures/image.png'):
    if unnormalise:
        img = img / 2 + 0.5

    img_np = img.numpy()
    img_t = np.transpose(img_np, (1, 2, 0))
    plt.imshow(img_t)
    plt.savefig(path_img)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels (rgb)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train_nn(net, path_model='./models/cifar_net.pth'):

    # Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # data is a list [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print stats
            running_loss += loss.item()
            if i % 2000 == 1999:  # every 2000 mini-batches
                print(
                    f'Epoch: {epoch + 1}, iter: {i + 1},',
                    f'loss: {running_loss / 2000:.2f}'
                )

    print('Finished training')

    # Save trained model
    torch.save(net.state_dict(), path_model)


def main(path_model='./models/cifar_net.pth'):

    net = Net()

    if os.path.isfile(path_model):
        # Load model
        net.load_state_dict(torch.load(path_model))
    else:
        # Train model
        train_nn(net, path_model)

    # Predict on test set
    # View sample input image
    data_iter = iter(testloader)
    images, labels = data_iter.next()
    imshow(
        torchvision.utils.make_grid(images),
        unnormalise=True, path_img='./figures/test_groundtruth.png'
    )
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print('Ground truth:\t', ' | '.join([classes[l] for l in labels]))

    outputs = net(images)
    _, pred = torch.max(outputs, 1)
    print('Predicted:\t', ' | '.join([classes[l] for l in pred]))

    # predict on all test samples
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class-level predictions
            for i, label in enumerate(labels):
                class_total[label] += 1
                class_correct[label] += (predicted == labels)[i].item()

    print(f'Accuracy on {total} test images: {correct / total * 100:.2f}%')
    for i in range(10):
        print(
            f'Accuracy of {classes[i]}:',
            f'{class_correct[i] / class_total[i] * 100:.2f}%'
        )

    # return class_correct, class_total
    return None


if __name__ == '__main__':
    main()


def misc():
    # Model weights
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimiser.state_dict():
    #     print(var_name, "\t", optimiser.state_dict()[var_name])

    # View weights
    net.state_dict()['conv1.weight'][1].shape

    # Weights as image
    img = net.state_dict()['conv1.weight']
    # img = torchvision.utils.make_grid(images)
    img.shape[0]
    img.shape
    img_np = img.numpy()
    img_np.shape
    img_t = np.transpose(img_np, (1, 2, 0))
    img_t.shape

    for i in range(img.shape[0]):
        imshow(img[i], path_img=f'./figures/conv1_wgt_{i}.png')

    path_img = './figures/fc1_wgt.png'
    fc1_wgt = net.state_dict()['fc1.weight']
    plt.imshow(fc1_wgt)
    plt.savefig(path_img)
    fc1_wgt.shape
    fc1_wgt.max()
    fc1_wgt.min()

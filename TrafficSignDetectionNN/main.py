from torch.jit import _script as optim

from train import train_model
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc
import pandas as pd

import os
import pickle


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#Test Comment for git commit and push

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # parameters: 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1,6, 5)
        # second convolution operation, 6 inputs,  16 outputs, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)

        #fully connected layers

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.flatten = nn.Flatten()
        # self.stack = nn.Sequential(
        #     nn.Linear(36, 40),
        #     nn.ReLU(),
        #     nn.Linear(40, 40),
        #     nn.Sigmoid()
        #)

    def forward(self, x):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 3x3 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = nnFunc.relu(self.conv1(x))

        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = nnFunc.relu(c1, (2, 2))

        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = nnFunc.relu(self.conv2(s2))

        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = nnFunc.max_pool2d(c3, 2)

        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)

        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = nnFunc.relu(self.fc1(s4))

        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = nnFunc.relu(self.fc2(f5))

        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)

        return output

            # self.flatten(x)
            # return self.stack(x)

    def get_stack(self):
        return self.stack

# def train_model(model):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim
#     model.train()

def main():

    #define hyperparameters
    epochs = 100
    batch_size = 256
    learning_rate = 1

    #load data from pickle file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    file_dir = os.path.join(dir_path, "data\\data8.pickle")
    imagesfrompkl = open(file_dir, "rb")
    images = pickle.load(imagesfrompkl)

    train_items = loadall(imagesfrompkl)

    #X_train, X_test, y_train, y_test = images['x_train'], images['x_test'], images['y_train'], images['y_test']

    train_dataset_from_loader = torch.utils.data.DataLoader(train_items, batch_size=batch_size)

    net = Network()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # record data for determining performance of network
    perf_metrics = pd.DataFrame(columns=['epoch', 'loss','Accuracy','Precision','Recall'])



    for i in range(epochs):
        net.train()

        for j, (inputs, true_class) in train_dataset_from_loader:
            # Begin forward pass
            output = net(inputs)
            loss = loss_fn(output, true_class)

            # Begin backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # print(net)
    #
    # params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())  # conv1's .weight
    #
    # input = torch.randn(1, 1, 32, 32)



    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))



    #Run Train function from train.py
    #my_network = Network()
    train_model()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

from torch.jit import _script as optim

import train
import torch
import torch.nn as nn


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#Test Comment for git commit and push

class Model(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(36, 40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.flatten(x)
        return self.stack(x)

    def get_stack(self):
        return self.stack

def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim
    model.train()

def main():
    my_model = Model()
    train.train_model(my_model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

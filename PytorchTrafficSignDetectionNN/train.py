import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def plot_images(images, labels, num_images=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        image = np.transpose(images[i], (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        plt.imshow(image)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()


def train_model():
    # print("Training model...")
    print("Loading data...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    file_dir = os.path.join(dir_path, "data\\data8.pickle")
    imagesfrompkl = open(file_dir, "rb")
    images = pickle.load(imagesfrompkl)

    X_train, X_test, y_train, y_test = images['x_train'], images['x_test'], images['y_train'], images['y_test']



    #model.fit(X_train, y_train)


    # TODO: need to convert image vectors into pytorch arrays
    # TODO: create neural network dimensions
    # TODO: set learning hyperparameters e.g(learning rate, epochs, batch size)
    # TODO: train NN on images using CPU
    # TODO: Check NN performance, precision, accuracy, f1 score
    # TODO: Performance modifiers: Cross validation, dropout layers, different network architectures (hidden layers, nodes per layer, etc)
    # TODO: get CUDA to work for training
    # TODO: webcam detection system, opencv?
    # TODO: figure out how to use NN in real time with webcam
    # TODO: print out label to command line

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
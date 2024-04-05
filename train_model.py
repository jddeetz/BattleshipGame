#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""This code is intended to be used via command line to train the neural network model, as defined in ./model/model_definition.py
"""

import numpy as np
import pickle
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Subset

from model.model_definition import BattleshipModel

# Defines global constants used for training the neural network model
BATCH_SIZE = 100
TEST_SET_FRACTION = 0.1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

def load_data(filepath:str) -> np.array:
    """This is just a helper function to load the pickled training data.

    Args:
        filepath: the string like path to the data
    """
    with open(filepath, "rb") as input_file:
        data = pickle.load(input_file)
    return data

def train_test_split(X:torch.Tensor, y:torch.Tensor):
    """This takes the two tensors (X and y) and randomly splits them into training and test datasets.

    Args:
        X, y: pytorch tensors

    Returns:
        train, test: pytorch DataLoader objects
    """
    # Get indices corresponding to the train and test sets
    train_idx, test_idx = random_split(np.arange(X.shape[0]), [1 - TEST_SET_FRACTION, TEST_SET_FRACTION])
    # Get the training and test sets for X and y
    X_train, X_test = Subset(X, train_idx), Subset(X, test_idx)
    y_train, y_test = Subset(y, train_idx), Subset(y, test_idx)
    # Get the data loaders for the training data and test data
    train = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=BATCH_SIZE)
    test = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=BATCH_SIZE)

    return train, test

def train_model() -> None:
    """ This function trains the neural network model to predict ship positions on Battleship boards.
    """
    #### Step 1: Load the masked (X/input data) and the clear game boards (y/output data).
    masked_boards = load_data("./data/fog_boards.pkl")
    unmasked_boards = load_data("./data/no_fog_boards.pkl")

    #### Step 2: Make the data into a pytorch tensor with the correct shape: (num_samples, num_channels = 1, 10, 10)
    num_boards = masked_boards.shape[0]
    masked_boards = torch.from_numpy((masked_boards.reshape(num_boards, 1, 10, 10)).astype(np.single))
    unmasked_boards = torch.from_numpy((unmasked_boards.reshape(num_boards, 1, 10, 10)).astype(np.single))

    #### Step 3: Separate the data into a training/test set
    train, test = train_test_split(masked_boards, unmasked_boards)

    #### Step 4: Define the model, the loss function and the optimizer
    model = BattleshipModel()
    loss_fn = MSELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    #### Step 5: Train the NN model over NUM_EPOCHS
    for epoch_num in range(NUM_EPOCHS):
        print("Beginning epoch {}".format(epoch_num))
        # Keep track of loss of each batch
        batch_loss = []
        # Loads batches of X and y data
        for X_train, y_train in train:
            # Forward pass: compute predicted game board, given the masked one
            y_pred = model(X_train)

            # Compute and print MSE loss.
            loss = loss_fn(y_pred, y_train)
            batch_loss.append(loss.item())
            print("MSE of batch #{} = {}".format(len(batch_loss), batch_loss[-1]))

            # Zero gradients, because otherwise they accumulate
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update parameters
            optimizer.step()

        # Evalate the NN Model against the test set
        # Calling model.eval to set batch normalization layers to evaluation mode
        model.eval()
        test_batch_loss = []
        for X_test, y_test in test:
            y_pred = model(X_test)
            test_batch_loss.append(loss_fn(y_pred, y_test).item())

        # Print the average MSE loss of all batches
        print("Average MSE of all batches in TRAIN set = {}".format(np.mean(batch_loss)))
        print("Average MSE of all batches in TEST set = {}".format(np.mean(test_batch_loss)))
        model.train()

    #### Step 6: Save the NN Model
    print("Saving model to disk")
    torch.save(model.state_dict(), "model/Battleship_Model.pt")

if __name__ == '__main__':
    train_model()
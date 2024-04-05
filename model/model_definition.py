#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
Defines architecture of the CNN model.
"""

import torch.nn as nn
from torchsummary import summary

class Conv(nn.Module):
    """Conv2d => Batch Normalization => ReLU or Sigmoid
    
    """
    def __init__(self, in_channels, out_channels, activation_function):
        super().__init__()
        if activation_function == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation_function == "tanh":
            self.activation = nn.Tanh()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1,1), padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  self.activation)

    def forward(self, x):
        return self.conv(x)

class BattleshipModel(nn.Module):
    """
    Definition of the Battleship Convolutional Neural Net
    """
    def __init__(self):
        # Inherit all of the __init__ properties of nn.Module
        super(BattleshipModel, self).__init__()

        # Defines all layers
        self.input_layer_1 = (Conv(1, 8, "relu"))
        self.hidden_layer_2 = (Conv(8, 16, "relu"))
        self.hidden_layer_3 = (Conv(16, 32, "relu"))
        self.hidden_layer_4 = (Conv(32, 16, "relu"))
        self.hidden_layer_5 = (Conv(16, 8, "relu"))
        self.output_layer_6 = (Conv(8, 1, "tanh"))

    def forward(self, x):
        x1 = self.input_layer_1(x)
        x2 = self.hidden_layer_2(x1)
        x3 = self.hidden_layer_3(x2)
        x4 = self.hidden_layer_4(x3)
        x5 = self.hidden_layer_5(x4)
        output = self.output_layer_6(x5)
        return output
    
if __name__ == '__main__':
    # If running this as main, print a summary of each layer, and the number of parameters.
    # Run this with python3 -m model.model_definition

    # Define the input size of the model
    input_size = (1, 10, 10)

    # Print a summary of the NN model
    battleship_model = BattleshipModel()
    summary(battleship_model, input_size)

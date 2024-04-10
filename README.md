# BattleshipGame
This is a project to build the game Battleship, using an AI based on convolutional neural networks (CNN). Battleship is a classic board game in which two players attempt to sink each others ships by firing shots, without knowing the locations of each others ships. An app to play the game is constructed using Bokeh, a python dashboarding module. This app is hosted on AWS.

## About the AI
The AI for the computer is powered by convolutional neural networks (CNN). The CNN was trained by randomly creating 10x10 pixel game boards in which the ship locations have a value of 1, and empty locations has a value of -1. Masked game boards were created by randomly assigning a value 0 to 30 - 100% of the spaces, and the CNNs task is to predict the true values of the spaces, without the mask. The CNN architecture is detailed in the model/ directory, and the whole model was built and trained using PyTorch.

## About Bokeh
This module enables sophisticated interactive dashboards to be created using python. Was Bokeh the best choice for this project, among other options like React, Gradio, etc.? I am not sure. But since I know how to use Bokeh, I chose this framework. The dashboard can be run simply with "bokeh serve battleship_app --show" from the project directory.

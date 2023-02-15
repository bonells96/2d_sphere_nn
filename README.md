# Neural Network for Predicting Points Inside the Unit Circle


This repository contains code for building and training a neural network using PyTorch to predict whether a point in $\mathbb{R}^2$ is inside the unit circle or not. The model is trained using a dataset of labeled points, where each point is represented as a pair of (x, y) coordinates and labeled as 1 if it's inside the unit circle or 0 if it's outside the unit circle.


## Requirements
To run the code in this repository, you'll need the following:

PyTorch (we used version 1.9.0)
NumPy
Matplotlib (optional, for visualizing the results)
Seaborn


## Data

The repository includes a data.py file, which generates a dataset of labeled points. The *GenerateUnitSphere2dDataset* class in this file can be used to generate a new dataset with a specified number of points and a specified radius for the unit circle. The default dataset includes 1000 points with a radius of 1.
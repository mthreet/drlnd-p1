# Project 1: Navigation

## Learning Algorithm
This project uses Q-Learning to train an agent to interact with the Banana environment. Q-Learning involves using a neural network as a function approximator to estimate the Q-function (action value function). For this project the neural network is comprised of 6 dense/linear layers and 5 activation layers with the following sizes:

* `Dense`: 37x64
* `ReLU`: 64
* `Dense`: 64x128
* `ReLU`: 128
* `Dense`: 128x256
* `ReLU`: 256
* `Dense`: 256x128
* `ReLU`: 128
* `Dense`: 128x64
* `ReLU`: 64
* `Dense`: 64x4

All dense layers have biases. The final dense layer does not have a ReLU activation so that the action-value estimates can take on both positive and negative values.


# Plot of Rewards
The following hyperparameters were used to train the agent that obtained the following plot of scores:

* Alpha
* Batch size
* etc.

Picture of plot

## Ideas for Future Work
Different activations (Leaky ReLU, sigmoid, tanh)
Different model architecture (see how small the model can be)
Try altered DQN algorithm (Dualing DQN, Rainbow DQN)
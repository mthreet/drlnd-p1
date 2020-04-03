# Project 1: Navigation

## Introduction
This project involved training a banana-collecting agent with a provided Unity Machine Learning Agent.

A reward of +1 is given for collecting a yellow banana, and a reward of -1 is given for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The environment is considered solved when the agent obtains an average reward of +13 over 100 episodes. The state-space has 37 features, including the agent's velocity and ray-based perceptions of objects. The action space is 4-dimensional, with values:

* `0` Move forward
* `1` Move backward
* `2` Turn left
* `3` Turn right

This problem was solved by using a neural network as a function approximator for the action-value function. This is called Q-Learning, and for more information check out [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) about Deep Q-Learning. The network is comprised of 6 dense/linear layers, takes the 37-dimensional state space as input, and outputs the estimated 4-dimensional action-value vector for that input state.

## Installation
Follow the instuctions at [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies) (specifically the "Dependencies" section) for information on installing the environment. **Step 4 about the iPython kernel can be ignored**. This will require Anaconda for Python 3.6 or higher.

After the environment is ready, clone this repository:
```
git clone https://github.com/mthreet/drlnd-p1
```

## Running the code
To run a pretrained model (that received an average score of +13.0 over 100 episodes), simply run [eval.py](eval.py):
```
python eval.py
```

To train a model with [train.py](train.py), simply run:
```
python train.py
```
**Note that this will overwrite the checkpoint unless the save name is changed on line 58 of [train.py](train.py). Line 21 of [eval.py](eval.py) must also then be changed to the new corresponding name.**
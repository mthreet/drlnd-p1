from unityagents import UnityEnvironment
import torch
from time import sleep

from agent import Agent
from model import DenseNet

# Load the environment
env = UnityEnvironment(file_name='Banana_Linux/Banana.x86_64')
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Get basic env parmaeters
env_info = env.reset(train_mode=False)[brain_name]
state_size = len(env_info.vector_observations[0])
action_size = brain.vector_action_space_size

# Load the Agent
net = DenseNet(state_size, action_size)
net.load_state_dict(torch.load('checkpoint.pth'))
agent = Agent(state_size, action_size, net)

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state)                      # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    # Pause so that simulation can be watched in real-time
    sleep(0.1)
    
print("Score: {}".format(score))

env.close()
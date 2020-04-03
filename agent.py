import numpy as np
import random
import torch
from collections import namedtuple, deque
from copy import deepcopy

from model import DenseNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    '''
    Base class for an agent that interacts with an environment and learns.
    '''

    def __init__(self, state_size: int, action_size: int,
                 network: torch.nn.Module, buffer_size: int=100000,
                 batch_size: int=64, gamma: float=0.99, tau: float=1e-3,
                 lr: float=1e-3, update_every: int=4):
        '''
        Initialize the Agent.

        Params:
            state_size: The number of environment states
            action_size: The number of agent actions
            network: The network to train the agent with. Uninitialized class
                     that constructs a torch.nn.Module
            buffer_size: The number of State, Action, Reward, Next State tuples
                        to save for replay
            gamma: The discount factor
            tau: The soft-update parameter
            lr: The learning rate
            update_every: Number of steps between network updates
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.network_local = network.to(device)
        self.network_target = deepcopy(network).to(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        self.optimizer = torch.optim.RMSprop(self.network_local.parameters(),
                                             lr=lr)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        self.optimizer.zero_grad()
        q_targets = torch.max(self.network_target(next_states),
                              dim=1, keepdim=True)[0]
        q_targets = self.gamma * q_targets
        q_targets = q_targets * (1-dones)
        q_targets = rewards + q_targets
        q_expected = self.network_local(states).gather(1, actions)
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        '''Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

import torch


class DenseNet(torch.nn.Module):
    '''Simple fully-connected network with variable number of layers'''

    def __init__(self, state_size: int, action_size: int):
        '''
        Initialize the network. The input layer will accept an input of
        shape (batch_size, state_size). The output layer will have a shape
        of (batch_size, action_size). The network will have 5 total dense
        layers.

        Params:
            state_size: The number of environment states
            action_size: The number of agent actions
        '''
        super(DenseNet, self).__init__()
        self.dense1 = torch.nn.Linear(state_size, 64)
        self.act1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(64, 128)
        self.act2 = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(128, 256)
        self.act3 = torch.nn.ReLU()
        self.dense4 = torch.nn.Linear(256, 128)
        self.act4 = torch.nn.ReLU()
        self.dense5 = torch.nn.Linear(128, 64)
        self.act5 = torch.nn.ReLU()
        self.dense6 = torch.nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Run a forward pass through the network

        Params:
            state: The torch tensor specifying the environment state. Should
                   be of shape (batch_size, state_size)

        Returns:
            q_vals: A torch tensor specifying the estimated action values for
                    the input state
        ''' 
        q_vals = self.dense1(state)
        q_vals = self.act1(q_vals)
        q_vals = self.dense2(q_vals)
        q_vals = self.act2(q_vals)
        q_vals = self.dense3(q_vals)
        q_vals = self.act3(q_vals)
        q_vals = self.dense4(q_vals)
        q_vals = self.act4(q_vals)
        q_vals = self.dense5(q_vals)
        q_vals = self.act5(q_vals)
        q_vals = self.dense6(q_vals)
        return q_vals
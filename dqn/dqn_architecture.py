# DQN Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
  def __init__(self, env):
    super(QNetwork, self).__init__()
    #--------- YOUR CODE HERE --------------
    self.fc1 = nn.Linear(8, 64)     # First fully connected layer
    self.fc2 = nn.Linear(64, 64)    # Second fully connected layer
    self.fc3 = nn.Linear(64, 5)     # Output layer
    #---------------------------------------

  def forward(self, x, device):
    #--------- YOUR CODE HERE --------------
    x = torch.tensor(x, dtype=torch.float32).to(device)   # Convert x to tensor and move to the appropriate device
    x = F.relu(self.fc1(x))                               # Activation function for the first hidden layer
    x = F.relu(self.fc2(x))                               # Activation function for the second hidden layer
    x = self.fc3(x)                                       # No activation function for the output layer (raw scores for Q-values)
    return x
    #---------------------------------------

  def select_discrete_action(self, obs, device):
    # Put the observation through the network to estimate q values for all possible discrete actions
    est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
    # Choose the discrete action with the highest estimated q value
    discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
    return discrete_action

  def action_discrete_to_continuous(self, discrete_action):
    #--------- YOUR CODE HERE --------------
    action_mapping = {
        0: [-1.0, -1.0],
        1: [-1.0, 1.0],
        2: [0.0, 0.0],
        3: [1.0, -1.0],
        4: [1.0, 1.0]
    }
    return np.array(action_mapping[discrete_action]).reshape((2,1))
    #---------------------------------------

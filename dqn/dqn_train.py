# DQN Train

import time
from render import Renderer
from arm_env import ArmEnv
import numpy as np
from math import dist
import torch.optim as optim
import torch.nn.functional as F
import random
import os

class TrainDQN:

  def __init__(self, env, seed=48):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(56)
    self.env = env
    self.device = torch.device('cpu')
    self.q_network = QNetwork(env).to(self.device)
    self.target_network = QNetwork(env).to(self.device)
    self.target_network.load_state_dict(self.q_network.state_dict())

  def save_model(self, episode_num, save_dir='models'):
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(save_dir, timestr)
    if not os.path.exists(os.path.join(model_dir)):
      os.makedirs(os.path.join(model_dir))
    savepath = os.path.join(model_dir, f'q_network_ep_{episode_num:04d}.pth')
    torch.save(self.q_network.state_dict(), savepath)
    print(f'model saved to {savepath}\n')
    return savepath


  def train(self):

    #--------- YOUR CODE HERE --------------
    num_episodes = 1900
    batch_size = 64
    gamma = 0.99
    target_update = 10
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
    self.replay_buffer = ReplayBuffer(10000)

    for episode in range(num_episodes):
      state = self.env.reset()
      state = torch.from_numpy(state).float().to(self.device)
      total_loss = 0
      step_count = 0

      while True:
        # Get q values for current state
        q_values = self.q_network(state.unsqueeze(0), self.device)

        # Get the action index with the highest Q-value
        action_index = q_values.max(1)[1].item()

        # Convert discrete action to continuous using the mapping in QNetwork
        action_formatted = self.q_network.action_discrete_to_continuous(action_index)

        # Step through the environment using the continuous action
        next_state, reward, done, _ = self.env.step(action_formatted)
        next_state = torch.from_numpy(next_state).float().to(self.device)

        # Store the transition in the replay buffer
        self.replay_buffer.put((state.numpy(), action_index, reward, next_state.numpy(), done))

        if len(self.replay_buffer.buffer) > batch_size:
          # Sample a batch from the replay buffer
          states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
          states = torch.from_numpy(np.array(states)).float().to(self.device)
          actions = torch.from_numpy(np.array(actions)).long().to(self.device)
          rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
          next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
          dones = torch.from_numpy(np.array(dones)).float().to(self.device)

          # Perform a learning step
          current_q_values = self.q_network(states, self.device).gather(1, actions.unsqueeze(1)).squeeze(1)
          next_q_values = self.target_network(next_states, self.device).max(1)[0]
          expected_q_values = rewards + gamma * next_q_values * (1 - dones)

          loss = F.mse_loss(current_q_values, expected_q_values)
          self.optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.2)
          self.optimizer.step()
          total_loss += loss.item()
          step_count += 1

        if done:
          break

        # Update state to the next state
        state = next_state

      # Periodically update the target network
      if episode % target_update == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

      # Log and save model every 100 episodes
      if episode % 100 == 0:
        model_path = self.save_model(episode)
        print(f"Episode {episode}, Total Loss: {total_loss / step_count if step_count > 0 else 0}, Steps: {step_count}")
    return model_path

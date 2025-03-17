import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
device='cuda'
import vessl

from InNOutSpace import *
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
        action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs.squeeze(0)[action]

    def update(self, rewards, log_probs, values, dones):
        Qvals = []
        Qval = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            Qval = reward + self.gamma * Qval * (1 - done)
            Qvals.insert(0, Qval)

        Qvals = torch.tensor(Qvals).to(device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values.squeeze()
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = F.mse_loss(values.squeeze(), Qvals)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 환경 및 A2C 설정
width, height = 5, 5
num_block = (1,20)
env = Space(width=width, height=height, num_block=num_block,goal=0,block_indices=[], target=0, allocation_mode=False)
action_dim = env.action_space.n
state_dim = width * height
agent = A2CAgent(state_dim, action_dim)

num_episodes = 50000
timestep_limit = 300
history=np.zeros(num_episodes)
for episode in range(num_episodes):
    state = env.get_state()
    log_probs, values, rewards, dones = [], [], [], []
    total_reward = 0

    for t in range(timestep_limit):
        action, log_prob = agent.select_action(state)
        next_state, reward, done = env.step(action)

        _, value = agent.model(torch.FloatTensor(state.flatten()).unsqueeze(0))

        log_probs.append(torch.log(log_prob))
        values.append(value.squeeze())
        rewards.append(reward)
        dones.append(done)
        total_reward += reward

        state = next_state

        if done:
            break

    agent.update(rewards, log_probs, values, dones)
    vessl.log(step=episode, payload={'reward': total_reward})

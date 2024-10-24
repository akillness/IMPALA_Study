import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class OffPolicyRL:
    def __init__(self, state_dim, action_dim, gamma=0.99):
        self.gamma = gamma
        self.behaviour_policy_net = PolicyNetwork(state_dim, action_dim)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.target_policy_net.parameters(), lr=0.001)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=0.001)

    def select_action(self, state, policy_net):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def compute_value(self, trajectory):
        states, actions, rewards = zip(*trajectory)
        states_tensor = torch.FloatTensor(states)
        rewards_tensor = torch.FloatTensor(rewards)
        values = self.value_net(states_tensor).squeeze()
        return values, rewards_tensor

    def update(self, trajectory):
        values, rewards = self.compute_value(trajectory)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Update value network
        value_loss = nn.MSELoss()(values, returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Update target policy network
        states, actions, _ = zip(*trajectory)
        states_tensor = torch.FloatTensor(states)
        action_probs = self.target_policy_net(states_tensor)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        selected_action_probs = action_probs.gather(1, actions_tensor).squeeze()
        advantages = returns - values.detach()
        policy_loss = -torch.mean(torch.log(selected_action_probs) * advantages)
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

# Example usage
state_dim = 4
action_dim = 2
gamma = 0.99

env = None  # Replace with your environment
agent = OffPolicyRL(state_dim, action_dim, gamma)

# Generate a trajectory using behavior policy μ
trajectory = []
state = env.reset()
for _ in range(100):  # Example trajectory length
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    trajectory.append((state, action, reward))
    state = next_state
    if done:
        break

# Update the agent using the trajectory
agent.update(trajectory)

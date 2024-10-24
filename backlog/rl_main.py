import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[:, action].item()

# Define the neural network for the value function
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent
class IWALAAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.01, beta=0.01, c_bar=1.0, lambda_=0.9):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=beta)
        self.gamma = gamma
        self.c_bar = c_bar
        self.lambda_ = lambda_

    def compute_v_trace_target(self, trajectory):
        n = len(trajectory)
        v_trace_target = self.value_net(torch.FloatTensor(trajectory[0][0])).item()
        cumulative_sum = 0

        for t in range(n):
            state, action, reward = trajectory[t]
            next_state = trajectory[t + 1][0] if t + 1 < n else state
            delta_t_V = reward + self.gamma * self.value_net(torch.FloatTensor(next_state)).item() - self.value_net(torch.FloatTensor(state)).item()
            rho_t = 1  # Assuming ρt = 1 for simplicity
            ci_product = np.prod([min(1, self.c_bar) * self.lambda_ for _ in range(t)])
            cumulative_sum += (self.gamma ** t) * ci_product * delta_t_V

        v_trace_target += cumulative_sum
        return v_trace_target

    def update_value_function(self, trajectories):
        for trajectory in trajectories:
            for t, (state, action, reward) in enumerate(trajectory):
                v_trace_target = self.compute_v_trace_target(trajectory[t:])
                v_pred = self.value_net(torch.FloatTensor(state))
                loss = (v_pred - v_trace_target) ** 2
                self.value_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()

    def update_policy(self, trajectories):
        for trajectory in trajectories:
            for t, (state, action, reward) in enumerate(trajectory):
                next_state = trajectory[t + 1][0] if t + 1 < len(trajectory) else state
                q_s = reward + self.gamma * self.value_net(torch.FloatTensor(next_state)).item() - self.value_net(torch.FloatTensor(state)).item()
                rho = 1  # Assuming ρt = 1 for simplicity
                state_tensor = torch.FloatTensor(state)
                action_tensor = torch.tensor(action)
                log_prob = torch.log(self.policy_net(state_tensor)[action_tensor])
                loss = -rho * log_prob * q_s
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

# Training the agent
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = IWALAAgent(state_dim, action_dim)

num_episodes = 1000
for episode in range(num_episodes):
    state, info = env.reset()
    trajectory = []
    done = False

    while not done:
        action, _ = agent.policy_net.get_action(state)
        next_state, reward, done, _, info = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state

        if done:
            agent.update_value_function([trajectory])
            agent.update_policy([trajectory])
            break

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

env.close()

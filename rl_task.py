import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gym

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy_dist = Categorical(logits=self.policy(x))
        value = self.value(x)
        return policy_dist, value

# Actor class to generate trajectories
class Actor:
    def __init__(self, env, policy, n_steps, device):
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.device = device

    def generate_trajectory(self):
        state = self.env.reset()[0]
        trajectory = []
        for _ in range(self.n_steps):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            policy_dist, _ = self.policy(state)
            action = policy_dist.sample()
            next_state, reward, done, _, _ = self.env.step(action.item())
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                break
        return trajectory

# Learner class to update the policy
class Learner:
    def __init__(self, policy, device, lr=1e-3):
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.writer = SummaryWriter()

    def update_policy(self, trajectories, global_step):
        policy_loss_total = 0
        value_loss_total = 0
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            
            policy_dists, values = self.policy(states)
            log_probs = policy_dists.log_prob(actions)
            advantages = rewards - values.squeeze()
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()

        # Log losses to TensorBoard
        self.writer.add_scalar('Loss/Policy', policy_loss_total / len(trajectories), global_step)
        self.writer.add_scalar('Loss/Value', value_loss_total / len(trajectories), global_step)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = ActorCritic(input_dim, action_dim).to(device)
    actor = Actor(env, policy, n_steps=5, device=device)
    learner = Learner(policy, device)
    
    # env.reset()
    global_step = 0
    for episode in tqdm(range(10001)):
        # env.render()
        trajectory = actor.generate_trajectory()
        learner.update_policy([trajectory], global_step)
        global_step += 1

        # Save the model every 100 episodes
        if episode % 10000 == 0:
            learner.save_model(f"model_{episode}.pth")
    
    # env.close()

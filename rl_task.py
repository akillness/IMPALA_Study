import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyperparameters
gamma = 0.99
learning_rate = 0.0005
num_episodes = 1000
batch_size = 32

# Environment
env = gym.make('CartPole-v1',render_mode="rgb_array")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_logits = nn.Linear(128, action_size)
        self.values = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.policy_logits(x), self.values(x)

# IMPALA Agent
class IMPALAAgent:
    def __init__(self, action_size):
        self.model = ActorCritic(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_action(self, state):
        logits, _ = self.model(state)
        action_prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_prob, 1).item()
        return action

    def compute_loss(self, states, actions, rewards, next_states, dones):
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        target_values = rewards + gamma * next_values * (1 - dones)
        advantages = target_values - values

        policy_logits, _ = self.model(states)
        action_log_probs = torch.log_softmax(policy_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        policy_loss = -(selected_log_probs * advantages.detach()).mean()

        value_loss = advantages.pow(2).mean()
        return policy_loss + value_loss

    def train(self, states, actions, rewards, next_states, dones):
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()
        self.optimizer.step()
        return loss.item()



def main():
    # TensorBoard writer
    writer = SummaryWriter()

    # Training
    agent = IMPALAAgent(env.action_space.n)
    for episode in range(num_episodes):
        state = torch.FloatTensor(env.reset()[0]).unsqueeze(0).to(device)  # 상태를 텐서로 변환하고 배치 차원을 추가
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)  # 상태를 텐서로 변환하고 배치 차원을 추가
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.FloatTensor([done]).to(device)
            action = torch.LongTensor([action]).to(device)  # 인덱스를 int64 타입으로 변환

            loss = agent.train(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward.item()

        # TensorBoard logging
        writer.add_scalar('Loss/train', loss, episode)
        writer.add_scalar('Reward/train', episode_reward, episode)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # 모델 저장
        if (episode + 1) % 100 == 0:  # 100 에피소드마다 모델 저장
            torch.save(agent.model.state_dict(), f"model_episode_{episode + 1}.pth")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
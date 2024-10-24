import sys
import gym
import pylab
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 1000

# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.actor = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.model = ActorCritic(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)

        if self.load_model:
            self.model.load_state_dict(torch.load("./save_model/cartpole_actor_critic_trained.pth"))

    def get_action(self, state):
        state = torch.FloatTensor(state)
        policy, _ = self.model(state)
        policy = policy.detach().numpy()
        return np.random.choice(self.action_size, 1, p=policy.flatten())[0]

    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done, dtype=torch.float32)

        _, value = self.model(state)
        _, next_value = self.model(next_state)

        target = reward + (1 - done) * self.discount_factor * next_value
        advantage = target - value

        policy, value = self.model(state)
        action_prob = policy.gather(1, action.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        actor_loss = -torch.log(action_prob) * advantage.detach()
        critic_loss = advantage.pow(2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model.state_dict(), "./save_model/cartpole_actor_critic.pth")
                    sys.exit()


import torch
import torch.nn as nn
import torch.optim as optim

'''
가중치 ci는 Retrace에서의 "trace cutting" 계수와 유사합니다. 
이들의 곱 cs . . . ct−1는 시간 t에서 관찰된 시간차 δtV가 이전 시간 s에서의 가치 함수 업데이트에 얼마나 영향을 미치는지를 측정합니다.
π와 µ가 더 다를수록(즉, off-policy일수록) 이 곱의 분산이 커집니다. 
우리는 분산 감소 기법으로 절단 수준 c¯를 사용합니다. 
그러나 이 절단은 우리가 수렴하는 해(ρ¯로 특징지어짐)에 영향을 미치지 않습니다.
따라서 절단 수준 c¯와 ρ¯는 알고리즘의 다른 특성을 나타냅니다:
 ρ¯는 우리가 수렴하는 가치 함수의 성격에 영향을 미치고, c¯는 우리가 이 함수에 수렴하는 속도에 영향을 미칩니다.
'''
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = self.build_actor(state_dim, action_dim)
        self.critic = self.build_critic(state_dim)

    def build_actor(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def build_critic(self, state_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class Actor:
    def __init__(self, env, actor_critic, n_steps):
        self.env = env
        self.actor_critic = actor_critic
        self.n_steps = n_steps
        self.local_policy = actor_critic

    def update_local_policy(self):
        self.local_policy = self.actor_critic

    def generate_trajectory(self):
        state,info = self.env.reset()
        trajectory = []
        for _ in range(self.n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.local_policy(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            # lstm 의 init을 이용해 hidden state 반영한 action 및 logits 포함

            next_state, reward, done, truncated, info = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                break
        return trajectory
    
    
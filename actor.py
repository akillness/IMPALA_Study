
import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np

# from impala import IMPALA
import impala

from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, env, n_steps, unroll):
        self.env = env
        self.n_steps = n_steps
        self.unroll = unroll
        self.name = "actor"
        # if self.name == 'thread_0':
        #     self.env = gym.wrappers.Monitor(self.env, 'save-mov', video_callable=lambda episode_id: episode_id%10==0)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.actor_critic = ActorCritic(state_dim,action_dim)
        
        self.global_policy =impala.IMPALA(
                                   state_shape=env.observation_space.shape,
                                   output_size=action_dim,
                                   activation=nn.ReLU(),
                                   final_activation=nn.Softmax(),
                                   discount_factor=0.99,
                                   lr=0.001,
                                   hidden=256,
                                   entropy_coef=0.01,
                                   reward_clip='tanh',#reward_clip=['tanh','abs_one','no_clip'],
                                   unroll=unroll # figure 2
                                   )
        
        self.local_policy = impala.IMPALA(
                                   state_shape=env.observation_space.shape,
                                   output_size=action_dim,
                                   activation=nn.ReLU(),
                                   final_activation=nn.Softmax(),
                                   discount_factor=0.99,
                                   lr=0.001,
                                   hidden=256,
                                   entropy_coef=0.01,
                                   reward_clip='tanh',
                                   unroll=unroll # figure 2
                                   )

    # def update_local_policy(self):
    #     self.local_policy = self.actor_critic

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
    
    def run(self):
        # self.env = self.gym.make('PongDeterministic-v4')        
        
        done = False
        obs, info = self.env.reset()
        history = np.stack((obs, obs, obs, obs), axis=2)
        state = copy.deepcopy(history)
        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        loss_step = 0

        writer = SummaryWriter('runs/' + self.name)

        while True:
            loss_step += 1
            episode_state = []
            episode_next_state = []
            episode_reward = []
            episode_done = []
            episode_action = []
            episode_behavior_policy = []
            for i in range(128):
                
                action, behavior_policy, max_prob = self.local_policy.policy_and_action(state)

                episode_step += 1
                total_max_prob += max_prob
            
                obs, reward, done, truncated, info = self.env.step(action + 1)
                # obs = utils.pipeline(obs)
                history[:, :, :-1] = history[:, :, 1:]
                history[:, :, -1] = obs
                next_state = copy.deepcopy(history)

                score += reward

                d = False
                if reward == 1 or reward == -1:
                    d = True

                episode_state.append(state)
                episode_next_state.append(next_state)
                episode_reward.append(reward)
                episode_done.append(d)
                episode_action.append(action)
                episode_behavior_policy.append(behavior_policy)

                state = next_state

                if done:
                    print(self.name, episode, score, total_max_prob / episode_step, episode_step)
                    writer.add_scalar('score', score, episode)
                    writer.add_scalar('max_prob', total_max_prob / episode_step, episode)
                    writer.add_scalar('episode_step', episode_step, episode)
                    episode_step = 0
                    total_max_prob = 0
                    episode += 1
                    score = 0
                    done = False
                    # if self.name == 'thread_0':
                    #     self.env.close()
                    obs,info = self.env.reset()
                    # obs = utils.pipeline(obs)
                    history = np.stack((obs, obs, obs, obs), axis=2)
                    state = copy.deepcopy(history)

            pi_loss, value_loss, entropy = self.global_network.train(
                state=np.stack(episode_state),
                next_state=np.stack(episode_next_state),
                reward=np.stack(episode_reward),
                done=np.stack(episode_done),
                action=np.stack(episode_action),
                behavior_policy=np.stack(episode_behavior_policy))
            
            # self.global_to_local()
            writer.add_scalar('pi_loss', pi_loss, loss_step)
            writer.add_scalar('value_loss', value_loss, loss_step)
            writer.add_scalar('entropy', entropy, loss_step)
        self.env.close()
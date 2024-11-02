
import gym
import numpy as np
import math
import torch

from collections import deque

import torch.multiprocessing as mp


def get_action_size(env_class, env_args):
    env = env_class(**env_args)
    action_size = env.action_size()
    env.close()
    del env
    return action_size

# RL 과제를 위한 CartPole 게임환경
class CartPole:
    def __init__(self, game_name, seed,max_episode_length=1e10, history_length=4, device='cpu'):
        self.device = device
        self.env = gym.make(game_name)
        self.env.reset(seed=seed)
        np.random.seed(seed)
        self.env._max_episode_steps = max_episode_length
        
        self.actions = self.env.action_space 
        self.history_length = history_length
        self.state_buffer = deque([], maxlen=history_length)
        self.training = True

    def _get_state(self):
        state, info = self.env.reset()
        return torch.tensor(state,dtype=torch.float32, device=self.device), info

    def _reset_buffer(self):
        for _ in range(self.history_length):
            self.state_buffer.append(torch.zeros(4, device=self.device))

    def reset(self):        
        self._reset_buffer()
        observation, info = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # # Repeat action 30 times, max pool over last 2 frames
        # frame_buffer = torch.zeros(2, 4, device=self.device)
        # reward, done = 0, False
        # for t in range(30):
        #     state, reward, done, _, info = self.env.step(action)
        #     if t == 2:
        #         frame_buffer[0]= torch.tensor(state,dtype=torch.float32, device=self.device)
        #     elif t == 3:
        #         frame_buffer[1]= torch.tensor(state,dtype=torch.float32, device=self.device)
        #     if done:
        #         break
        # observation = frame_buffer.max(0)[0]
        # self.state_buffer.append(observation)
        
        state, reward, done, _, info = self.env.step(action)
        observation = torch.tensor(state,dtype=torch.float32, device=self.device)
        self.state_buffer.append(observation)
        
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def action_size(self):
        return self.actions.n

    def close(self):
        self.env.close()



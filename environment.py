
import gym
import numpy as np
import math
import torch

import atari_py
import cv2
import random 

from collections import deque

import torch.multiprocessing as mp


# # 호환성을 위한 bool8 정의
# if not hasattr(np, "bool8"):
#     np.bool8 = np.bool_

def get_action_size(env_class, env_args):
    env = env_class(**env_args)
    action_size = env.action_size()
    env.close()
    del env
    return action_size

class Atari:
    def __init__(self, game_name, seed, max_episode_length=1e10, history_length=4, device='cpu'):
        self.device = device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setInt('max_num_frames_per_episode', max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        # self.ale.setBool('display_screen', True)  # 기본 렌더링 비활성화
        self.ale.setInt('frame_skip', 4)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(game_name))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict(zip(range(len(actions)), actions))
        # self.reward_clip = reward_clip
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=history_length)
        self.training = True  # Consistent with model training mode
        self.viewer = None

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # # # Perform up to 30 random no-ops before starting
            # for _ in range(random.randrange(30)):
            #     self.ale.act(0)  # Assumes raw action 0 is always no-op
            #     if self.ale.game_over():
            #         self.ale.reset_game()

        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def render_screen(self,screen):
        """
        렌더링된 게임 화면을 OpenCV 창에 표시.

        Args:
            screen (np.ndarray): 그레이스케일 화면 데이터.
        """
        # 화면 크기 확인
        print(f"Screen shape: {screen.shape}")  # 예: (210, 160)

        # OpenCV 창에 화면 출력
        # 화면 표시
        # screen = self.ale.getScreenRGB()
        cv2.imshow('Atari Screen', screen)

        # OpenCV 창 종료 조건 (ESC 키)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            cv2.destroyAllWindows()
            exit()
        
    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)

        reward, done = 0, False
        for t in range(self.window):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break

        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if self.lives > lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives

        # Return state, reward, done
        # reward = max(min(reward, self.reward_clip), -self.reward_clip)
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_size(self):
        return len(self.actions)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# RL 과제를 위한 CartPole 게임환경
class CartPole:
    def __init__(self, game_name, seed,max_episode_length=1e10, history_length=4, device='cpu'):
        self.device = device
        self.env = gym.make(game_name)
        self.env.reset(seed=seed)
        np.random.seed(seed)
        self.env._max_episode_steps = max_episode_length
        
        self.actions = self.env.action_space.n 
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
        state, reward, done, _, info = self.env.step(action)
        observation = torch.tensor(state,dtype=torch.float32, device=self.device)
        self.state_buffer.append(observation)
        # print(f"state : {action}, reward : {reward}")        
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def action_size(self):
        return self.actions

    def close(self):
        self.env.close()
        # exit()



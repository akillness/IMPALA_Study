import argparse

import torch 
import torch.nn as nn
import gym

# from model import IMPALA
from model import Network
from actor import actor
from learner import learner

import numpy as np

from environment import CartPole

# import threading, queue
from concurrent.futures import ThreadPoolExecutor

# 호환성 설정
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


# Device 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


# CartPole 환경 생성
env = CartPole(game_name='CartPole-v1', seed=123, render=True)

# 저장된 모델 로드
model_path = "./checkpoint.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

# state_dim = env.observation_space.shape[0]
action_dim = env.action_size()


model = Network(action_size=action_dim)
# init_core_state = model.inital_state()
model.load_state_dict(checkpoint)
model.eval()

# 환경 실행 및 렌더링
for episode in range(5):  # 1개의 에피소드 실행
    state = env.reset()
    done = False
    total_reward = 0

    # core_state = init_core_state
    # logits = torch.zeros((1, action_dim), dtype=torch.float32)
    # last_action = torch.zeros((1, 1), dtype=torch.int64)
    # reward = torch.tensor(0, dtype=torch.float32).view(1, 1)
    # done = torch.tensor(True, dtype=torch.bool).view(1, 1)
    # init_state = (state, last_action, reward, done, logits)
    # persistent_state = init_state

    while True:
        env.render()  # 환경 시각화

        # state= torch.FloatTensor(state).unsqueeze(0).to(device)
        # action, logits = model.get_policy(state)
        # action, logits, selected_core_states, core_state = model.get_policy_and_action(state.unsqueeze(0), last_action, reward, done, core_state)
        action, logits = model.get_policy_and_action(state)
        # action = torch.argmax(logits).item()

        state, r, d = env.step(action)
        # reward = torch.tensor(r, dtype=torch.float32).view(1, 1)
        # done = torch.tensor(d, dtype=torch.bool).view(1, 1)
        total_reward += r
        
        if d:
            break


    print(f"에피소드 {episode + 1} 완료, 총 보상: {total_reward}")

env.close()
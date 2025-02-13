import argparse

import torch 
import torch.nn as nn
import gym

# from model import IMPALA
from impala import IMPALA
from actor import actor
from learner import learner

import numpy as np

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
env = gym.make('CartPole-v1', render_mode="human")

# 저장된 모델 로드
model_path = "./model/impala.pt"
checkpoint = torch.load(model_path, map_location=device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = IMPALA(action_size=2)
model.load_state_dict(checkpoint)
model.eval()

core_state = None
# 환경 실행 및 렌더링
for episode in range(5):  # 1개의 에피소드 실행
    state,_ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # 환경 시각화

        state= torch.FloatTensor(state).unsqueeze(0).to(device)
        # policy_dist, _ = model(state)
        action, logits = model.get_policy_and_action(state)
        action = torch.argmax(logits, dim=-1).item()  # 가장 높은 확률의 행동 선택

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"에피소드 {episode + 1} 완료, 총 보상: {total_reward}")

env.close()
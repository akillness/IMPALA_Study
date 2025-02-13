import random
from collections import deque
import torch

class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.memory = deque(maxlen=capacity)  # 메모리 크기 고정
        self.device = device  # GPU/CPU 설정

    def push(self, state, action, next_state, reward, done):
        """경험 저장"""
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)
        
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        """배치 샘플링"""
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in memory.")
        
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # 텐서로 변환 및 차원 조정
        states = torch.stack(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)
        
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.memory)
    

# CAPACITY = 10000
# BATCH_SIZE = 64

# # Replay Memory 초기화
# memory = ReplayMemory(CAPACITY, device='cuda' if torch.cuda.is_available() else 'cpu')

# # 예시: 경험 저장
# state = [0.1, 0.2, 0.3]
# action = 2
# next_state = [0.4, 0.5, 0.6]
# reward = 1.0
# done = False
# memory.push(state, action, next_state, reward, done)

# # 배치 샘플링
# states, actions, next_states, rewards, dones = memory.sample(BATCH_SIZE)
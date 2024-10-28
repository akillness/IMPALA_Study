import gym
import numpy as np

# 환경 생성
env = gym.make('CartPole-v1')

# 랜덤 시드 설정
seed = 42
env.reset(seed=seed)
np.random.seed(seed)

# 환경 초기화
env.reset()

# 예제 실행
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # 랜덤 액션 선택
    env.step(action)  # 환경에 액션 적용

env.close()

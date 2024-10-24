import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gym

# Actor-Critic 모델 정의
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, output_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy_dist = Categorical(logits=self.policy(x))
        value = self.value(x)
        return policy_dist, value

# Worker 프로세스 정의
def worker(worker_id, global_model, optimizer, env_name, gamma):
    env = gym.make(env_name)
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())
    state = env.reset()
    done = False
    while not done:
        policy_dist, value = local_model(torch.tensor(state, dtype=torch.float32))
        action = policy_dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        _, next_value = local_model(torch.tensor(next_state, dtype=torch.float32))
        advantage = reward + (1 - done) * gamma * next_value.item() - value.item()
        loss = -policy_dist.log_prob(action) * advantage + 0.5 * advantage ** 2
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())
        state = next_state

# 메인 함수 정의
def main():
    env_name = 'CartPole-v1'
    global_model = ActorCritic(gym.make(env_name).observation_space.shape[0], gym.make(env_name).action_space.n)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)
    processes = []
    # num_workers = mp.cpu_count()
    num_workers = 1
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, global_model, optimizer, env_name, 0.99))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

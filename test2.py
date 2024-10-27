import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, action_size=2, input_size=4, hidden_size=512):
        super(Network, self).__init__()
        self.action_space = action_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 256)
        self.head = Head(action_size)

    def forward(self, x, last_action, reward, dones, hx=None, actor=False):
        seq_len, bs, x, last_action, reward = combine_time_batch(x, last_action, reward, actor)
        last_action = torch.zeros(last_action.shape[0], self.action_space,
                                  dtype=torch.float32, device=x.device).scatter_(1, last_action, 1)
        x = F.leaky_relu(self.fc1(x), inplace=True)
        x = torch.cat((x, reward, last_action), dim=1)
        x = x.view(seq_len, bs, -1)
        lstm_out = []
        hx = hx.to(x.device)
        init_core_state = torch.zeros((2, bs, 256), dtype=torch.float32, device=x.device)
        for state, d in zip(torch.unbind(x, 0), torch.unbind(dones, 0)):
            hx = torch.where(d.view(1, -1, 1), init_core_state, hx)
            hx = self.lstm(state, hx.unbind(0))
            lstm_out.append(hx[0])
            hx = torch.stack(hx, 0)
        x = torch.cat(lstm_out, 0)
        logits, values = self.head(x, actor)
        logits[torch.isnan(logits)] = 1e-12
        if not actor:
            return logits.view(seq_len, -1, bs), values.view(seq_len, bs)
        else:
            action = torch.softmax(logits, 1).multinomial(1).item()
            return action, logits.view(1, -1), hx

def generate_trajectory(env, network, max_steps=1000):
    state = env.reset()
    done = False
    hx = torch.zeros((2, 1, 256), dtype=torch.float32)
    trajectory = []

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        last_action = torch.zeros(1, 1, dtype=torch.int64)  # 초기 액션
        reward = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor([done], dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            action, logits, hx = network(state_tensor, last_action, reward, done_tensor, hx, actor=True)

        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, done))

        if done:
            break

        state = next_state

    return trajectory

# 사용 예시
# env = gym.make('CartPole-v1')
# network = Network()
# trajectory = generate_trajectory(env, network)

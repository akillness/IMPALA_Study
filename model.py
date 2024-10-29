import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import combine_time_batch


class IMPALA(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=512):
        super(IMPALA, self).__init__()
        self.action_space = action_size
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc = nn.Linear(3136, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 256)
        self.actorcritic = ActorCritic(action_size,hidden_size//2)

    def forward(self, input_tensor, last_action, reward, done_flags, hidden_state=None, actor=False):
        # state 의 trajectory의 길이 단위별로 history 설정 및 batch size 만큼 차원변경
        # state 의 history-4 stack
        seq_len, batch_size, input_tensor, last_action, reward = combine_time_batch(input_tensor, last_action, reward, actor)
        last_action = torch.zeros(last_action.shape[0], self.action_space,
                                  dtype=torch.float32, device=input_tensor.device).scatter_(1, last_action, 1)
        # 3-layer conv
        input_tensor = F.leaky_relu(self.conv1(input_tensor), inplace=True)
        input_tensor = F.leaky_relu(self.conv2(input_tensor), inplace=True)
        input_tensor = F.leaky_relu(self.conv3(input_tensor), inplace=True)
        input_tensor = input_tensor.view(input_tensor.shape[0], -1)
        input_tensor = F.leaky_relu(self.fc(input_tensor), inplace=True)
        input_tensor = torch.cat((input_tensor, reward, last_action), dim=1)
        input_tensor = input_tensor.view(seq_len, batch_size, -1)
        '''
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc(x), inplace=True)
        x = torch.cat((x, reward, last_action), dim=1)
        x = x.view(seq_len, bs, -1)
        '''

        # state 의 history-4 stack 에 대한 lstm feature를 actorcritic 에 state로 추가
        lstm_outputs = []
        hidden_state = hidden_state.to(input_tensor.device)
        init_core_state = torch.zeros((2, batch_size, 256), dtype=torch.float32, device=input_tensor.device)
        for state, d in zip(torch.unbind(input_tensor, 0), torch.unbind(done_flags, 0)):
            hidden_state = torch.where(d.view(1, -1, 1), init_core_state, hidden_state)
            hidden_state = self.lstm(state, hidden_state.unbind(0))
            lstm_outputs.append(hidden_state[0])
            hidden_state = torch.stack(hidden_state, 0)
        input_tensor = torch.cat(lstm_outputs, 0)
        logits, values = self.actorcritic(input_tensor, actor)

        logits[torch.isnan(logits)] = 1e-12
        if not actor:
            return logits.view(seq_len, -1, batch_size), values.view(seq_len, batch_size)
        else:
            action = torch.softmax(logits, 1).multinomial(1).item()
            return action, logits.view(1, -1), hidden_state


class ActorCritic(nn.Module):
    def __init__(self, action_space, hidden_size):
        super().__init__()
        self.actor_linear = nn.Linear(hidden_size, action_space)
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, x, actor):
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values

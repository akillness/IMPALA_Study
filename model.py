import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_stacked_state_dim(stacked_state, last_action, reward, actor=False):
    if actor:
        return 1, 1, stacked_state, last_action, reward
    stacked_state_len = stacked_state.shape[0]
    batch_size = stacked_state.shape[1]
    stacked_state = stacked_state.reshape(stacked_state_len * batch_size, *stacked_state.shape[2:])
    last_action = last_action.reshape(stacked_state_len * batch_size, -1)
    reward = reward.reshape(stacked_state_len * batch_size, 1)
    return stacked_state_len, batch_size, stacked_state, last_action, reward

def conv_block(in_channels, out_channels, device, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding).to(device)

def convnet_forward(state):
    device = state.device
    conv_out = state
    
    for i, (num_ch, num_blocks) in enumerate([(32, 2), (64, 2), (64, 2)]):
        if i == 0:
            in_ch = conv_out.shape[1]
        else:
            in_ch = prev_num_ch

        # Downscale.
        conv_out = conv_block(in_ch, num_ch, device)(conv_out)
        conv_out = conv_out.to(device)
        conv_out = F.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1)

        # Residual block(s).
        for j in range(num_blocks):
            identity = conv_out
            conv_out = F.relu(conv_out)
            conv_out = conv_block(num_ch, num_ch, device)(conv_out)
            conv_out = F.relu(conv_out)
            conv_out = conv_block(num_ch, num_ch, device)(conv_out)
            conv_out += identity
        
        prev_num_ch = num_ch
    
    return conv_out

class IMPALA(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=512):
        super(IMPALA, self).__init__()
        self.action_space = action_size

        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        
        self.fc = nn.Linear(3136, hidden_size)
        # self.fc = nn.Linear(2304, hidden_size) <-- resdual conv 사용시
        # self.fc = nn.Linear(16,hidden_size)
        
        # FC output size + one-hot of last action + last reward.
        self.core_output_size = self.fc.out_features + action_size + 1
        # self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 64)
        self.lstm = nn.LSTMCell(self.core_output_size, self.core_output_size)
        # self.core = nn.LSTM(core_output_size, core_output_size, 2)
        
        self.actor_critic = ActorCritic(self.core_output_size, action_size)
    
    def inital_state(self,batch_size=1):
        return torch.zeros((2, batch_size, self.core_output_size), dtype=torch.float32)
    
    def forward(self, input_tensor, last_action, reward, done_flags, core_state=None, actor=False):
        # state 의 trajectory의 길이 단위별로 history 설정 및 batch size 만큼 차원변경
        # state 의 history-4 stack

        seq_len, batch_size, x, last_action, reward = reshape_stacked_state_dim(input_tensor, last_action, reward, actor)
        last_action = torch.zeros(last_action.shape[0], self.action_space,
                                  dtype=torch.float32, device=input_tensor.device).scatter_(1, last_action, 1)
        
        # x = convnet_forward(x) <-- resdual conv는 고려
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        # x = F.leaky_relu(self.conv1(x), inplace=True)
        # x = F.leaky_relu(self.conv2(x), inplace=True)
        # x = F.leaky_relu(self.conv3(x), inplace=True)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc(x), inplace=True)
        # x = F.leaky_relu(self.fc(x), inplace=True)

        x = torch.cat((x, reward, last_action), dim=1)
        x = x.view(seq_len, batch_size, -1)

        # sequential to conv
        # input_tensor = convnet_forward(input_tensor)
        # input_tensor = input_tensor.view(input_tensor.shape[0], -1)
        # if not actor :
        #     input_tensor = input_tensor.T
        # input_tensor = F.relu(self.fc(input_tensor), inplace=True)
        # if not actor :
        #     input_tensor = input_tensor.expand(seq_len*batch_size,-1)
        # input_tensor = torch.cat((input_tensor, reward, last_action), dim=1)
        # input_tensor = input_tensor.view(seq_len, batch_size, -1)

        # # sequental
        # input_tensor = input_tensor.view(input_tensor.shape[0], -1)
        # input_tensor = F.relu(self.fc(input_tensor), inplace=True)

        # # input_tensor = F.relu(self.batchnorm(self.fc(input_tensor)))
        # input_tensor = torch.cat((input_tensor, reward, last_action), dim=1)
        # input_tensor = input_tensor.view(seq_len, batch_size, -1)

        # state 의 history-4 stack 에 대한 lstm feature를 actorcritic 에 state로 추가
        lstm_outputs = []
        core_state = core_state.to(x.device)
        init_core_state = self.inital_state(batch_size=batch_size).to(x.device)
        for state, d in zip(torch.unbind(x, 0), torch.unbind(done_flags, 0)):
            core_state = torch.where(d.view(1, -1, 1), init_core_state, core_state)
            core_state = self.lstm(state, core_state.unbind(0))
            lstm_outputs.append(core_state[0])
            core_state = torch.stack(core_state, 0)
        x = torch.cat(lstm_outputs, 0)

        # x = torch.cat(lstm_outputs, 0)

        logits, values = self.actor_critic(x)
        if not actor: # target to learner 
            return logits.view(seq_len, -1, batch_size), values.view(seq_len, batch_size)
        else: # target to actor
            action = torch.softmax(logits, 1).multinomial(1).item() # Case atari, Categorical
            return action, logits.view(1, -1), core_state


class ActorCritic(nn.Module):
    def __init__(self, hidden_size, action_size):
        super().__init__()
        self.actor_linear = nn.Linear(hidden_size, action_size)
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values

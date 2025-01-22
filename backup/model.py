import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_stacked_state_dim(stacked_state, last_action, reward, actor=False):
    if actor:
        return 1, 1, stacked_state, last_action, reward
    
    T = stacked_state.shape[0]
    B = stacked_state.shape[1]
    stacked_state = stacked_state.reshape(T * B, *stacked_state.shape[2:])
    last_action = last_action.reshape(T * B, -1)
    reward = reward.reshape(T * B, 1)
    return T, B, stacked_state, last_action, reward
'''
def conv_block(in_channels, out_channels, device, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding).to(device)

def convnet_forward(state):
    device = state.device
    conv_out = state
    
    for i, (num_ch, num_blocks) in enumerate([[(32, 2), (64, 2), (64, 2)]]):
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
'''

def apply_function_nested(structure, func):
    if isinstance(structure, list):
        return [apply_function_nested(item, func) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(apply_function_nested(item, func) for item in structure)
    else:
        return func(structure)
    
class IMPALA(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=256, training = False):
        super(IMPALA, self).__init__()
        self.training = training
        self.num_actions = action_size
        '''
        # hidden_size=512
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64,kernel_size=3, stride=1)

        self.fc = nn.Linear(3136, hidden_size)

        # FC output size + one-hot of last action + last reward.
        self.core_output_size = self.fc.out_features + action_size + 1
        self.hidden_size = self.core_output_size
        # self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 64)
        self.lstm = nn.LSTMCell(self.core_output_size, self.core_output_size)

        self.actor_critic = ActorCritic(self.core_output_size, action_size)
        '''
        # self.fc = nn.Linear(2304, hidden_size) <-- resdual conv 사용시
        # self.fc = nn.Linear(16,hidden_size)
        
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        input_channels = 4
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, hidden_size)

        # FC output size + last reward.
        self.core_output_size = self.fc.out_features + 1
        self.hidden_size  = hidden_size

        self.lstm = nn.LSTMCell(self.core_output_size, hidden_size)
        # self.lstm = nn.LSTM(self.core_output_size, hidden_size, num_layers=1, batch_first=True)

        self.actor_critic = ActorCritic(hidden_size, action_size)

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(1, batch_size, self.hidden_size)
            for _ in range(2)
        )
    def forward(self, input_tensor, last_action, reward, done_flags, core_state=None, actor=False):
        # state 의 trajectory의 길이 단위별로 history 설정 및 batch size 만큼 차원변경
        # state 의 history-4 stack
        # T,B, *_ = input_tensor.shape
        # x = torch.flatten(input_tensor, 0, 1)
        T, B, x, last_action, reward = reshape_stacked_state_dim(input_tensor, last_action, reward, actor)
        
        '''
        # x = convnet_forward(x) <-- resdual conv는 고려
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)

        # x = F.leaky_relu(self.conv1(x), inplace=True)
        # x = F.leaky_relu(self.conv2(x), inplace=True)
        # x = F.leaky_relu(self.conv3(x), inplace=True)
        x = x.view(x.shape[0], -1)

        '''

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(1, -1)
        x = F.relu(self.fc(x))

        # x = F.relu(self.fc(x), inplace=True)
        # x = F.leaky_relu(self.fc(x), inplace=True)
        

        clipped_rewards = torch.clamp(reward, -1, 1)
        # x = torch.cat((x, clipped_rewards, last_action), dim=1)
        x = torch.cat((x, clipped_rewards), dim=1)
        x = x.view(T, B, -1)

        # sequential to conv
        # last_action = torch.zeros(last_action.shape[0], self.num_actions,
        #                           dtype=torch.float32, device=input_tensor.device).scatter_(1, last_action, 1)
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
        init_core_state = torch.zeros((B, self.hidden_size), dtype=torch.float32, device=x.device)
        for state, d in zip(torch.unbind(x, 0), torch.unbind(done_flags, 0)):
            core_state = torch.where(d.view(1, -1, 1), init_core_state, core_state)
            core_state = self.lstm(state, core_state.unbind(0))
            lstm_outputs.append(core_state[0])
            core_state = torch.stack(core_state, 0)
        x = torch.cat(lstm_outputs, 0)
        '''
        init_core_state = self.initial_state()
        for i,(state, d) in enumerate(zip(x.unbind(), done_flags.unbind())):
            nd = d.view(1 ,-1, 1)
            # core_state = nest.map(nd.mul, init_core_state)
            core_state = [nd.mul(cs) for cs in init_core_state].pop(i)
            output, core_state = self.lstm(state.unsqueeze(0), core_state)
            lstm_outputs.append(core_state)
        x = torch.flatten(torch.cat(lstm_outputs), 0, 1)
        '''


        logits, values = self.actor_critic(x)

        '''if not actor: # target to learner 
            # 먼저 logits의 크기를 재조정합니다.
            reshaped_logits = logits.view(T, -1, B)
            
            # dim=1에 대해 argmax를 수행하려면 logits의 차원 배치가 올바른지 확인합니다.
            # 여기서는 seq_len, batch_size, -1로 차원을 변경합니다.
            logits = reshaped_logits.permute(0, 2, 1)  # (seq_len, batch_size, -1)
            predicted_indices = torch.argmax(logits, dim=2)  # dim=2에서 argmax 계산

            return reshaped_logits, values.view(T, B)
        else: # target to actor
            action = torch.softmax(logits, 1).multinomial(1).item() # Case atari, Categorical
            return action, logits.view(1, -1), core_state'''

        if not actor: # target to learner 
            return logits.view(T, -1, B), values.view(T, B)
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

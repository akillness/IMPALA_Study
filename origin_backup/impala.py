import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def reshape_stacked_state_dim(stacked_state, last_action, reward, done, actor=False):
    stacked_state_len = stacked_state.shape[0]
    batch_size = stacked_state.shape[1]
    stacked_state = stacked_state.reshape(stacked_state_len * batch_size, *stacked_state.shape[2:])
    last_action = last_action.reshape(stacked_state_len * batch_size, -1)
    reward = reward.reshape(stacked_state_len * batch_size, 1)
    done = done.reshape(stacked_state_len * batch_size, 1)
    return stacked_state_len, batch_size, stacked_state, last_action, reward, done

class IMPALA(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=128):
        super(IMPALA, self).__init__()
        self.action_space = action_size
     
        self.fc = nn.Linear(input_channels, hidden_size)

        # FC output size + one-hot of last action + last reward.
        self.core_output_size = self.fc.out_features + action_size + 1
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(hidden_size, self.hidden_size)
        
        self.head = ActorCritic(hidden_size, action_size)
    
    def inital_state(self,batch_size=1):
        return torch.zeros((2, batch_size, self.hidden_size), dtype=torch.float32)
    
    def unroll(self,x,batch_size,done_flags,core_state):
        # sequential to conv
        # state 의 history-4 stack 에 대한 lstm feature를 actorcritic 에 state로 추가
        lstm_outputs = []
        core_state = core_state.to(x.device)
        init_core_state = self.inital_state(batch_size=batch_size).to(x.device)
        step = 0
        for state, d in zip(torch.unbind(x, 0), torch.unbind(done_flags, 0)):
            core_state = torch.where(d.view(1, -1, 1), init_core_state, core_state)
            core_state = self.lstm(state, core_state.unbind(0))
            lstm_outputs.append(core_state[0])
            core_state = torch.stack(core_state, 0)  
            step += 1

        # print(f"step : {step}")
        x = torch.cat(lstm_outputs, 0)
        logits, value = self.head(x)           
        return self.head(x), core_state

    def get_policy_and_action(self, state, core_state):
        x = F.relu(self.fc(state))
        logits, value = self.head(x)        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        return action, logits.view(1, -1), core_state

    def forward(self, state, last_action, reward, done_flags, core_state=None, actor=False):
        # state 의 trajectory의 길이 단위별로 history 설정 및 batch size 만큼 차원변경
        seq_len, batch_size, x, last_actions, rewards, dones = reshape_stacked_state_dim(state, last_action, reward, done_flags, actor)
        # last_action = torch.zeros(last_action.shape[0], self.action_space,
        #                           dtype=torch.float32, device=state.device).scatter_(1, last_action, 1)

        # x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))

        # clipped_rewards = torch.clamp(reward, -1, 1)            
        # x = torch.cat((x, clipped_rewards, last_action), dim=1)

        x = x.view(seq_len, batch_size, -1)

        logits, values = self.head(x)      
        # (logits, values), hidden_state= self.unroll(x, batch_size,done_flags,core_state)
        # x = x.view(seq_len, batch_size, -1)
        # x = F.relu(self.fc(x))
        '''logits, values = self.actor_critic(x)'''

        # return logits.view(seq_len, -1), values.view(seq_len, -1)
        return (seq_len, batch_size, self.action_space, last_actions, rewards, dones),logits.view(seq_len * batch_size, self.action_space), values.view(seq_len * batch_size)
        
        '''
        if not actor: # target to learner 
            return logits.view(seq_len, -1, batch_size), values.view(seq_len, batch_size)
        else: # target to actor
            action = torch.softmax(logits, 1).multinomial(1).item() # Case atari, Categorical
            return action, logits.view(1, -1), hidden_state
        '''

class ActorCritic(nn.Module):
    def __init__(self,hidden_size, action_size):
        super().__init__()
        # self.fc = nn.Linear(input_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_size)
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x = F.relu(self.fc(x))
        logits = self.actor_linear(x)
        # logits = torch.softmax(self.actor_linear(x), dim=-1)
        values = self.critic_linear(x)
        return logits, values

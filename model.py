import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import *


class Network(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=256):
        super(Network, self).__init__()
        self.action_size = action_size
        # self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, 3)
        # self.fc = nn.Linear(3136, hidden_size)
        
        self.fc = nn.Linear(input_channels, hidden_size)
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTMCell(hidden_size*4 + action_size + 1, self.hidden_size)
        self.head = Head(hidden_size,action_size)
        
        self.dist = Categorical(hidden_size,action_size)

    def inital_state(self,batch_size=1):
        return torch.zeros((2, batch_size, self.hidden_size), dtype=torch.float32)

    def unroll(self, x, b_size, d_flags, core_state):
        lstm_outputs = []
        core_state = core_state.to(x.device)
        init_core_state = self.inital_state(batch_size=b_size).to(x.device)
        step = 0
        for s, d in zip(torch.unbind(x, 0), torch.unbind(d_flags, 0)):
            core_state = torch.where(d.view(1, -1, 1), init_core_state, core_state)
            core_state = self.lstm(s, core_state.unbind(0))
            lstm_outputs.append(core_state[0])
            core_state = torch.stack(core_state, 0)  
            step += 1

        # print(f"step : {step}")
        selected_core_states = torch.cat(lstm_outputs, 0)
        return selected_core_states, core_state
    
    def get_policy_and_action_with_unroll(self,x,last_action, reward, dones, core_state):
        
        last_action = torch.zeros(last_action.shape[0], self.action_size,
                                  dtype=torch.float32, device=x.device).scatter_(1, last_action, 1)
        
        x = F.relu(self.fc(x), inplace=True)
        x = x.view(x.shape[0], -1)
        
        x = torch.cat((x, reward, last_action), dim=1)
        x = x.view(1, 1, -1)
        
        selected_core_states, core_state = self.unroll(x,1,dones,core_state)

        logits, _ = self.head(selected_core_states)
        # dist = torch.distributions.Categorical(logits=logits)
        # action = dist.sample().item()
        
        action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
        # action = torch.softmax(logits, 1).multinomial(1).item()
        return action, logits, selected_core_states, core_state

    def get_policy(self, x):
        x = F.dropout(self.fc(x),p=0.8)
        x = F.relu(x)
        logits,values = self.head(x)
        return logits,values
    
    def get_policy_and_action(self,x):        
        x = F.dropout(self.fc(x),p=0.8)
        x = F.relu(x)
        logits,_ = self.head(x)
        
        action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
        
        # dist = self.dist(x)
        # logits = dist.probs
        # action = torch.argmax(logits).item()
        
        return action, logits

class Head(nn.Module):
    def __init__(self, hidden_size, action_space):
        super().__init__()
        self.actor_linear = nn.Linear(hidden_size, action_space)
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, final_activation):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = activation
        self.final_activation = final_activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        policy = self.final_activation(self.fc2(x))
        value = self.fc2(x)
        return policy, value

class CNNModel(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, activation, final_activation):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_size * 32 * 32, output_size)  # Assuming input size is 32x32
        self.activation = activation
        self.final_activation = final_activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        policy = self.final_activation(self.fc(x))
        value = self.fc(x)
        return policy, value

def build_model(s, ns, hidden, activation, output_size, final_activation, state_shape, unroll):
    state = s.view(-1, *state_shape)
    next_state = ns.view(-1, *state_shape)

    if len(state_shape) == 1:
        model = Model(state_shape[0], hidden, output_size, activation, final_activation)
        policy, value = model(state)
        _, next_value = model(next_state)

    elif len(state_shape) == 3:
        model = CNNModel(state_shape[0], hidden, output_size, activation, final_activation)
        policy, value = model(state)
        _, next_value = model(next_state)

    policy = policy.view(-1, unroll, output_size)
    value = value.view(-1, unroll)
    next_value = next_value.view(-1, unroll)

    return policy, value, next_value

# # 예제 입력 데이터
# s = torch.randn(5, 10, 3)  # [batch_size, time_steps, features]
# ns = torch.randn(5, 10, 3)  # [batch_size, time_steps, features]
# hidden = 128
# activation = F.relu
# output_size = 10
# final_activation = F.softmax
# state_shape = (10, 3)
# unroll = 5
# name = "example_model"

# policy, value, next_value = build_model(s, ns, hidden, activation, output_size, final_activation, state_shape, unroll, name)
# print(policy.shape, value.shape, next_value.shape)

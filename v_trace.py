import torch
import torch.nn as nn
import torch.optim as optim

def compute_delta_v(value_function, state, next_state, reward, gamma, rho):
    """
    Compute the temporal difference delta V.

    Parameters:
    - value_function: A function that approximates the value V(xs).
    - state: The current state.
    - next_state: The next state.
    - reward: The reward received.
    - gamma: Discount factor.
    - rho: Importance sampling ratio.

    Returns:
    - delta_v: The computed temporal difference delta V.
    """
    v_state = value_function(torch.FloatTensor(state)).item()
    v_next_state = value_function(torch.FloatTensor(next_state)).item()
    delta_v = rho * (reward + gamma * v_next_state - v_state)
    return delta_v

def compute_c_term(c, lambda_, s, t, on_policy):
    """
    Compute the c term for V-trace target.

    Parameters:
    - c: Importance sampling correction factor.
    - lambda_: Additional discounting parameter.
    - s: Start index.
    - t: Current index.
    - on_policy: Boolean indicating if the calculation is on-policy.

    Returns:
    - c_term: The computed c term.
    """
    if on_policy:
        return 1
    else:
        return lambda_ * torch.prod(torch.FloatTensor([min(c[i], 1) for i in range(s, t)]))

def compute_v_trace_target(value_function, trajectory, gamma, c, rho, lambda_=1.0, on_policy=False):
    """
    Compute the n-steps V-trace target for value approximation at state xs.

    Parameters:
    - value_function: A function that approximates the value V(xs).
    - trajectory: A list of tuples (state, action, reward) representing the trajectory.
    - gamma: Discount factor.
    - c: Importance sampling correction factor.
    - rho: Importance sampling ratio.
    - lambda_: Additional discounting parameter.
    - on_policy: Boolean indicating if the calculation is on-policy.

    Returns:
    - v_trace_target: The computed V-trace target.
    """
    states, actions, rewards = zip(*trajectory)
    values = value_function(torch.FloatTensor(states))
    v_trace_target = torch.zeros_like(values)

    for s in range(len(states)):
        vs = values[s]
        delta_s_v = compute_delta_v(value_function, states[s], states[s + 1] if s + 1 < len(states) else states[s], rewards[s], gamma, rho[s])
        vs += delta_s_v
        for t in range(s + 1, min(s + n, len(states))):
            gamma_term = gamma ** (t - s)
            c_term = compute_c_term(c, lambda_, s, t, on_policy)
            delta_t_v = compute_delta_v(value_function, states[t], states[t + 1] if t + 1 < len(states) else states[t], rewards[t], gamma, rho[t])
            vs += gamma_term * c_term * delta_t_v
        v_trace_target[s] = vs

    return v_trace_target

# Example usage
def value_function(states):
    # Placeholder for the actual value function
    return torch.zeros(len(states))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the value network
state_dim = 4  # Example state dimension for CartPole
value_network = ValueNetwork(state_dim)
optimizer = optim.Adam(value_network.parameters(), lr=0.01)

def value_function(states):
    """
    Approximates the value V(xs) for a batch of states.

    Parameters:
    - states: A batch of states.

    Returns:
    - values: The approximated values for the states.
    """
    value_network.eval()
    with torch.no_grad():
        values = value_network(torch.FloatTensor(states))
    return values

# Example usage
trajectory = [
    (torch.tensor([0.0, 0.0, 0.0, 0.0]), 0, 1.0),
    (torch.tensor([1.0, 0.0, 0.0, 0.0]), 1, 1.0),
    (torch.tensor([2.0, 0.0, 0.0, 0.0]), 0, 1.0)
]

gamma = 0.99
c = [0.9, 0.9, 0.9]  # Example importance sampling correction factors
rho = [1.0, 1.0, 1.0]  # Example importance sampling ratios
n = 2  # Example n-steps
lambda_ = 0.95  # Example additional discounting parameter

# Compute V-trace target for on-policy case
v_trace_target_on_policy = compute_v_trace_target(value_function, trajectory, gamma, c, rho, lambda_=lambda_, on_policy=True)
print("On-policy V-trace target:", v_trace_target_on_policy)

# Compute V-trace target for off-policy case
v_trace_target_off_policy = compute_v_trace_target(value_function, trajectory, gamma, c, rho, lambda_=lambda_, on_policy=False)
print("Off-policy V-trace target:", v_trace_target_off_policy)


trajectory = [
    (torch.tensor([0.0, 0.0]), 0, 1.0),
    (torch.tensor([1.0, 0.0]), 1, 1.0),
    (torch.tensor([2.0, 0.0]), 0, 1.0)
]

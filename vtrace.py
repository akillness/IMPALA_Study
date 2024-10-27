import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

# def compute_delta_v(value_function, state, next_state, reward, gamma, rho):
#     """
#     Compute the temporal difference delta V.

#     Parameters:
#     - value_function: A function that approximates the value V(xs).
#     - state: The current state.
#     - next_state: The next state.
#     - reward: The reward received.
#     - gamma: Discount factor.
#     - rho: Importance sampling ratio.

#     Returns:
#     - delta_v: The computed temporal difference delta V.
#     """
#     v_state = value_function(torch.FloatTensor(state)).item()
#     v_next_state = value_function(torch.FloatTensor(next_state)).item()
#     delta_v = rho * (reward + gamma * v_next_state - v_state)
#     return delta_v

# def compute_c_term(c, lambda_, s, t, on_policy):
#     """
#     Compute the c term for V-trace target.

#     Parameters:
#     - c: Importance sampling correction factor.
#     - lambda_: Additional discounting parameter.
#     - s: Start index.
#     - t: Current index.
#     - on_policy: Boolean indicating if the calculation is on-policy.

#     Returns:
#     - c_term: The computed c term.
#     """
#     if on_policy:
#         return 1
#     else:
#         return lambda_ * torch.prod(torch.FloatTensor([min(c[i], 1) for i in range(s, t)]))

# def compute_v_trace_target(value_function, trajectory, gamma, c, rho, lambda_=1.0, on_policy=False):
#     """
#     Compute the n-steps V-trace target for value approximation at state xs.

#     Parameters:
#     - value_function: A function that approximates the value V(xs).
#     - trajectory: A list of tuples (state, action, reward) representing the trajectory.
#     - gamma: Discount factor.
#     - c: Importance sampling correction factor.
#     - rho: Importance sampling ratio.
#     - lambda_: Additional discounting parameter.
#     - on_policy: Boolean indicating if the calculation is on-policy.

#     Returns:
#     - v_trace_target: The computed V-trace target.
#     """
#     states, actions, rewards = zip(*trajectory)
#     values = value_function(torch.FloatTensor(states))
#     v_trace_target = torch.zeros_like(values)

#     for s in range(len(states)):
#         vs = values[s]
#         delta_s_v = compute_delta_v(value_function, states[s], states[s + 1] if s + 1 < len(states) else states[s], rewards[s], gamma, rho[s])
#         vs += delta_s_v
#         for t in range(s + 1, min(s + n, len(states))):
#             gamma_term = gamma ** (t - s)
#             c_term = compute_c_term(c, lambda_, s, t, on_policy)
#             delta_t_v = compute_delta_v(value_function, states[t], states[t + 1] if t + 1 < len(states) else states[t], rewards[t], gamma, rho[t])
#             vs += gamma_term * c_term * delta_t_v
#         v_trace_target[s] = vs

#     return v_trace_target

# # Example usage
# def value_function(states):
#     # Placeholder for the actual value function
#     return torch.zeros(len(states))

# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


def log_probs_from_logits_and_actions(policy_logits, actions):
    assert policy_logits.dim() == 3, "policy_logits should have rank 3"
    assert actions.dim() == 2, "actions should have rank 2"
    return -F.cross_entropy(policy_logits, actions, reduction='none')

def from_logits(behavior_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, next_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    target_action_log_probs = log_probs_from_logits_and_actions(target_policy_logits, actions)
    behavior_action_log_probs = log_probs_from_logits_and_actions(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs

    transpose_log_rhos = log_rhos.transpose(0, 1)
    transpose_discounts = discounts.transpose(0, 1)
    transpose_rewards = rewards.transpose(0, 1)
    transpose_values = values.transpose(0, 1)
    transpose_next_value = next_value.transpose(0, 1)

    transpose_vs, transpose_clipped_rho = from_importance_weights(
        log_rhos=transpose_log_rhos,
        discounts=transpose_discounts,
        rewards=transpose_rewards,
        values=transpose_values,
        bootstrap_value=transpose_next_value[-1],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    return transpose_vs, transpose_clipped_rho

def log_probs_from_softmax_and_actions(policy_softmax, actions, action_size):
    onehot_action = F.one_hot(actions, action_size).float()
    selected_softmax = torch.sum(policy_softmax * onehot_action, dim=2)
    log_prob = torch.log(selected_softmax)
    return log_prob

def from_softmax(behavior_policy_softmax, target_policy_softmax, actions,
                 discounts, rewards, values, next_value, action_size,
                 clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
            
    target_action_log_probs = log_probs_from_softmax_and_actions(target_policy_softmax, actions, action_size)
    behavior_action_log_probs = log_probs_from_softmax_and_actions(behavior_policy_softmax, actions, action_size)
    log_rhos = target_action_log_probs - behavior_action_log_probs

    transpose_log_rhos = log_rhos.transpose(0, 1)
    transpose_discounts = discounts.transpose(0, 1)
    transpose_rewards = rewards.transpose(0, 1)
    transpose_values = values.transpose(0, 1)
    transpose_next_value = next_value.transpose(0, 1)

    transpose_vs, transpose_clipped_rho = from_importance_weights(
        log_rhos=transpose_log_rhos,
        discounts=transpose_discounts,
        rewards=transpose_rewards,
        values=transpose_values,
        bootstrap_value=transpose_next_value[-1],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    return transpose_vs, transpose_clipped_rho

def from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value,
                            clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    rhos = torch.exp(log_rhos) # rhos 의 최대 최소, 최대, 평균 구할수 있는 지점
    if clip_rho_threshold is not None:
        clipped_rhos = torch.minimum(torch.tensor(clip_rho_threshold), rhos)
    else:
        clipped_rhos = rhos
    
    cs = torch.minimum(torch.tensor(1.0), rhos)
    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    sequences = (discounts, cs, deltas)

    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = torch.zeros_like(bootstrap_value)
    vs_minus_v_xs = torch.zeros_like(values)

    for t in reversed(range(len(discounts))):
        vs_minus_v_xs[t] = scanfunc(vs_minus_v_xs[t + 1] if t + 1 < len(discounts) else initial_values, 
                                    (discounts[t], cs[t], deltas[t]))

    vs = vs_minus_v_xs + values

    return vs.detach(), clipped_rhos

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
'''
Target Policy
Target policy는 학습자(Learner)가 최적화하려는 정책입니다. 
이는 에이전트가 궁극적으로 따르기를 원하는 정책으로, 학습 과정에서 지속적으로 업데이트됩니다. 
Target policy는 학습자가 수집한 데이터를 기반으로 가치 함수와 정책을 업데이트하는 데 사용됩니다. 
이 정책은 주로 학습자의 신경망 파라미터로 표현됩니다

Local Policy
Local policy는 각 액터(actor)가 환경과 상호작용할 때 사용하는 정책입니다. 
IMPALA에서는 여러 액터가 병렬로 환경과 상호작용하며 데이터를 수집합니다. 
이때 각 액터는 자신의 로컬 정책(local policy)을 따릅니다. 
로컬 정책은 주기적으로 학습자의 타겟 정책으로부터 업데이트되지만, 항상 최신 상태는 아닐 수 있습니다. 
이는 분산 학습에서 발생하는 지연(latency) 때문입니다

Behaviour Policy
행동 정책은 에이전트가 환경에서 어떤 행동을 선택할지를 결정하는 정책입니다. 
IMPALA에서는 여러 개의 액터(actor)가 환경과 상호작용하며 데이터를 수집합니다. 
이 액터들은 행동 정책을 따르며, 이 정책은 학습자(learner)에 의해 주기적으로 업데이트됩니다. 
행동 정책은 주로 탐험(exploration)과 활용(exploitation) 사이의 균형을 맞추기 위해 설계됩니다.

Value Function
가치 함수는 특정 상태에서의 기대 보상을 추정하는 함수입니다. 
IMPALA에서는 V-trace라는 오프-폴리시(off-policy) 보정 방법을 사용하여 가치 함수를 추정합니다. 
V-trace는 액터들이 수집한 데이터를 학습자가 효과적으로 사용할 수 있도록 도와줍니다. 
이를 통해 학습자는 더 안정적이고 효율적으로 학습할 수 있습니다.


'''
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
    onehot_action = F.one_hot(actions.long(), action_size).float()
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

import torch
import torch.nn.functional as F

def compute_value_loss(vs, value):
    error = vs[:, 0].detach() - value[:, 0]
    l2_loss = torch.square(error)
    return torch.sum(l2_loss) * 0.5

def compute_policy_loss(softmax, actions, advantages, output_size):
    onehot_action = F.one_hot(actions, output_size).float()
    selected_softmax = torch.sum(softmax * onehot_action, dim=2)
    cross_entropy = torch.log(selected_softmax)
    advantages = advantages.detach()
    policy_gradient_loss_per_timestep = cross_entropy[:, 0] * advantages[:, 0]
    return -torch.sum(policy_gradient_loss_per_timestep)

def compute_entropy_loss(softmax):
    policy = softmax
    log_policy = torch.log(softmax)
    entropy_per_time_step = -policy * log_policy
    entropy_per_time_step = torch.sum(entropy_per_time_step[:, 0], dim=1)
    return -torch.sum(entropy_per_time_step)


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

    rhos = torch.exp(log_rhos)
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

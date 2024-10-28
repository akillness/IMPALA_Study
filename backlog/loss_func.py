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

import torch
import datetime

import numpy as np

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterServer(object):
    def __init__(self, lock):
        self.lock = lock
        self.weight = None

    def pull(self):
        # with self.lock:
        return self.weight

    def push(self, weigth):
        # with self.lock:
        self.weight = weigth

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


'''
def test_policy(
    policy,
    env,
    episodes: int,
    deterministic: bool,
    max_episode_len: int,
    log_dir,
    verbose: bool = False,
):
    start_time = datetime.datetime.now()
    start_text = f"Started testing at {start_time:%d-%m-%Y %H:%M:%S}\n"
    
    # if type(env) == str:
    #     env = make_env(env)
    
    # if log_dir is not None:
    #     Path(log_dir).mkdir(parents=True, exist_ok=True)
    #     fpath = Path(log_dir).joinpath(f"test_log_{start_time:%d%m%Y%H%M%S}.txt")
    #     fpath.write_text(start_text)
        
    if verbose:
        print(start_text)
        
    policy.eval()
    rewards = []
    for e in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, device=device, dtype=dtype)
        d = False
        ep_rewards = []
        for t in range(max_episode_len):
            env.render()
            action, _ = policy.select_action(obs, deterministic)
            obs, r, d, _, _ = env.step(action.item())
            obs = torch.tensor(obs, device=device, dtype=dtype)
            ep_rewards.append(r)
            if d:
                break
        rewards.append(sum(ep_rewards))
        ep_text = f"Episode {e+1}: Reward = {rewards[-1]:.2f}\n"
        if log_dir is not None:
            with open(fpath, mode="a") as f:
                f.write(ep_text)
        if verbose:
            print(ep_text)
    avg_reward = np.mean(rewards)
    std_dev = np.std(rewards)
    complete_text = (
        f"-----\n"
        f"Testing completed in "
        f"{(datetime.datetime.now() - start_time).seconds} seconds\n"
        f"Average Reward per episode: {avg_reward}"
    )
    if verbose:
        print(complete_text)
    if log_dir is not None:
        with open(fpath, mode="a") as f:
            f.write(complete_text)

    return avg_reward, std_dev'''
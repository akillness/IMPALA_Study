"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import torch
import torch.nn.functional as F

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages clipped_rhos")


def log_probs_from_logits_and_actions(policy_logits, actions):
    """Computes action log-probs from policy logits and actions.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions.

    Args:
      policy_logits: A float32 tensor of shape [T, NUM_ACTIONS, B] with
        un-normalized log-probabilities parameterizing a softmax policy.
      actions: An int32 tensor of shape [T, B] with actions.

    Returns:
      A float32 tensor of shape [T, B] corresponding to the sampling log
      probability of the chosen action w.r.t. the policy.
    """
    # policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    # actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # assert len(policy_logits.shape) == 3
    # assert len(actions.shape) == 2

    return -F.cross_entropy(policy_logits, actions, reduction='none')


def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # cs = torch.min(torch.ones_like(rhos), rhos)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
        
        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(cs.shape[0] - 1, -1, -1):
            # acc = deltas[t] + discounts[t] * cs[t] * acc
            # acc = deltas[t] + discounts * cs[t] * (acc-values)
            acc = deltas[t] + discounts * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)
        
        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)
        
        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos

        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
        
        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach(), clipped_rhos=clipped_rhos)

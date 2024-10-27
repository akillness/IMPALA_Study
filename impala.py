
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from actor import ActorCritic, Actor
from learner import Learner

# import vtrace

class IMPALA(nn.Module):

    def __init__(self,state_shape, output_size, policy, behavior_policy, value_function, discount_factor, lr,hidden, coef, reward_clip, unroll):
        super(IMPALA, self).__init__()
        self.state_shape = state_shape
        self.output_size = output_size

        self.policy = policy  # Target policy πρ¯
        self.behavior_policy = behavior_policy  # Behavior policy µ
        self.value_function = value_function  # Value function
        self.discount_factor = discount_factor  # Discount factor
        self.lr = lr  # Learning rate

        self.clip_rho_threshold = 1.0
        self.clip_pg_rho_threshold = 1.0

        self.unroll = unroll
        self.trajectory_size = unroll + 1
        self.coef = coef
        self.reward_clip = reward_clip

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr, eps=0.01, momentum=0.0, alpha=0.99)

        self.policy = ActorCritic(state_shape,output_size)
        # self.policy_net = self.policy.actor
        # self.value_net = self.policy.critic

    def foward(self,x):
        return self.policy(x)
    
    def train(self, state, next_state, reward, done, action, behavior_policy):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.BoolTensor(done)
        action = torch.LongTensor(action)
        behavior_policy = torch.FloatTensor(behavior_policy)

        # Compute clipped rewards
        if self.reward_clip == 'tanh':
            squeezed = torch.tanh(reward / 5.0)
            clipped_rewards = torch.where(reward < 0, 0.3 * squeezed, squeezed) * 5.0
        elif self.reward_clip == 'abs_one':
            clipped_rewards = torch.clamp(reward, -1.0, 1.0)
        elif self.reward_clip == 'no_clip':
            clipped_rewards = reward

        discounts = (~done).float() * self.discount_factor

        policy, value = self(state)
        _, next_value = self(next_state)

        vs, rho = self.compute_v_trace(behavior_policy, policy, action, discounts, clipped_rewards, value, next_value)

        pg_advantage = rho * (clipped_rewards + self.discount_factor * (1 - done) * vs - value)

        loss = self.compute_loss(vs, value, policy, action, pg_advantage)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_v_trace(self, behavior_policy, target_policy, actions, discounts, rewards, values, next_values, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
        with torch.no_grad():

            target_action_log_probs = torch.log(torch.gather(target_policy, dim=2, index=actions.unsqueeze(-1)).squeeze(-1))
            behavior_action_log_probs = torch.log(torch.gather(behavior_policy, dim=2, index=actions.unsqueeze(-1)).squeeze(-1))
            log_rhos = target_action_log_probs - behavior_action_log_probs

            rhos = torch.exp(log_rhos)
            
            if clip_rho_threshold is not None:
                clipped_rhos = torch.minimum(torch.tensor(clip_rho_threshold), rhos)
            else:
                clipped_rhos = rhos
            
            cs = torch.minimum(torch.tensor(1.0), rhos)
            values_t_plus_1 = torch.cat([values[1:], next_values[-1].unsqueeze(0)], dim=0)
            
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
            
            vs_minus_v_xs = torch.zeros_like(values)
            vs_minus_v_xs[-1] = deltas[-1]
            
            for t in reversed(range(len(discounts) - 1)):
                vs_minus_v_xs[t] = deltas[t] + discounts[t] * cs[t] * vs_minus_v_xs[t + 1]
            
            vs = vs_minus_v_xs + values
            
            pg_advantages = clipped_rhos * (rewards + discounts * next_values - values)
            
        return vs, pg_advantages

    def compute_loss(self, vs, values, policy, actions, pg_advantage):
        value_loss = nn.MSELoss()(vs, values)
        entropy_loss = -torch.mean(torch.sum(policy * torch.log(policy + 1e-10), dim=-1))
        policy_loss = -torch.mean(pg_advantage * torch.log(policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)))
        total_loss = policy_loss + value_loss + self.coef * entropy_loss
        return total_loss
    
    def policy_and_action(self):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.policy(state)
        policy = policy.squeeze(0).detach().numpy()
        action = np.random.choice(self.output_size, p=policy)
        return action, policy, max(policy)




    
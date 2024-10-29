
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import loss_func

from actor_modify import ActorCritic, Actor
from learner import Learner

import core
import vtrace

class IMPALA(nn.Module):

    def __init__(self,state_shape, output_size,activation,final_activation,discount_factor, lr, hidden, entropy_coef, reward_clip, unroll):
        super(IMPALA, self).__init__()

        self.device = "cpu"
        if torch.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

        '''
            # Target Policy : πρ¯
            Target policy는 학습자(Learner)가 최적화하려는 정책입니다. 
            이는 에이전트가 궁극적으로 따르기를 원하는 정책으로, 학습 과정에서 지속적으로 업데이트됩니다. 
            Target policy는 학습자가 수집한 데이터를 기반으로 가치 함수와 정책을 업데이트하는 데 사용됩니다. 
            이 정책은 주로 학습자의 신경망 파라미터로 표현됩니다

            # Behaviour Policy : µ
            행동 정책은 에이전트가 환경에서 어떤 행동을 선택할지를 결정하는 정책입니다. 
            IMPALA에서는 여러 개의 액터(actor)가 환경과 상호작용하며 데이터를 수집합니다. 
            이 액터들은 행동 정책을 따르며, 이 정책은 학습자(learner)에 의해 주기적으로 업데이트됩니다. 
            행동 정책은 주로 탐험(exploration)과 활용(exploitation) 사이의 균형을 맞추기 위해 설계됩니다.

            # Local Policy
            Local policy는 각 액터(actor)가 환경과 상호작용할 때 사용하는 정책입니다. 
            IMPALA에서는 여러 액터가 병렬로 환경과 상호작용하며 데이터를 수집합니다. 
            이때 각 액터는 자신의 로컬 정책(local policy)을 따릅니다. 
            로컬 정책은 주기적으로 학습자의 타겟 정책으로부터 업데이트되지만, 항상 최신 상태는 아닐 수 있습니다. 
            이는 분산 학습에서 발생하는 지연(latency) 때문입니다

            # Value Function : v trace
            가치 함수는 특정 상태에서의 기대 보상을 추정하는 함수입니다. 
            IMPALA에서는 V-trace라는 오프-폴리시(off-policy) 보정 방법을 사용하여 가치 함수를 추정합니다. 
            V-trace는 액터들이 수집한 데이터를 학습자가 효과적으로 사용할 수 있도록 도와줍니다. 
            이를 통해 학습자는 더 안정적이고 효율적으로 학습할 수 있습니다.

        '''
        
        self.discount_factor = discount_factor  # Discount factor
        self.lr = lr  # Learning rate

        self.clip_rho_threshold = 1.0
        self.clip_pg_rho_threshold = 1.0

        self.unroll = unroll
        self.trajectory_size = unroll + 1
        self.coef = entropy_coef # entropy_coef = 0.01
        self.reward_clip = reward_clip # reward_clip = ['tanh', 'abs_one', 'no_clip']

        self.local = ActorCritic(state_shape[0],output_size).to(self.device)
        # self.policy_net = self.policy.actor
        # self.value_net = self.policy.critic

        self.state_shape = state_shape
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.hidden = hidden
        self.clip_rho_threshold = 1.0
        self.clip_pg_rho_threshold = 1.0
        self.discount_factor = 0.99
        self.lr = 0.001
        self.unroll = unroll
        self.trajectory_size = unroll + 1
        self.coef = entropy_coef
        self.reward_clip = reward_clip

        self.batch_size = 32 # ph -> graph
        self.s_ph = torch.zeros((self.unroll, *self.state_shape), dtype=torch.float32).to(self.device)
        self.ns_ph = torch.zeros((self.unroll, *self.state_shape), dtype=torch.float32).to(self.device)
        self.a_ph = torch.zeros((self.unroll), dtype=torch.int32).to(self.device)
        self.d_ph = torch.zeros((self.unroll), dtype=torch.bool).to(self.device)
        self.behavior_policy = torch.zeros(( self.unroll, self.output_size), dtype=torch.float32).to(self.device)
        self.r_ph = torch.zeros((self.unroll), dtype=torch.float32).to(self.device)

        if self.reward_clip == 'tanh':
            squeezed = torch.tanh(self.r_ph / 5.0)
            self.clipped_rewards = torch.where(self.r_ph < 0, 0.3 * squeezed, squeezed) * 5.0
        elif self.reward_clip == 'abs_one':
            self.clipped_rewards = torch.clamp(self.r_ph, -1.0, 1.0)
        elif self.reward_clip == 'no_clip':
            self.clipped_rewards = self.r_ph

        self.discounts = (~self.d_ph).float() * self.discount_factor

        # self.policy, self.value, self.next_value = core.build_model(
        #     self.s_ph, self.ns_ph, self.hidden, self.activation, self.output_size,
        #     self.final_activation, self.state_shape, self.unroll).to(self.device)
        
        self.policy,self.value = self.local(self.s_ph)
        self.policy = self.policy.view(-1,unroll,output_size) 
        self.value = self.value.view(-1,unroll)
        _, self.next_value = self.local(self.ns_ph.view(-1, *state_shape))
        self.next_value = self.next_value.view(-1,unroll)

        self.transpose_vs, self.transpose_clipped_rho = vtrace.from_softmax(
            behavior_policy_softmax=self.behavior_policy,
            target_policy_softmax=self.policy,
            actions=self.a_ph, discounts=self.discounts, rewards=self.clipped_rewards,
            values=self.value, bootstrap_value=self.next_value, action_size=self.output_size).to(self.device)

        self.vs = self.transpose_vs.transpose(0, 1)
        self.rho = self.transpose_clipped_rho.transpose(0, 1)

        self.vs_ph = torch.zeros((self.unroll), dtype=torch.float32).to(self.device)
        self.pg_advantage_ph = torch.zeros(( self.unroll), dtype=torch.float32).to(self.device)

        # loss term
        self.value_loss = loss_func.compute_value_loss(self.vs_ph, self.value)
        self.entropy = loss_func.compute_entropy_loss(self.policy)
        self.pi_loss = loss_func.compute_policy_loss(self.policy, self.a_ph, self.pg_advantage_ph, self.output_size)

        self.total_loss = self.pi_loss + self.value_loss + self.entropy * self.coef
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr, eps=0.01, momentum=0.0, alpha=0.99)

        # self.lstm = nn.LSTMCell(hidden + output_size + 1, hidden)
        # self.fc = nn.Linear(hidden, output_size)

    def foward(self,x):
        # lstm_out, hidden = self.lstm(x, hidden)
        return self.policy(x)
    
    # learner 기능
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

    # v trace는 value function
    # 특정상태에서 기대 보상을 추정하는 함수
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
        '''
        # self.lstm(state,self.hidden)
        # train_data = torch.randn(100, 10, 1)  # [batch_size, time_steps, features]
        # train_labels = torch.randn(100, 1)    # [batch_size, output_size]
        '''
        policy, _ = self.policy(state)
        policy = policy.squeeze(0).detach().numpy()
        action = np.random.choice(self.output_size, p=policy)
        return action, policy, max(policy)

    def test(self, state, action, reward, done, behavior_policy):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)
        behavior_policy = torch.tensor(behavior_policy, dtype=torch.float32).to(self.device)

        # Now you can use these tensors in your model



    
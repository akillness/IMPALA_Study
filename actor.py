# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""

import torch
from model import IMPALA

class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []
        self.actor_id = None
        self.lstm_hidden_state = None

    def append(self, state, action, reward, done, logit):
        self.obs.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logit.append(logit)
        self.dones.append(done)

    def finish(self):
        self.obs = torch.stack(self.obs)
        self.rewards = torch.cat(self.rewards, 0).squeeze()
        self.actions = torch.cat(self.actions, 0).squeeze()
        self.dones = torch.cat(self.dones, 0).squeeze()
        self.logit = torch.cat(self.logit, 0)

    def cuda(self):
        self.obs = self.obs.cuda()
        self.actions = self.actions.cuda()
        self.dones = self.dones.cuda()
        self.lstm_hidden_state = self.lstm_hidden_state.cuda()
        self.rewards = self.rewards.cuda()
        self.logit = self.logit.cuda()

    def mps(self):
        device = torch.device("mps")
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        self.lstm_hidden_state = self.lstm_hidden_state.to(device)
        self.rewards = self.rewards.to(device)
        self.logit = self.logit.to(device)

    def get_last(self):
        """must call this function before finish()"""
        obs = self.obs[-1]
        logits = self.logit[-1]
        last_action = self.actions[-1]
        reward = self.rewards[-1]
        done = self.dones[-1]
        return obs, last_action, reward, done, logits

    @property
    def length(self):
        return len(self.rewards)

    def __repr__(self):
        return "ok" # End of episode when life lost "Yes"


def actor(idx, experience_queue, sync_ps, env, args):
    """Simple actor """
    steps = 0
    length = args.length
    action_size = args.action_size
    model = IMPALA(action_size=action_size)
    init_lstm_state = torch.zeros((2, 1, 256), dtype=torch.float32)
    
    env.start()
    """Run the env for n steps and return a trajectory rollout."""
    obs = env.reset()
    hidden_state = init_lstm_state
    logits = torch.zeros((1, action_size), dtype=torch.float32)
    last_action = torch.zeros((1, 1), dtype=torch.int64)
    reward = torch.tensor(0, dtype=torch.float32).view(1, 1)
    done = torch.tensor(True, dtype=torch.bool).view(1, 1)
    init_state = (obs, last_action, reward, done, logits)
    persistent_state = init_state
    rewards = 0
    while True:
        # Sync trained model
        model.load_state_dict(sync_ps.pull())
        
        rollout = Trajectory()
        rollout.actor_id = idx
        rollout.lstm_hidden_state = hidden_state.squeeze()
        rollout.append(*persistent_state)
        total_reward = 0
        with torch.no_grad():
            while True:
                if rollout.length == length + 1:
                    rewards += total_reward
                    persistent_state = rollout.get_last()
                    rollout.finish()
                    # Queue trajectory data( all of state )
                    experience_queue.put(rollout)
                    print(f"Trajectory is full, Actor({idx}) / Total Reward : {total_reward} ")
                    break
                if done:
                    rewards = 0.
                    steps = 0
                    hidden_state = init_lstm_state
                    __, last_action, reward, done, _ = init_state
                    obs = env.reset()

                action, logits, hidden_state = model(obs.unsqueeze(0), last_action, reward,
                                           done, hidden_state, actor=True)
                obs, reward, done = env.step(action)
                total_reward += reward
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
                done = torch.tensor(done, dtype=torch.bool).view(1, 1)
                rollout.append(obs, last_action, reward, done, logits.detach())
                steps += 1

    env.close()
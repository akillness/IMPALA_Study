# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""

import torch, os
import numpy as np
from model import Network

from environment import CartPole

from torch.utils.tensorboard import SummaryWriter

from random import shuffle
# # 호환성을 위한 bool8 정의
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
    
class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self, max_size):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.last_actions = []
        self.actor_id = None
        # self.lstm_hx = None
        
        self.max_size = max_size
        self.position = 0
          
    def append(self, state, action, reward, done, logit):
        if len(self.rewards) < self.max_size+1:
            self.obs.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.logits.append(None)
            self.dones.append(None)
        
        self.obs[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.logits[self.position] = logit
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.max_size
        
        # self.obs.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.logit.append(logit)
        # self.dones.append(done)

    def finish(self):
        self.obs = torch.stack(self.obs)
        # step_reward = 0.
        # for i in range(len(self.rewards) - 1, -1, -1):
        #     if self.rewards[i] is not None:
        #         step_reward *= gamma
        #         step_reward += self.rewards[i]
        #         self.rewards[i] = step_reward
                
        self.rewards = torch.stack(self.rewards).squeeze()
        self.actions = torch.stack(self.actions).squeeze()
        self.dones = torch.stack(self.dones).squeeze()
        self.logits = torch.stack(self.logits)
        
        

    def cuda(self):        
        self.obs = self.obs.cuda()
        self.actions = self.actions.cuda()
        self.dones = self.dones.cuda()
        # self.lstm_hx = self.lstm_hx.cuda()
        self.rewards = self.rewards.cuda()
        self.logits = self.logits.cuda()

    def mps(self):
        device = torch.device("mps")
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        # self.lstm_hx = self.lstm_hx.to(device)
        self.rewards = self.rewards.to(device)
        self.logits = self.logits.to(device)
        
    def get_last(self):
        """must call this function before finish()"""
        obs = self.obs[-1]
        logits = self.logits[-1]
        last_action = self.actions[-1]
        reward = self.rewards[-1]
        done = self.dones[-1]
        return obs, last_action, reward, done, logits

    def get(self,device):                
        # # 텐서로 변환 및 차원 조정        
        return self.logits.to(device), self.obs.to(device), self.actions.to(device), self.rewards.to(device), self.dones.to(device) #, self.lstm_hx.to(device)

    @property
    def length(self):
        return len(self.rewards)

    def __repr__(self):
        return "ok"


def actor(idx, ps, data, args, terminate_event): #, env, args):
    """Simple actor """
    episode = 0
    length = args.length
    action_size = args.action_size
    local_network = Network(action_size=action_size)
    # init_core_state = local_network.inital_state()
    # save_path = args.save_path
    # load_path = args.load_path
    # env.start()
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    """Run the env for n steps and return a trajectory rollout."""
    env = CartPole(game_name=args.game_name,seed=args.seed)
    
    # state = torch.zeros((local_network.hidden_size,), dtype=torch.float32)
    '''
    core_state = init_core_state
    logits = torch.zeros((1, action_size), dtype=torch.float32)
    last_action = torch.zeros((1, 1), dtype=torch.int64)
    reward = torch.tensor(1, dtype=torch.float32).view(1, 1)
    done = torch.tensor(False, dtype=torch.bool).view(1, 1)
    init_state = (obs, last_action, reward, done, logits)
    persistent_state = init_state
    '''
    
    episode = 0
    is_done = False
    while not terminate_event.is_set():
        obs = env.reset()
        # print("Actor: {} Steps: {} Reward: {}".format(idx, steps, rewards))
        
        with ps.lock:
            local_network.load_state_dict(ps.pull())
        
        
            
        rollout = Trajectory(max_size=length)
        # print("Actor: {} trajectory init".format(idx))
        rollout.actor_id = idx
        # rollout.lstm_hx = core_state.squeeze()
        # rollout.append(*persistent_state)
        total_reward = 0
        
        with torch.no_grad():
            '''
            while True:
                if rollout.length == length + 1:
                    # rewards += total_reward
                    # persistent_state = rollout.get_last()
                    rollout.finish()
                    data.put(rollout)
                    break
                if is_done:
                    # rewards = 0.
                    # steps = 0
                    core_state = init_core_state
                    __, last_action, reward, done, _ = init_state
                    obs = env.reset()

                # action, logits, core_state = local_network(obs.unsqueeze(0).unsqueeze(1), last_action, reward,
                #                            done, core_state, actor=True)
                action, logits, selected_core_states, core_state = local_network.get_policy_and_action(obs.unsqueeze(0), last_action, reward,
                                           done, core_state)
                obs, r, is_done = env.step(action)
                
                # d = False
                # if reward == 1 or reward ==-1:
                #     d = True
                
                total_reward += r
                
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(r, dtype=torch.float32).view(1, 1)
                done = torch.tensor(is_done, dtype=torch.bool).view(1, 1)
                # rollout.append(obs, last_action, reward, done, logits.detach())                
                flatten_core_state = torch.flatten(selected_core_states) # 텐서를 평탄화
                rollout.append(flatten_core_state, last_action, reward, done, logits.detach())
                
                # if is_done:                    
                #     break
                
                while terminate_event.is_set():
                    if args.verbose == 1:
                        print(f"Actor {idx} terminating.")
                        break
            '''
            # while True:
            for _ in range(args.length):                
                action, logits = local_network.get_policy_and_action(obs)
                obs, r, is_done = env.step(action)
 
                total_reward += r
                
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(r, dtype=torch.float32).view(1, 1)
                done = torch.tensor(is_done, dtype=torch.bool).view(1, 1)
                rollout.append(obs, last_action, reward, done, logits.detach())                                
                
                if is_done:                                 
                    break
            
            rollout.finish()
            data.put(rollout)
            
            # 새로운 모델 받기
            # if os.path.exists(args.load_path):
            #     local_network.cpu()
            #     local_network.state_dict(torch.load(args.load_path, map_location= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), weights_only=True))
            
            if args.verbose == 1:
                print("Actor: {} put Steps: {} rewards:{}".format(idx, episode, total_reward))
                    
            writer.add_histogram(
                f"actor_{idx}/actions/action_taken", action, episode
            )
            writer.add_histogram(
                f"actor_{idx}/actions/logits", logits.detach(), episode
            )
            writer.add_scalar(
                f"actor_{idx}/rewards/trajectory_reward",
                total_reward,
                episode,
            )
            writer.close()
            episode += 1
            
            while terminate_event.is_set():
                if args.verbose == 1:
                    print(f"Actor {idx} terminating.")
                    break

    env.close()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

from torch.optim.lr_scheduler import PolynomialLR, LambdaLR

import vtrace

import time

from torch.utils.tensorboard import SummaryWriter

from impala import IMPALA

from environment import CartPole,get_action_size

def transpose_batch_to_stack(batch,device):
    states = []
    actions = []
    rewards = []
    dones = []
    logits = []

    for t in batch:
        state, action, reward, done, logit = zip(*t.memory)
        states.extend(state)
        rewards.extend(reward)
        dones.extend(done)
        actions.extend(action)
        logits.extend(logit)

    state = torch.stack(states).transpose(0, 1).squeeze().to(device)
    actions = torch.stack(actions).transpose(0, 1).squeeze().to(device)
    rewards = torch.stack(rewards).transpose(0, 1).squeeze().to(device)
    dones = torch.stack(dones).transpose(0, 1).squeeze().to(device)
    logits = torch.stack(logits).transpose(0,1).squeeze().to(device)
    return logits, state, actions, rewards, dones 

# def compute_baseline_loss(advantages):
#     return 0.5 * torch.sum(advantages ** 2)

# def compute_entropy_loss(logits):
#     """Return the entropy loss, i.e., the negative entropy of the policy."""
#     policy = F.softmax(logits, dim=-1)
#     log_policy = F.log_softmax(logits, dim=-1)
#     return -torch.sum(-policy * log_policy)

# def compute_policy_gradient_loss(logits, actions, advantages):
#     cross_entropy = F.cross_entropy(logits, actions, reduction='none')# sparse_softmax_cross_entropy_with_logits
#     return torch.sum(cross_entropy * advantages.detach())


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(logits, dim=-1), target=torch.flatten(actions), reduction="none",
    # ).view_as(advantages)
    # return torch.sum(cross_entropy * advantages.detach())

    cross_entropy = F.nll_loss(
        F.log_softmax(logits, dim=-1), 
        target=torch.flatten(actions), 
        reduction="none"
    )
    
    # Ensure advantages are 1D: [68] (adjust according to your advantage calculation)
    # Example: If advantages are 2D but should use diagonal elements
    if len(advantages.shape) == 2:
        advantages = advantages.diagonal()  # Converts to 1D [68]
    ''' 
    # Gather advantages corresponding to taken actions (actions must be 0 or 1)
    selected_advantages = advantages[torch.arange(actions.size(0)), actions]
    '''
    # Compute loss
    loss = (cross_entropy * advantages).mean()
    
    return loss

class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self,max_size):
        self.actor_id = None
        # self.lstm_core_state = None
        self.max_size = max_size
        self.memory = []
        self.position = 0
    
    def clear(self):
        self.memory.clear()

    def append(self, state, action, reward, done, logit):
        
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        
        self.memory[self.position] = [state,action,reward,done,logit]
        self.position = (self.position + 1) % self.max_size
    
        
    def finish(self,GAMMA):
        # self.state = torch.stack(self.state).squeeze()
        # 역방향으로 감쇄 적용
        step_reward = 0.
        for i in range(len(self.memory) - 1, -1, -1):
            if self.memory[i][2] is not None:
                step_reward *= GAMMA
                step_reward += self.memory[i][2]
                self.memory[i][2] = step_reward
        return step_reward

    def get_last(self):
        """must call this function before finish()"""
        state = self.state[-1]
        # next_state = self.next_state[-1]
        logits = self.logit[-1]
        last_action = self.actions[-1]
        reward = self.rewards[-1]
        done = self.dones[-1]
        return state, last_action, reward, done, logits
    
    def sample(self, batch_size, device):
        """배치 샘플링"""
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in memory.")
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, dones, logits = zip(*batch)
        
        # 텐서로 변환 및 차원 조정
        states = torch.stack(states).squeeze().to(device)
        actions = torch.cat(actions).squeeze().to(device)
        # next_states = torch.stack(next_states).to(device)
        rewards = torch.cat(rewards).squeeze().to(device)
        dones = torch.cat(dones).squeeze().to(device)
        logits = torch.stack(logits).squeeze().to(device)

        
        return states, actions, rewards, dones, logits
    
    def get(self,device):
        states, actions, rewards, dones, logits = zip(*self.memory)
        
        # 텐서로 변환 및 차원 조정
        states = torch.stack(states).squeeze().to(device)
        actions = torch.cat(actions).squeeze().to(device)
        # next_states = torch.stack(next_states).to(device)
        rewards = torch.cat(rewards).squeeze().to(device)
        dones = torch.cat(dones).squeeze().to(device)
        logits = torch.stack(logits).squeeze().to(device)

        
        return states, actions, rewards, dones, logits

    @property
    def length(self):
        return len(self.memory)

    def __repr__(self):
        return "ok" # End of episode when life lost "Yes"


# def actor(idx, experience_queue, sync_ps, env, args, terminate_event):
def actor(idx, experience_queue, learner_model,  sync_ps, args, terminate_event):
    
    total_steps = args.total_steps
    length = args.length
    action_size = args.action_size
    actor_model = IMPALA(action_size=action_size)
    # actor_model.share_memory()
    
    # optimizer = optim.Adam(learner_model.parameters(), lr=args.lr, eps=args.epsilon,
    #                    weight_decay=args.decay)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        eps=args.epsilon,
        alpha=args.decay,
    )

    lr_lambda = lambda epoch : 1 - min(epoch * args.length * args.batch_size, args.total_steps) / args.total_steps
    scheduler = LambdaLR(optimizer,lr_lambda)
    
    # scheduler = PolynomialLR(optimizer, total_iters=args.total_steps, power=1.0)
    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.log_dir)
    
    discount_gamma = args.gamma
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost

    step = 0
    best = 0

    env = CartPole(game_name=args.game_name,seed=args.seed)
    
    
    
    total_reward = 0.
    rollout_cnt = 0   
    while not terminate_event.is_set():
        # Sync actor model's wieghts
        # with sync_ps.lock:
        # actor_model.load_state_dict(learner_model.state_dict())
        actor_model.load_state_dict(sync_ps.pull())

        rollout = Trajectory(max_size=length)
        rollout.actor_id = idx
        # rollout.lstm_core_state = core_state.squeeze()
        # rollout.append(*persistent_state)
        
        # """Run the env for n steps and return a trajectory rollout."""
        obs = env.reset()
        with torch.no_grad():
              
            while rollout.length < length + 1:                                    
                action, logits = actor_model.get_policy_and_action(obs)
                obs, rewards, d = env.step(action)
        
                # total_reward += reward
                last_actions = torch.tensor(action, dtype=torch.int64).view(1, 1)
                rewards = torch.tensor(rewards, dtype=torch.float32).view(1, 1)
                dones = torch.tensor(d, dtype=torch.bool).view(1, 1)

                rollout.append(obs, last_actions, rewards, dones, logits.detach())
                rollout_cnt += 1
                if d:
                    break
            
            total_reward = rollout.finish(args.gamma)
            if args.verbose == 1:
                print(f"Actor:{idx}, completed Reward :{total_reward}")
            
            writer.add_histogram(
                    f"actor_{idx}/actions/action_taken", action, rollout_cnt
                )
            writer.add_histogram(
                f"actor_{idx}/actions/logits", logits.detach(), rollout_cnt
            )
            writer.add_scalar(
                f"actor_{idx}/rewards/trajectory_reward",
                total_reward,
                rollout_cnt,
            )
            writer.close()
            
        # learner
        total_loss = 0.0 
        value_loss = 0.0 
        policy_loss = 0.0 
        policy_entropy = 0.0

        reward = 0.0
        importance_sampling_ratios = 0.0 
        
        batch_size = args.batch_size
        # batch.append(rollout)
        
        traj_cnt = 0
        
        while traj_cnt < batch_size:
            
            if torch.cuda.is_available():
                device = torch.device("gpu")
                learner_model.cuda()
            elif torch.mps.is_available():
                device = torch.device("mps")
                learner_model.to(device)
        
        # if len(batch) < batch_size:
        #     continue
        
            # check batch time
            start_batch_time = time.time()
            batch_time = time.time() - start_batch_time
            
            # state, actions, rewards, dones, behavior_logits = rollout.sample(batch_size=args.batch_size,device=device)
            state, actions, rewards, dones, behavior_logits = rollout.get(device=device)
            # behavior_logits, state, actions, rewards, dones = transpose_batch_to_stack(batch=batch,device=device)
            '''
            # print(f"Trajectories : {len(state)}")
            if args.reward_clip == "abs_one":
                clipped_rewards = torch.clamp(rewards, -1, 1)
            elif args.reward_clip == 'soft_asymmetric':
                squeezed = torch.tanh(rewards / 5.0)
                # Negative rewards are given less weight than positive rewards.
                clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0
            else:
                clipped_rewards = rewards
            '''
            
            # check forward time 
            start_forward_time = time.time()
            # reshape, target_logits, target_values = learner_model(state, actions, clipped_rewards, dones)
            
            target_logits, target_values = learner_model.get_policy(state)

            forward_time = time.time() - start_forward_time        
            actions, behavior_logits, dones, rewards = actions[1:], behavior_logits[1:], dones[1:], rewards[1:]
            bootstrap_value = target_values[-1]
            # target_logits, target_values = target_logits[:-1], target_values[:-1]
            target_logits = target_logits[:-1]
            discounts = (~dones).float() * discount_gamma
            
            # 행동 정책과 목표 정책의 로그 확률 계산
            # 선택된 행동의 로그 확률 추출
            target_action_log_probs = action_log_probs(target_logits, actions)
            behavior_action_log_probs = action_log_probs(behavior_logits, actions)
            
            
            # 로그 중요도 샘플링 비율 계산
            log_rhos = target_action_log_probs - behavior_action_log_probs
            # compute value estimates and logits for observed states
            # compute log probs for current and old policies # Importance sampling ratio 기록
            '''
            vs, pg_advantages, importance_sampling_ratios = vtrace.from_importance_weights( 
                log_rhos=log_rhos,
                discounts=discounts,
                rewards=rewards,
                values=target_values,
                bootstrap_value=bootstrap_value,
            )'''
            
            v = target_values.squeeze(1)
            rhos = torch.exp(log_rhos)
            clipped_rhos = torch.clamp(rhos, max=1.0)
            cs = torch.clamp(rhos, max=1.0)

            deltas = clipped_rhos * (rewards + discount_gamma * v[1:] - bootstrap_value)
            
            acc = torch.zeros(discounts.shape[0]+1,device=device)
            # result = []
            for t in range(discounts.shape[0] - 1, -1, -1):
                acc[t] = deltas[t] + discounts[t] * cs[t] * (acc[t+1]-v[t+1])
                # result.append(acc)
                
            vs= torch.add(acc, v)
            
            pg_advantages = rhos * (rewards + discounts * vs[1:] - bootstrap_value)

            importance_sampling_ratios += rhos
            
            # batch_size
            '''        
            # baseline_loss, Weighted MSELoss
            critic_loss = baseline_cost * compute_baseline_loss(vs-target_values)

            # policy gradient loss
            pg_loss = compute_policy_gradient_loss(target_logits,actions,pg_advantages)

            # entropy_loss
            entropy_loss = entropy_cost * compute_entropy_loss(target_logits)

            # total loss 
            loss = (pg_loss + critic_loss + entropy_loss)
            '''
            critic_loss = compute_baseline_loss(vs-target_values)
            policy_loss = compute_policy_gradient_loss(
                target_logits,actions,pg_advantages
            )
            policy_entropy = -1 * compute_entropy_loss(target_logits)
            loss = (
                baseline_cost * critic_loss
                + policy_loss
                - entropy_cost* policy_entropy
            )
            

            loss = torch.add(loss, loss / batch_size)
            
                    
            # Update Optimizaer
            optimizer.zero_grad()
            # check backward time
            start_backward_time = time.time()
            
            loss.backward()
            
            # loss.backward()
            backward_time = time.time() - start_backward_time

            # Omptimisation
            torch.nn.utils.clip_grad_norm_(
                learner_model.parameters(), args.global_gradient_norm
            )
        
            # update optimizaer
            optimizer.step()
            # schedualer update
            scheduler.step()

            # Tensorboard 용 
            total_loss = loss.item() 
            value_loss += critic_loss.item() / batch_size
            policy_loss += policy_loss.item() / batch_size
            policy_entropy += policy_entropy.item() / batch_size

        
            # reward_mean += rewards.mean().item() / batch_size
            reward += rewards.sum().item() / batch_size

            # Save the trained model
            learner_model.cpu()
            # with sync_ps.lock:
            # sync_ps.push(learner_model.state_dict())

            if reward > best:
                sync_ps.push(learner_model.state_dict())
                torch.save(learner_model.state_dict(), args.save_path)
                # torch.save(
                # {"model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                # "scheduler_state_dict": scheduler.state_dict()}
                # ,save_path)
                best = reward

            traj_cnt += 1
            if args.verbose == 1:
                print(
                    f"[learner Updating model weights "
                    f" for Update {traj_cnt + 1}"
                )
            
        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss',{
            'total': total_loss,
            'critic': value_loss,
            'policy': policy_loss,
            'entropy': policy_entropy,
        }, step)

        # writer.add_histogram('Action',actions,step)

        writer.add_scalar('Learning_rate',scheduler.get_last_lr()[0], step )
        
        writer.add_scalars('Rewards', {
            # 'mean': reward_mean,
            'sum': reward
        }, step)
        
        writer.add_scalars('Importance_sampling_ratio', {
            'min': importance_sampling_ratios.min().item(),
            'max': importance_sampling_ratios.max().item(),
            'mean': importance_sampling_ratios.mean().item()
        }, step)
        
        writer.add_scalars('Time', {
            'batch': batch_time,
            'forward': forward_time,
            'backward': backward_time
        }, step)
        
        #log to console
        if args.verbose >= 2:
            print(                
                f"Batch Mean Reward: {reward:.2f} | Loss: {total_loss:.2f}"            
            )
        
        step += 1

        if torch.cuda.is_available():
            learner_model.cuda()
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            learner_model.to(device)

        while terminate_event.is_set():
            if args.verbose >= 1:
                print(f"Actor {idx} terminating.")
            break
        
        # batch.clear()
        
    print("Exiting actoer process.")
    env.close()
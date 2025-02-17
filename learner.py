"""Learner with parameter server"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import vtrace

import time

from utils import *

from random import shuffle

from torch.optim.lr_scheduler import PolynomialLR, LambdaLR

from torch.utils.tensorboard import SummaryWriter

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(logits, dim=-1), target=torch.flatten(actions), reduction="none",
    ).view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=torch.flatten(actions),
        reduction="none",
    ).view_as(actions)
    
def learner(model, data, ps, args, terminate_event):

    """Learner to get trajectories from Actors."""
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.epsilon,
                              weight_decay=args.decay,
                              momentum=args.momentum)
    
    model.share_memory()
    
    # lr_lambda = lambda epoch : 1 - min(epoch * args.length * args.batch_size, args.total_steps) / args.total_steps
    # scheduler = LambdaLR(optimizer,lr_lambda)
    
    scheduler = PolynomialLR(optimizer, total_iters=args.total_steps, power=1.0)
    
    writer = SummaryWriter(log_dir=args.log_dir)
        
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    gamma = args.gamma
    # discount_gamma = args.gamma
    save_path = args.save_path
    
    total_steps = args.total_steps
    """Gets trajectories from actors and trains learner."""
    
    step = 0
    
    while not terminate_event.is_set():
    
        trajectory = data.get()
        
        if torch.cuda.is_available():
            device = torch.device("gpu")
            trajectory.cuda()
            model.cuda()
        elif torch.mps.is_available():
            trajectory.mps()
            device = torch.device("mps")
            model.to(device)
        else:
            device = torch.device("cpu")
        
        total_loss = 0.0 
        value_loss = 0.0 
        policy_loss = 0.0 
        policy_entropy = 0.0

        reward = 0.0
        importance_sampling_ratios = 0.0 
        learner_cnt = 0
        start_batch_time = time.time()
        
        while args.batch_size > learner_cnt:
            # behaviour_logits, obs, actions, rewards, dones, hx = make_time_major(batch)
            behaviour_logits, obs, actions, rewards, dones = trajectory.get(device)
            optimizer.zero_grad()
            
            if args.reward_clip == "abs_one":
                clipped_rewards = torch.clamp(rewards, -1, 1)
            elif args.reward_clip == 'soft_asymmetric':
                squeezed = torch.tanh(rewards / 5.0)
                # Negative rewards are given less weight than positive rewards.
                clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0
            else:
                clipped_rewards = rewards
                
            # check forward time 
            start_forward_time = time.time()
            target_logits, target_values = model.get_policy(obs)
            forward_time = time.time() - start_forward_time        
            
            bootstrap_value = target_values[-1]
            actions, behaviour_logits, dones, rewards = actions[1:], behaviour_logits[1:], dones[1:], clipped_rewards[1:]
            # actions, behaviour_logits, dones, rewards = actions[1:], behaviour_logits[1:], dones[1:], rewards[1:]
            target_logits, target_values = target_logits[:-1], target_values[:-1]                          
        
            
            # 선택된 행동의 로그 확률 추출
            # target_action_log_probs = action_log_probs(target_logits, actions)
            # behavior_action_log_probs = action_log_probs(behaviour_logits, actions)
            
            target_action_log_probs = vtrace.log_probs_from_logits_and_actions(target_logits, actions)
            behavior_action_log_probs = vtrace.log_probs_from_logits_and_actions(behaviour_logits, actions)
            
            # 로그 중요도 샘플링 비율 계산
            log_rhos = target_action_log_probs - behavior_action_log_probs
            
            '''
            discounts = (~dones).float() * gamma
            
            vs, clip_rhos = vtrace.from_importance_weights(
                log_rhos=log_rhos,
                discounts=discounts,
                rewards=rewards,
                values=target_values,
                bootstrap_value=bootstrap_value,
                )
            
            pg_advantages = clip_rhos * (rewards + discounts * vs - target_values)
            ''' 
            
            vs, pg_advantages, clip_rhos = vtrace.from_importance_weights(
                log_rhos=log_rhos,
                discounts=gamma,
                rewards=rewards,
                values=target_values,
                bootstrap_value=bootstrap_value,
                )
            
            
            # policy gradient loss
            pg_cross_entropy = F.cross_entropy(target_logits, actions, reduction='none')
            policy_loss = (pg_cross_entropy * pg_advantages.detach()).sum()
            # baseline_loss
            value_loss = baseline_cost * .5 * (vs - target_values).pow(2).sum()
            # entropy_loss
            entropy_loss = entropy_cost * -(-F.softmax(target_logits, 1) * F.log_softmax(target_logits, 1)).sum(-1).sum()
            
            loss = (
                value_loss
                + policy_loss
                + entropy_loss
            )

            # Tensorboard 용 
            total_loss += loss.item() / batch_size
            value_loss += value_loss.item() / batch_size
            policy_loss += policy_loss.item() / batch_size
            policy_entropy += entropy_loss.item() / batch_size
            
            '''
            # baseline_loss
            traj_value_fn_loss = compute_baseline_loss(vs - target_values)
            # policy gradient loss
            # traj_policy_loss = compute_policy_gradient_loss(
            #     target_logits, actions, pg_advantages
            # )
            pg_cross_entropy = F.cross_entropy(target_logits, actions, reduction='none')
            traj_policy_loss = (pg_cross_entropy * pg_advantages.detach()).sum()
            
            # entropy_loss
            traj_policy_entropy = -1 * compute_entropy_loss(target_logits)
            loss = (
                baseline_cost * traj_value_fn_loss
                + traj_policy_loss
                - entropy_cost * traj_policy_entropy
            )
            
            # Tensorboard 용 
            # loss = torch.add(loss, traj_loss / batch_size)
            total_loss += loss.item() / batch_size
            value_loss += traj_value_fn_loss.item() / batch_size
            policy_loss += traj_policy_loss.item() / batch_size
            policy_entropy += traj_policy_entropy.item() / batch_size
            '''
            
            reward += torch.sum(rewards).item() / batch_size
            importance_sampling_ratios += clip_rhos / batch_size
            # importance_sampling_ratios += importance_sampling_ratios / batch_size
            learner_cnt+=1
        
        batch_time = time.time() - start_batch_time
        
        # loss = torch.add(loss, loss / batch_size)
        # check backward time
        start_backward_time = time.time()
        loss.backward()
        # loss.backward()
        backward_time = time.time() - start_backward_time
        
        # Omptimisation
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), args.global_gradient_norm
        # )
        
        # update optimizaer
        optimizer.step()
        # schedualer update
        scheduler.step()
        
        model.cpu()
        with ps.lock:
            ps.push(model.state_dict()) 
            
        if (step % args.save_interval) == 0:           
            torch.save(model.state_dict(), save_path)
            # best = reward
            
        if torch.cuda.is_available():
            model.cuda()
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
                    
        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss',{
            'total': total_loss,
            'value': value_loss ,
            'policy': policy_loss ,
            'entropy': policy_entropy ,
        }, step)

        # writer.add_histogram('Action',actions,step)

        # writer.add_scalar('Learning_rate',scheduler.get_last_lr()[0], step )
        
        writer.add_scalars('Rewards', {
            # 'mean': reward_mean,
            'sum': reward
        }, step)
        
        # importance_sampling_ratios = importance_sampling_ratios/batch_size
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
    
        if step >= total_steps:
            terminate_event.set()

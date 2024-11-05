import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import PolynomialLR

import vtrace

import time
from torch.utils.tensorboard import SummaryWriter


'''
    # Target Policy : πρ¯
    Target policy는 학습자(Learner)가 최적화하려는 정책. 
    이는 에이전트가 궁극적으로 따르기를 원하는 정책으로, 학습 과정에서 지속적으로 업데이트 됨. 
    Target policy는 학습자가 수집한 데이터를 기반으로 가치 함수와 정책을 업데이트하는 데 사용. 
    :: 이 정책은 주로 학습자의 신경망 파라미터로 표현됩니다

    # Behaviour Policy : µ
    행동 정책은 에이전트가 환경에서 어떤 행동을 선택할지를 결정하는 정책. 
    IMPALA에서는 여러 개의 액터(actor)가 환경과 상호작용하며 데이터를 수집. 
    이 액터들은 행동 정책을 따르며, 이 정책은 학습자(learner)에 의해 주기적으로 업데이트 됨. 
    행동 정책은 주로 탐험(exploration)과 활용(exploitation) 사이의 균형을 맞추기 위해 설계.

    # Local Policy
    Local policy는 각 액터(actor)가 환경과 상호작용할 때 사용하는 정책. 
    IMPALA에서는 여러 액터가 병렬로 환경과 상호작용하며 데이터를 수집. 
    ㄴ 이때 각 액터는 자신의 로컬 정책(local policy)을 따릅니다. 
    로컬 정책은 주기적으로 학습자의 타겟 정책(Target Policy)으로부터 업데이트되지만, 항상 최신 상태는 아닐 수 있음. 
    :: 이는 분산 학습에서 발생하는 지연(latency) 때문 - Policy lag

    # Value Function : v trace
    가치 함수는 특정 상태에서의 기대 보상을 추정하는 함수. 
    IMPALA에서는 V-trace라는 오프-폴리시(off-policy) 보정 방법을 사용하여 가치 함수를 추정(Approximation). 
    V-trace는 액터들이 수집한 데이터를 학습자가 효과적으로 사용할 수 있도록 함. 
    :: 이를 통해 학습자는 더 안정적이고 효율적으로 학습할 수 있음.

## Etc

    Single Task 
    ㄴ Hyper parameter combination 을 통해 학습가능함
    ㄴ method : cliping reward

    # 수식 - DeepMind Lab 환경
    def optimistic_asymmetric_clipping(reward):
        reward_tanh = torch.tanh(reward)
        clipped_reward = 0.3 * torch.min(reward_tanh, torch.tensor(0.0)) + 5.0 * torch.max(reward_tanh, torch.tensor(0.0))
        return clipped_reward
'''

def transpose_batch_to_stack(batch):
    obs = []
    actions = []
    rewards = []
    dones = []
    hidden_state = []
    logits = []
    for t in batch:
        obs.append(t.obs)
        rewards.append(t.rewards)
        dones.append(t.dones)
        actions.append(t.actions)
        logits.append(t.logit)
        hidden_state.append(t.lstm_hidden_state)
    obs = torch.stack(obs).transpose(0, 1)
    actions = torch.stack(actions).transpose(0, 1)
    rewards = torch.stack(rewards).transpose(0, 1)
    dones = torch.stack(dones).transpose(0, 1)
    logits = torch.stack(logits).permute(1, 2, 0)
    hidden_state = torch.stack(hidden_state).transpose(0, 1)
    return logits, obs, actions, rewards, dones, hidden_state

def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return 0.5 * torch.sum(torch.square(advantages))

def compute_entropy_loss(logits):
    # policy = F.softmax(logits, dim=-1)
    # log_policy = F.log_softmax(logits, dim=-1)
    # entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    # return -torch.sum(entropy_per_timestep)
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return -torch.sum(policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.cross_entropy(logits, actions, reduction='none')
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach() 
    return torch.sum(policy_gradient_loss_per_timestep)

def learner(model, experience_queue, sync_ps, args, terminate_event):
    """Learner to get parameters from IMPALA"""
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.epsilon,
    #                           weight_decay=args.decay,
    #                           momentum=args.momentum)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon,
                       weight_decay=args.decay)

    scheduler = PolynomialLR(optimizer, total_iters=args.total_steps, power=1.0)
    model.share_memory()
    
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    gamma = args.gamma
    save_path = args.save_path
    total_steps = args.total_steps

    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.log_dir)

    
    batch = []
    best = 0.
    step = 0
    
    while not terminate_event.is_set():
        """Gets trajectory from experience of actors and trains learner."""
        
        total_loss = 0.0 
        policy_loss = 0.0 
        entropy_loss = 0.0 
        reward_mean = 0.0 
        reward_sum = 0.0 
        
        # check batch time
        start_batch_time = time.time()
        # Dequeue trajectory data( all of state )
        trajectory = experience_queue.get()
        batch.append(trajectory)
        if torch.cuda.is_available():
            trajectory.cuda()
            model.cuda()
        elif torch.mps.is_available():
            trajectory.mps()
            device = torch.device("mps")
            model.to(device)

        if len(batch) < batch_size:
            continue

        # print(f"state : {trajectory}"
        behaviour_logits, obs, actions, rewards, dones, hidden_state = transpose_batch_to_stack(batch)
        batch_time = time.time() - start_batch_time

        if args.reward_clip == 'abs_one':
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif args.reward_clip == 'soft_asymmetric':
            squeezed = torch.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0
        else:
            clipped_rewards = rewards
        
        optimizer.zero_grad()
        
        # check forward time 
        start_forward_time = time.time()
        logits, values = model(obs, actions, clipped_rewards, dones, hidden_state=hidden_state)
        forward_time = time.time() - start_forward_time

        bootstrap_value = values[-1]
        actions, behaviour_logits, dones, clipped_rewards = actions[1:], behaviour_logits[1:], dones[1:], clipped_rewards[1:]
        logits, values = logits[:-1], values[:-1]

        discounts = (~dones).float() * gamma
        
        # compute value estimates and logits for observed states
        # compute log probs for current and old policies
        target_log_probs = vtrace.log_probs_from_logits_and_actions(logits, actions)
        behaviour_log_probs = vtrace.log_probs_from_logits_and_actions(behaviour_logits, actions)
        log_rhos = target_log_probs - behaviour_log_probs
        vs, pg_advantages = vtrace.from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=clipped_rewards,
            values=values,
            bootstrap_value=bootstrap_value)
        
        # vs, pg_advantages = vtrace.from_logits(
        #     behaviour_policy_logits=behaviour_logits,
        #     target_policy_logits=logits,
        #     actions=actions,
        #     discounts=discounts,
        #     rewards=clipped_rewards,
        #     values=values,
        #     bootstrap_value=bootstrap_value)
   
        
        # baseline_loss, Weighted MSELoss
        advantages = vs-values
        critic_loss = compute_baseline_loss(advantages)
        loss = baseline_cost * critic_loss

        # policy gradient loss
        loss += compute_policy_gradient_loss(logits,actions,pg_advantages)

        # entropy_loss
        entropy = entropy_cost * compute_entropy_loss(logits)
        loss += entropy
        # loss -= entropy

        # check backward time
        start_backward_time = time.time()
        loss.backward()
        backward_time = time.time() - start_backward_time

        # omptimisation
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.global_gradient_norm
        )
       
        # update optimizaer
        optimizer.step()
        
        
        # scheduler update
        scheduler.step()
        model.cpu()
        
        sync_ps.push(model.state_dict())
        if clipped_rewards.mean().item() > best:
            torch.save(model.state_dict(), save_path)
        

        total_loss += loss.item() / batch_size
        policy_loss += critic_loss.item() / batch_size
        entropy_loss += entropy.item() / batch_size
        reward_mean += clipped_rewards.mean().item() / batch_size
        reward_sum += clipped_rewards.sum().item() / batch_size

        # Importance sampling ratio 기록
        importance_sampling_ratios = torch.exp(log_rhos)
        
        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss',{
            'total': total_loss,
            'critic': policy_loss,
            'entropy': entropy_loss,
        }, step)

        writer.add_histogram('Action',actions,step)

        writer.add_scalar('Learning_rate',scheduler.get_last_lr()[0], step )
        writer.add_scalars('Rewards', {
            'mean': reward_mean,
            'sum': reward_sum
        }, step)
        
        writer.add_scalars('Importance_sampling_ratio', {
            'min': importance_sampling_ratios.min().item(),
            'max': importance_sampling_ratios.max().item(),
            'avg': importance_sampling_ratios.mean().item()
        }, step)
        
        writer.add_scalars('Time', {
            'batch': batch_time,
            'forward': forward_time,
            'backward': backward_time
        }, step)
        
        # log to console
        if args.verbose >= 1:
            # reward_mean = clipped_rewards.mean()
            print( f"Batch Mean Reward: {reward_mean:.2f} | Loss: {total_loss:.2f}" )
            
        step += 1

        if torch.cuda.is_available():
            model.cuda()
        
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
        
        batch = []

        if step >= total_steps:
            terminate_event.set()
            
    print("Exiting leraner process.")
    writer.close()

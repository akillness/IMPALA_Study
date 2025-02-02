import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import PolynomialLR, LambdaLR

import vtrace

import time,os
from torch.utils.tensorboard import SummaryWriter


'''
    # Target Policy : πρ¯ -- Behaviour Policy를 통해 나온 최신 policy
    Target policy는 학습자(Learner)가 최적화하려는 정책. 
    이는 에이전트가 궁극적으로 따르기를 원하는 정책으로, 학습 과정에서 지속적으로 업데이트 됨. 
    Target policy는 학습자가 수집한 데이터를 기반으로 가치 함수와 정책을 업데이트하는 데 사용. 
    :: 이 정책은 주로 학습자의 신경망 파라미터로 표현됩니다

    # Behaviour Policy : µ
    행동 정책은 에이전트가 환경에서 어떤 행동을 선택할지를 결정하는 정책. 수집된 과거의 policy 
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
    state = []
    actions = []
    rewards = []
    dones = []
    core_state = []
    logits = []

    length = 0
    batch_size= len(batch)
    for t in batch:
        state.append(t.state)
        rewards.append(t.rewards)
        dones.append(t.dones)
        actions.append(t.actions)
        logits.append(t.logit)
        core_state.append(t.lstm_core_state)
        length = t.length


    state = torch.stack(state).transpose(0, 1)
    actions = torch.stack(actions).transpose(0, 1)
    rewards = torch.stack(rewards).transpose(0, 1)
    dones = torch.stack(dones).transpose(0, 1)
    logits = torch.stack(logits).permute(1, 0, 2)
    core_state = torch.stack(core_state).transpose(0, 1)

    # state = torch.stack(state).view(length*batch_size,-1)
    # actions = torch.stack(actions).view(length*batch_size,-1)
    # rewards = torch.stack(rewards).view(length*batch_size,-1)
    # dones = torch.stack(dones).view(length*batch_size,-1)
    # logits = torch.stack(logits).view(length*batch_size,-1)
    # core_state = torch.stack(core_state).transpose(0, 1)
    

    return logits, state, actions, rewards, dones, core_state

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return -torch.sum(-policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.cross_entropy(logits, actions, reduction='none')# sparse_softmax_cross_entropy_with_logits
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach() 
    return -torch.sum(policy_gradient_loss_per_timestep)
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(logits, dim=-1), target=torch.flatten(actions), reduction="none",
    # ).view_as(advantages)
    # return torch.sum(cross_entropy * advantages.detach())
    
def learner(model, experience_queue, sync_ps, args, terminate_event):
    """Learner to get parameters from IMPALA"""
    
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    discount_gamma = args.gamma
    save_path = args.save_path
    total_steps = args.total_steps

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon,
    #                    weight_decay=args.decay)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        eps=args.epsilon,
        alpha=args.decay,
    )

    lr_lambda = lambda epoch : 1 - min(epoch * args.length * args.batch_size, args.total_steps) / args.total_steps
    scheduler = LambdaLR(optimizer,lr_lambda)
    
    # scheduler = PolynomialLR(optimizer, total_iters=args.total_steps, power=1.0)
    '''
    # Load state from a checkpoint, if possible.
    if os.path.exists(save_path):        
        checkpoint_states = torch.load(
            save_path, map_location= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), weights_only=True
        )
        model.cpu()
        model.load_state_dict(checkpoint_states)
        # model.load_state_dict(checkpoint_states["model_state_dict"])
        # optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
    '''

    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.log_dir)
    
    batch = []
    step = 0
    best = 0
    
    while not terminate_event.is_set():
            
        """Gets trajectory from experience of actors and trains learner."""
        
        loss = 0.0 
        value_loss = 0.0 
        policy_loss = 0.0 
        policy_entropy = 0.0

        reward = 0.0
        reward_mean = 0.0 
        
        # check batch time
        start_batch_time = time.time()
        # Dequeue trajectory data( all of state )
        trajectory = experience_queue.get()
        
        # print(f"rewards : {trajectory.rewards.sum()}")
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

        # sync_ps.lock.acquire()  # Only one thread learning at a time.

        behavior_logits, state, actions, rewards, dones, core_state = transpose_batch_to_stack(batch)
        # print(f"rewards : {rewards.sum().item()}")
        batch_time = time.time() - start_batch_time
        
        # print(f"Trajectories : {len(state)}")
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
        reshape, target_logits, target_values = model(state, actions, clipped_rewards, dones, core_state=core_state)
        
        '''
        # reshape for vtrace
        '''
        behavior_logits = behavior_logits.reshape(reshape[0]*reshape[1],reshape[2])
        actions = reshape[3].squeeze()
        clipped_rewards = reshape[4]
        dones = reshape[5]
        
        forward_time = time.time() - start_forward_time        
        actions, behavior_logits, dones, rewards = actions[1:], behavior_logits[1:], dones[1:], clipped_rewards[1:]
        bootstrap_value = target_values[-1]
        target_logits, target_values = target_logits[:-1], target_values[:-1]

        discounts = (~dones).float() * discount_gamma
        
        # 행동 정책과 목표 정책의 로그 확률 계산
        # 선택된 행동의 로그 확률 추출
        target_action_log_probs = vtrace.action_log_probs(target_logits, actions)
        behavior_action_log_probs = vtrace.action_log_probs(behavior_logits, actions)
        
        
        # 로그 중요도 샘플링 비율 계산
        log_rhos = target_action_log_probs - behavior_action_log_probs
        # compute value estimates and logits for observed states
        # compute log probs for current and old policies # Importance sampling ratio 기록
        vs, pg_advantages, importance_sampling_ratios = vtrace.from_importance_weights( 
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=target_values,
            bootstrap_value=bootstrap_value,
        )

        # vs, pg_advantages = vtrace.from_logits(
        #     behavior_policy_logits=behavior_logits,
        #     target_policy_logits=logits,
        #     actions=actions,
        #     discounts=discounts,
        #     rewards=clipped_rewards,
        #     values=values,
        #     bootstrap_value=bootstrap_value)
        
        # batch_size

        # baseline_loss, Weighted MSELoss
        critic_loss = baseline_cost * compute_baseline_loss(vs-target_values)

        # policy gradient loss
        pg_loss = compute_policy_gradient_loss(target_logits,actions,pg_advantages)

        # entropy_loss
        entropy_loss = entropy_cost * compute_entropy_loss(target_logits)

        # total loss 
        loss = (pg_loss + critic_loss + entropy_loss)

        # Update Optimizaer
        optimizer.zero_grad()
        # check backward time
        start_backward_time = time.time()
        loss.backward()
        # loss.backward()
        backward_time = time.time() - start_backward_time

        # Omptimisation
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.global_gradient_norm
        )
    
        # update optimizaer
        optimizer.step()
        # schedualer update
        scheduler.step()

        # Tensorboard 용 
        loss += loss.item() / batch_size
        value_loss += critic_loss.item() / batch_size
        policy_loss += pg_loss.item() / batch_size
        policy_entropy += entropy_loss.item() / batch_size

        reward_mean += rewards.mean().item() / batch_size
        reward += rewards.sum().item() / batch_size

        # Save the trained model
        model.cpu()
        sync_ps.push(model.state_dict())        
        if reward > best:
            torch.save(model.state_dict(), save_path)
            # torch.save(
            # {"model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict()}
            # ,save_path)
            best = reward

        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss',{
            'total': loss,
            'critic': value_loss,
            'policy': policy_loss,
            'entropy': policy_entropy,
        }, step)

        writer.add_histogram('Action',actions,step)

        writer.add_scalar('Learning_rate',scheduler.get_last_lr()[0], step )
        
        writer.add_scalars('Rewards', {
            'mean': reward_mean,
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
        if args.verbose >= 1:
            print(f"Step: {step} , Batch Mean Clip Reward: {reward_mean:.2f} , Loss: {loss:.6f}")
        
        step += 1

        if torch.cuda.is_available():
            model.cuda()
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
        
        batch.clear()
        if step >= total_steps:
            terminate_event.set()
        # sync_ps.lock.release() 
        
    print("Exiting leraner process.")
    writer.close()

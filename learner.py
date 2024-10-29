import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import make_time_major
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

def learner(model, experience_queue, sync_ps, args):
    """Learner to get trajectories from Actors."""
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.epsilon,
                              weight_decay=args.decay,
                              momentum=args.momentum)
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    gamma = args.gamma
    save_path = args.save_path

    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.log_dir)

    """Gets trajectories from actors and trains learner."""
    batch = []
    best = 0.
    step = 0
    while True:
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
        behaviour_logits, obs, actions, rewards, dones, hidden_state = make_time_major(batch)
        batch_time = time.time() - start_batch_time

        optimizer.zero_grad()
        # check forward time 
        start_forward_time = time.time()
        logits, values = model(obs, actions, rewards, dones, hidden_state=hidden_state)
        forward_time = time.time() - start_forward_time

        bootstrap_value = values[-1]
        actions, behaviour_logits, dones, rewards = actions[1:], behaviour_logits[1:], dones[1:], rewards[1:]
        logits, values = logits[:-1], values[:-1]
        discounts = (~dones).float() * gamma
        vs, pg_advantages = vtrace.from_logits(
            behaviour_policy_logits=behaviour_logits,
            target_policy_logits=logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value)
        
        # policy gradient loss
        cross_entropy = F.cross_entropy(logits, actions, reduction='none')
        loss = (cross_entropy * pg_advantages.detach()).sum()
        # baseline_loss, Weighted MSELoss
        critic_loss = baseline_cost * .5 * (vs - values).pow(2).sum()
        loss += critic_loss
        # entropy_loss
        entropy = -(-F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(-1).sum()
        loss += entropy_cost * entropy

        # check backward time
        start_backward_time = time.time()
        loss.backward()
        backward_time = time.time() - start_backward_time

        optimizer.step()
        model.cpu()
        
        sync_ps.push(model.state_dict())
        if rewards.mean().item() > best:
            torch.save(model.state_dict(), save_path)
        
        # Importance sampling ratio 기록
        importance_sampling_ratios = torch.exp(logits - behaviour_logits)

        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss', {
            'total': loss.item(),
            # 'cross_entropy': cross_entropy.mean().item(),
            'critic': critic_loss.item()
        }, step)
        
        writer.add_scalar('Entropy', entropy.item(), step)
        
        writer.add_scalars('Rewards', {
            'mean': rewards.mean().item(),
            'sum': rewards.sum().item()
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

        step += 1

        if torch.cuda.is_available():
            model.cuda()
        
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
        
        batch = []

    writer.close()

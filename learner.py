import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import make_time_major
import vtrace

import time
from torch.utils.tensorboard import SummaryWriter

def learner(model, data, ps, args):
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
        trajectory = data.get()
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
        behaviour_logits, obs, actions, rewards, dones, hx = make_time_major(batch)
        batch_time = time.time() - start_batch_time

        optimizer.zero_grad()
        # check forward time 
        start_forward_time = time.time()
        logits, values = model(obs, actions, rewards, dones, hx=hx)
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
        # baseline_loss
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
        
        ps.push(model.state_dict())
        if rewards.mean().item() > best:
            torch.save(model.state_dict(), save_path)
        
        # Importance sampling ratio 기록
        importance_sampling_ratios = torch.exp(logits - behaviour_logits)

        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss', {
            'total': loss.item(),
            'cross_entropy': cross_entropy.mean().item(),
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
        # Check if MPS is available and move the model to MPS
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
        
        batch = []

    writer.close()

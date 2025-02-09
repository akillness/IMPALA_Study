
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import PolynomialLR, LambdaLR

import vtrace

import time

from torch.utils.tensorboard import SummaryWriter

from impala import IMPALA

from environment import CartPole,get_action_size

def transpose_batch_to_stack(batch):
    state = []
    actions = []
    rewards = []
    dones = []
    logits = []

    for t in batch:
        state.append(t.state)
        rewards.append(t.rewards)
        dones.append(t.dones)
        actions.append(t.actions)
        logits.append(t.logit)

    state = torch.stack(state).transpose(0, 1)
    actions = torch.stack(actions).transpose(0, 1)
    rewards = torch.stack(rewards).transpose(0, 1)
    dones = torch.stack(dones).transpose(0, 1)
    logits = torch.stack(logits).permute(1, 0, 2)
    return logits, state, actions, rewards, dones 

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return -torch.sum(-policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.cross_entropy(logits, actions, reduction='none')# sparse_softmax_cross_entropy_with_logits
    return -torch.sum(cross_entropy * advantages.detach())
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(logits, dim=-1), target=torch.flatten(actions), reduction="none",
    # ).view_as(advantages)
    # return torch.sum(cross_entropy * advantages.detach())

class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self,max_size):
        self.state = []
        # self.next_state = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []
        self.actor_id = None
        # self.lstm_core_state = None
        self.max_size = max_size
        self.cur_size = 0
    
    def clear(self):
        self.state = []
        # self.next_state = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []

    def append(self, state, action, reward, done, logit):
        if self.max_size <= len(self.state):
            self.state.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.logit.pop(0)
            self.dones.pop(0)

        self.state.append(state)
        # self.next_state.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logit.append(logit)
        self.dones.append(done)

    def finish(self,GAMMA):
        self.state = torch.stack(self.state).squeeze()
        # 역방향으로 감쇄 적용
        step_reward = 0.
        for i in range(len(self.rewards) - 1, -1, -1):
            if self.rewards[i] is not None:
                step_reward *= GAMMA
                step_reward += self.rewards[i]
                self.rewards[i] = step_reward

        self.rewards = torch.cat(self.rewards, 0).squeeze()
        self.actions = torch.cat(self.actions, 0).squeeze()
        self.dones = torch.cat(self.dones, 0).squeeze()
        self.logit = torch.stack(self.logit).squeeze()
        return step_reward

    def cuda(self):
        self.state = self.state.cuda()
        # self.next_state = self.next_state.cuda()
        self.actions = self.actions.cuda()
        self.dones = self.dones.cuda()
        # self.lstm_core_state = self.lstm_core_state.cuda()
        self.rewards = self.rewards.cuda()
        self.logit = self.logit.cuda()

    def mps(self):
        device = torch.device("mps")
        self.state = self.state.to(device)
        # self.next_state = self.next_state.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        # self.lstm_core_state = self.lstm_core_state.to(device)
        self.rewards = self.rewards.to(device)
        self.logit = self.logit.to(device)

    def get_last(self):
        """must call this function before finish()"""
        state = self.state[-1]
        # next_state = self.next_state[-1]
        logits = self.logit[-1]
        last_action = self.actions[-1]
        reward = self.rewards[-1]
        done = self.dones[-1]
        return state, last_action, reward, done, logits
    
    def get(self):
        return self.state, self.actions, self.rewards, self.dones, self.logit

    @property
    def length(self):
        return len(self.rewards)

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
    
    # """Run the env for n steps and return a trajectory rollout."""
    obs = env.reset()
    
    total_reward = 0.

    while not terminate_event.is_set():
        # Sync actor model's wieghts
        # with sync_ps.lock:
        # actor_model.load_state_dict(learner_model.state_dict())
        actor_model.load_state_dict(sync_ps.pull())

        rollout = Trajectory(max_size=length)
        rollout.actor_id = idx
        # rollout.lstm_core_state = core_state.squeeze()
        # rollout.append(*persistent_state)
        
        with torch.no_grad():
            
            # steps = 0
            is_done = False            
            while True:
                if is_done:
                    
                    obs = env.reset()
                    # persistent_state = rollout.get_last()
                    total_reward = rollout.finish(args.gamma)
                    # rollout.finish()
                    if args.verbose >= 1:
                        print(f"Actor:{idx}, Total Reward :{total_reward}")
                    total_reward = 0.
                    break

                    # print(f'{experience_queue._buffer}, {rollout.length}')

                action, logits = actor_model.get_policy_and_action(obs)
                        
                obs, reward, is_done = env.step(action)
        
                # total_reward += reward
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
                done = torch.tensor(is_done, dtype=torch.bool).view(1, 1)

                rollout.append(obs, last_action, reward, done, logits.detach())
                
        total_loss = 0.0 
        value_loss = 0.0 
        policy_loss = 0.0 
        policy_entropy = 0.0

        reward = 0.0
        reward_mean = 0.0 
        
        if torch.cuda.is_available():
            rollout.cuda()
            learner_model.cuda()
        elif torch.mps.is_available():
            rollout.mps()
            device = torch.device("mps")
            learner_model.to(device)

        # check batch time
        start_batch_time = time.time()
        batch_time = time.time() - start_batch_time
        
        state, actions, rewards, dones, behavior_logits = rollout.get()
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
        # reshape, target_logits, target_values = learner_model(state, actions, clipped_rewards, dones)
        
        target_logits, target_values = learner_model.get_policy(state)
        '''
        # reshape for vtrace
        '''
        
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
            learner_model.parameters(), args.global_gradient_norm
        )
    
        # update optimizaer
        optimizer.step()
        # schedualer update
        scheduler.step()

        # Tensorboard 용 
        total_loss = loss.item() 
        value_loss = critic_loss.item() 
        policy_loss = pg_loss.item()
        policy_entropy = entropy_loss.item()

        reward_mean = rewards.mean().item()
        reward = rewards.sum().item()

        # Save the trained model
        learner_model.cpu()
        # with sync_ps.lock:
        sync_ps.push(learner_model.state_dict())

        if reward > best:
            torch.save(learner_model.state_dict(), args.save_path)
            # torch.save(
            # {"model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict()}
            # ,save_path)
            best = reward

        # TensorBoard에 손실 및 보상 기록
        writer.add_scalars('Loss',{
            'total': total_loss,
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
        # if args.verbose >= 1:
        #     print(f"Step: {step} , Batch Mean Clip Reward: {reward_mean:.2f} , Loss: {loss:.6f}")
        
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
        
    print("Exiting actoer process.")
    env.close()
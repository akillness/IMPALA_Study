
import torch
# from model import IMPALA

from impala import IMPALA

from environment import Atari,CartPole,get_action_size

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

    @property
    def length(self):
        return len(self.rewards)

    def __repr__(self):
        return "ok" # End of episode when life lost "Yes"


# def actor(idx, experience_queue, sync_ps, env, args, terminate_event):
def actor(idx, experience_queue, sync_ps, args, terminate_event):
    
    total_steps = args.total_steps
    length = args.length
    action_size = args.action_size
    actor_model = IMPALA(action_size=action_size)
    # actor_model.share_memory()
    
    env = CartPole(game_name=args.game_name,seed=args.seed)
    
    # """Run the env for n steps and return a trajectory rollout."""
    state = env.reset()
    
    total_reward = 0.

    while not terminate_event.is_set():
        # Sync actor model's wieghts
        # with sync_ps.lock:
        actor_model.load_state_dict(sync_ps.pull())

        rollout = Trajectory(max_size=length)
        rollout.actor_id = idx
        # rollout.lstm_core_state = core_state.squeeze()
        # rollout.append(*persistent_state)
        
        with torch.no_grad():
            
            steps = 0
            is_done = False            
            while True:
                if is_done:
                    
                    state = env.reset()
                    # persistent_state = rollout.get_last()
                    total_reward = rollout.finish(args.gamma)
                    if args.verbose >= 1:
                        print(f"Actor:{idx}, Total Reward :{total_reward}")
                    experience_queue.put(rollout)
                    # total_reward = 0.
                    break

                    # print(f'{experience_queue._buffer}, {rollout.length}')

                action, logits = actor_model.get_policy_and_action(state)
                        
                state, reward, is_done = env.step(action)
                # total_reward += reward

                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
                done = torch.tensor(is_done, dtype=torch.bool).view(1, 1)

                rollout.append(state, last_action, reward, done, logits.detach())
                
                # state = next_state
                steps += 1
                while terminate_event.is_set():
                    if args.verbose >= 1:
                        print(f"Actor {idx} terminating.")
                    break
        
    print("Exiting actoer process.")
    env.close()
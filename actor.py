
import torch
from model import IMPALA

from environment import Atari,get_action_size

class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self,max_size):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []
        self.actor_id = None
        self.lstm_core_state = None
        self.max_size = max_size
        self.cur_size = 0
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []

    def append(self, state, action, reward, done, logit):

        # if self.max_size <= len(self.obs):
        #     self.obs.pop(0)
        #     self.actions.pop(0)
        #     self.rewards.pop(0)
        #     self.logit.pop(0)
        #     self.dones.pop(0)

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
        self.lstm_core_state = self.lstm_core_state.cuda()
        self.rewards = self.rewards.cuda()
        self.logit = self.logit.cuda()

    def mps(self):
        device = torch.device("mps")
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        self.lstm_core_state = self.lstm_core_state.to(device)
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


# def actor(idx, experience_queue, sync_ps, env, args, terminate_event):
def actor(idx, experience_queue, sync_ps, args, terminate_event):
    steps = 0
    total_steps = args.total_steps
    length = args.length
    action_size = args.action_size
    model = IMPALA(action_size=action_size)
    init_lstm_state = model.inital_state()
    
    env = Atari(game_name=args.game_name,seed=args.seed)
    
    # env = environment.Environment(gym_env)
    # """Run the env for n steps and return a trajectory rollout."""
    # gym_env.start()
    # obs = gym_env.reset()
    obs = env.reset()
    core_state = init_lstm_state
    logits = torch.zeros((1, action_size), dtype=torch.float32)
    last_action = torch.zeros((1, 1), dtype=torch.int64)
    reward = torch.tensor(0, dtype=torch.float32).view(1, 1)
    done = torch.tensor(False, dtype=torch.bool).view(1, 1)
    init_state = (obs, last_action, reward, done, logits)
    persistent_state = init_state
    # rewards = 0
    while not terminate_event.is_set():
        # Sync trained model
        with sync_ps.lock:
            model.load_state_dict(sync_ps.pull())

        # rollout.clear()
        rollout = Trajectory(max_size=length)
        rollout.actor_id = idx
        rollout.lstm_core_state = core_state.squeeze()
        # rollout.lstm_core_state = core_state
        rollout.append(*persistent_state)
        total_reward = 0
        with torch.no_grad():
            # while steps < total_steps:
            while True:
                if rollout.length == length + 1:
                    # rewards += total_reward
                    persistent_state = rollout.get_last()
                    rollout.finish()
                    experience_queue.put(rollout)
                    # if args.verbose >= 1:
                    #     print("Actor: {} rewards:{}".format(idx, total_reward))
                    break
                if done:
                    if steps < length-1:
                        # persistent_state = rollout.get_last()
                        rollout.clear()

                    total_reward = 0.
                    steps = 0
                    core_state = init_lstm_state
                    __, last_action, reward, done, _ = init_state
                    obs = env.reset()

                    
                # action, logits, hidden_state = model(obs.unsqueeze(0).unsqueeze(1), last_action, reward, done, hidden_state, actor=True)
                action, logits, core_state = model(obs.unsqueeze(0), last_action, reward, done, core_state, actor=True)
                          
                obs, reward, done = env.step(action)
                total_reward += reward

                # logits = torch.tensor(agent_output['policy_logits'], dtype=torch.float32).view(-1, 1)                
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(total_reward, dtype=torch.float32).view(1, 1)
                done = torch.tensor(done, dtype=torch.bool).view(1, 1)

                rollout.append(obs, last_action, reward, done, logits.detach())
                steps += 1

                while terminate_event.is_set():
                    if args.verbose >= 1:
                        print(f"Actor {idx} terminating.")
                    break
            
            # if rollout.length == length:
            #     # rewards += total_reward
            #     persistent_state = rollout.get_last()
            #     rollout.finish()
            #     # Queue trajectory data( all of state )
            #     experience_queue.put(rollout)              
            #     # print(f"Actor : {idx}, Total Reward : {total_reward}")
            #     # break
        
    print("Exiting actoer process.")
    env.close()
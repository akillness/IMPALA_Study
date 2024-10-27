import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gym

# 구조설계
# Actor - Learner Archietecture
# ㄴ Actor로 부터 Tracjactory를 얻고, T의 값이 State 이기 때문에 Policy-Lag에 강건
# ㄴ 추가) LSTM 으로 reward, action concat 해서 state로 바꾸기 - 원본코드비교
# V-Trace 를 이용해 on-policy 를 n-step bellman 업데이트로 축소
# ㄴ IS(Importance sampling) 하여 off-policy 데이터를 on-policy 데이터로 사용


# Define the Actor-Critic Network # 끝
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
    # def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    # def forward(self, x):
    #     x = torch.relu(self.fc(x))
    #     policy_dist = Categorical(logits=self.policy(x))
    #     value = self.value(x)
    #     return policy_dist, value
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        value = self.critic(x)
        logits = self.actor(x)
        return logits, value

class IMPALA(nn.Module):
    def __init__(self, seq_len, batch_size, action_size, device, learning_rate,hidden_size=256):
        super(IMPALA, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 256)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.policy = ActorCritic( action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.fc = nn.Linear(3136, hidden_size)

    def initial_state(self, batch_size, hidden_size=256):
        # LSTMCell 초기 hidden state와 cell state 생성
        return (torch.zeros(batch_size, hidden_size),
                torch.zeros(batch_size, hidden_size))
    
    # lstm state를 policy
    def forward(self,state,action,reward,done,hidden_size=256):
        traj_len = state.shape[0]
        batch_size = state.shape[1]
        
        x = state.reshape(traj_len * batch_size, *state.shape[2:])
        x = x.view(x.shape[0], -1)
        x = self.fc(x) # lstm을 거친후 fully connected layer로 합쳐짐

        x = torch.cat((x, reward, action), dim=1)
        x = x.view(traj_len, batch_size, -1)

        init_core_state = torch.zeros((2, batch_size, hidden_size), dtype=torch.float32, device=self.device)
        lstm_out = []
        for s, d in zip(torch.unbind(x, 0), torch.unbind(done, 0)):
            hidden_x = torch.where(d.view(1, -1, 1), init_core_state, hidden_x)
            hidden_x = self.lstm(s, hidden_x.unbind(0))
            lstm_out.append(hidden_x[0])
            hidden_x = torch.stack(hidden_x, 0)
        input = torch.cat(lstm_out, 0)
        logits, values = self.policy(input, actor)

        # return logits,values
        return logits.view(traj_len, -1, batch_size), values.view(traj_len, batch_size)
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    '''
    def get_action(self, state):
        logits, _ = self.model(state)
        action_prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(action_prob, 1).item()
        return action

    def compute_loss(self, states, actions, rewards, next_states, dones):
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        target_values = rewards + gamma * next_values * (1 - dones)
        advantages = target_values - values

        policy_logits, _ = self.model(states)
        action_log_probs = torch.log_softmax(policy_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        policy_loss = -(selected_log_probs * advantages.detach()).mean()

        value_loss = advantages.pow(2).mean()
        return policy_loss + value_loss

    def train(self, states, actions, rewards, next_states, dones):
        self.optimizer.zero_grad()
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    '''

# Actor class to generate trajectories
class Actor:
    def __init__(self, env, input_dim, action_size, n_steps, device,hidden_size=512):
        self.env = env
        self.action_size = action_size
        self.n_steps = n_steps
        self.device = device
        self.trajectory = []
        self.policy = ActorCritic(action_size).to(device)
        self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 256)
        
        # self.model = IMPALA(action_size=action_size,device=device,learning_rate=)

        # 학습된 모델 로드 > Deep RL 모델
        # n step 이후 trajatory return (여기서 trajactory 만 return )
        # ㄴ n_step 은 trajactory length를 의미
        # lstm 의 hidden state가 반영된 action과 logits(policy)을 함께 반영

    def generate_trajectory(self):
        state, info_ = self.env.reset()
        trajectory = []
        # n_steps
        for _ in range(self.n_steps):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

            policy_p, value_q = self.policy(state) # Logit p, Value q
            policy_dist = Categorical(logits=policy_p)
            action = policy_dist.sample()

            next_state, reward, done, truncated_, info_ = self.env.step(action.item())
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                # next_state, info = env.reset()
                break

            # action, logits, hx = model(obs.unsqueeze(0), last_action, reward,
            #                                done, hx, actor=True)
            # obs, reward, done = env.step(action)
            # total_reward += reward
            # last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
            # reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
            # done = torch.tensor(done, dtype=torch.bool).view(1, 1)
            # rollout.append(obs, last_action, reward, done, logits.detach())
            # steps += 1
        return trajectory
    
    

# Learner class to update the policy
class Learner:
    def __init__(self, policy, device, lr=1e-3):
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.writer = SummaryWriter()

    def update_policy(self, trajectories, global_step):
        policy_loss_total = 0
        value_loss_total = 0
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            
            # states.unsqueeze(0) : seq_len, batch_size // key!!!

            policy_dists, values = self.policy(states)
            log_probs = policy_dists.log_prob(actions)
            advantages = rewards - values.squeeze()
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()

        # Log losses to TensorBoard
        self.writer.add_scalar('Loss/Policy', policy_loss_total / len(trajectories), global_step)
        self.writer.add_scalar('Loss/Value', value_loss_total / len(trajectories), global_step)

    

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    env = gym.make("CartPole-v1")
    
    # env.reset() 
    # ㄴ Cart Position, Cart Velocity, Pole Angle, Pole Angle Velocity

    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    batch_size = 32
    actor = Actor(env, action_size=action_dim,input_dim=input_dim, n_steps=5, device=device)
    learner = Learner(actor.policy, device)
    
    # env.reset()
    global_step = 0
    for episode in tqdm(range(1000)):
        # env.render()
        trajectory = actor.generate_trajectory()
        learner.update_policy([trajectory], global_step)
        global_step += 1

        # Save the model every 100 episodes
        if episode % 10000 == 0:
            learner.save_model(f"model_{episode}.pth")
    
    # env.close()

import torch


class SyncParameters(object):
    def __init__(self, lock):
        self.lock = lock
        self.weight = None

    def pull(self):
        with self.lock:
            return self.weight

    def push(self, weigth):
        with self.lock:
            self.weight = weigth


def make_time_major(batch):
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


def combine_time_batch(x, last_action, reward, actor=False):
    if actor:
        return 1, 1, x, last_action, reward
    seq_len = x.shape[0]
    bs = x.shape[1]
    x = x.reshape(seq_len * bs, *x.shape[2:])
    last_action = last_action.reshape(seq_len * bs, -1)
    reward = reward.reshape(seq_len * bs, 1)
    return seq_len, bs, x, last_action, reward

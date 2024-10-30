import torch

def transpose_batch(batch):
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

def reshape_stacked_state_dim(x, last_action, reward, actor=False):
    if actor:
        return 1, 1, x, last_action, reward
    stacked_state_len = x.shape[0]
    batch_size = x.shape[1]
    x = x.reshape(stacked_state_len * batch_size, *x.shape[2:])
    last_action = last_action.reshape(stacked_state_len * batch_size, -1)
    reward = reward.reshape(stacked_state_len * batch_size, 1)
    return stacked_state_len, batch_size, x, last_action, reward

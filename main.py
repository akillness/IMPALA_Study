
import torch

import impala

import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else "cpu")

def main():

    # Actor
    # ㄴtrajactory
    # ㄴlstm
    # Learner
    # ㄴenv
    # ㄴvtrace
    # Distribution
    # RL

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]



if __name__ == "__main__":
    main()
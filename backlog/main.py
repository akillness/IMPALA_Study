
from impala import IMPALA
from actor_modify import Actor
import gym

def main():

    # Actor
    # ㄴtrajactory 
    # ㄴlstm :: local policy
    # Learner
    # ㄴenv 
    # ㄴvtrace :: value function
    # Distribution RL

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    actor = Actor(env,n_steps=5,unroll=5)    
    # init_hx = torch.zeros((2, 1, 256), dtype=torch.float32)
    # state_dim = env.observation_space.shape[0]
    actor.generate_trajectory()
    actor.run()



if __name__ == "__main__":
    main()
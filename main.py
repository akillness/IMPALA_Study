
import argparse

from environment import Atari, EnvironmentProxy, get_action_size

from model import IMPALA
from actor import actor
from learner import learner

import torch.multiprocessing as mp
from utils import ParameterServer

# IMPALA : Importance Weighted Actor-Learner Architecture ( vs A3C )
# - Actor
    # ㄴlstm : 2D-> 1D and batch dimesion reshape, local policy
    # ㄴoutput : trajactory 
# - Learner
    # ㄴvtrace : value function, on-policy 처럼 사용하여 target policy 학습, behavior policy update
# Environment ( Game )
    # ㄴstacking history-4, clipping reward
# Distribution RL

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=5,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=123,
                        help="the seed of random, default is 123")
    parser.add_argument("--game_name", type=str, default='breakout',
                        help="the name of atari game, default is breakout")
    parser.add_argument('--length', type=int, default=20,
                        help='Number of steps to run the agent')
    parser.add_argument('--total_steps', type=int, default=80000000,
                        help='Number of steps to run the agent')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of steps to run the agent')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--entropy_cost", type=float, default=0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
    parser.add_argument("--lr", type=float, default=0.0006,
                        help="Learning rate, default is 0.0006")
    parser.add_argument("--decay", type=float, default=.99,
                        help="RMSProp optimizer decay, default is .99")
    parser.add_argument("--momentum", type=float, default=0,
                        help="RMSProp momentum, default is 0")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="RMSProp epsilon, default is 0.1")
    parser.add_argument('--save_path', type=str, default="./model/checkpoint.pt",
                        help='Set the path to save trained model parameters')
    parser.add_argument('--load_path', type=str, default="./model/checkpoint.pt",
                        help='Set the path to load trained model parameters')
    parser.add_argument('--log_dir', type=str, default="./runs/",
                        help='Set the path to check learning state using tensorboard')    
    parser.add_argument('--reward_clip', type=str, default="tanh",
                        help='Set clipping reward type, default is "abs_one" (tanh,abs_one,no_clip)')

    # global gradient norm : 40
    # reward clipping : [-1, 1]
    # 
    args = parser.parse_args()
    data = mp.Queue(maxsize=1)
    lock = mp.Lock()
    
    
    env_args = {'game_name': args.game_name, 'seed': args.seed, 'reward_clip': args.reward_clip}
    action_size = get_action_size(Atari, env_args)
    args.action_size = action_size    
    

    # env_name = 'CartPole-v1'
    # args.game_name = env_name
    # env_args = {'game_name': args.game_name, 'seed': args.seed}
    # action_size = get_action_size(CartPole, env_args)
    # args.action_size = action_size
    # env = EnvironmentProxy(CartPole,env_args)
    # actor(0,ps,data,env,args)
    # learner(model,data,ps,args)

    # optional : using 4-actor process 
    ps = ParameterServer(lock)
    model = IMPALA(action_size=args.action_size)
    ps.push(model.state_dict())
    
    # env = EnvironmentProxy(Atari,env_args)
    # actor(0,ps,data,env,args)
    # learner(model,data,ps,args)

    # env = EnvironmentProxy(CartPole,env_args)
    # actor(0,ps,data,env,args,hidden_size)
    # learner(model,data,ps,args)
    
    # env
    envs = [EnvironmentProxy(Atari, env_args)
            for idx in range(args.actors)]
    # learner
    learner = mp.Process(target=learner, args=(model, data, ps, args))

    # actor
    actors = [mp.Process(target=actor, args=(idx, ps, data, envs[idx], args))
              for idx in range(args.actors)]
    
    # multi-process
    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()
    
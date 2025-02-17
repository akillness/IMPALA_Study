import torch
import argparse
import torch.multiprocessing as mp

import threading, os

import datetime

from model import Network
from learner import learner
from actor import actor
from environment import CartPole, EnvironmentProxy, get_action_size
from utils import ParameterServer

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=6,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=123,
                        help="the seed of random, default is 123")
    parser.add_argument("--game_name", type=str, default='CartPole-v1',
                        help="the name of atari game, default is breakout")
    parser.add_argument('--length', type=int, default=128,
                        help='Number of steps to run the agent')
    parser.add_argument('--total_steps', type=int, default=80000000,
                        help='Number of steps to run the agent')
    # parser.add_argument('--num_steps', type=int, default=100,
    #                   help='Number of Steps to learn')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of steps to run the agent')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--entropy_cost", type=float, default=0.00025,#0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
    parser.add_argument("--lr", type=float, default=0.00048,
                        help="Learning rate, default is 0.00048")
    parser.add_argument("--decay", type=float, default=.99,
                        help="RMSProp optimizer decay, default is .99")
    parser.add_argument("--momentum", type=float, default=0,
                        help="RMSProp momentum, default is 0")
    parser.add_argument("--epsilon", type=float, default=.1,
                        help="RMSProp epsilon, default is 0.1")
    parser.add_argument('--save_path', type=str, default="./checkpoint.pt",
                        help='Set the path to save trained model parameters')
    parser.add_argument('--load_path', type=str, default="./checkpoint.pt",
                        help='Set the path to load trained model parameters')
    parser.add_argument('--log_dir', type=str, default="./runs/",
                        help='Set the path to check learning state using tensorboard')    
    parser.add_argument('--reward_clip', type=str, default="abs_one",
                        help='Set clipping reward type, default is "abs_one" (soft_asymmetric,abs_one,no_clip)')
    parser.add_argument("--verbose", type=int, default=2,
                        help="RMSProp print log flag, default is 0")
    parser.add_argument("--global_gradient_norm", type=int, default=40,#40,
                        help="RMSProp gradient norm, default is 40")    
    parser.add_argument('--save_interval', type=int, default=100)
    
    args = parser.parse_args()
    
    terminate_event = mp.Event()
    
    start_time = datetime.datetime.now()

    # mp.set_start_method("fork", force=True)

    print(f"[main] Start time: {start_time:%d-%m-%Y %H:%M:%S}")
    
    data = mp.Queue(maxsize=args.actors)
    lock = mp.Lock()
    env_args = {'game_name': args.game_name, 'seed': args.seed}
    action_size = get_action_size(CartPole, env_args)
    args.action_size = action_size
    ps = ParameterServer(lock)
    
    global_network = Network(action_size=args.action_size)
    if os.path.exists(args.load_path):
        global_network.cpu()
        global_network.state_dict(torch.load(args.load_path, map_location= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), weights_only=True))
        
    ps.push(global_network.state_dict())

    # envs = [EnvironmentProxy(Atari, env_args)
    #         for idx in range(args.actors)]


    learner = mp.Process(target=learner, args=(global_network, data, ps, args, terminate_event))

    actors = [mp.Process(target=actor, args=(idx, ps, data, args, terminate_event))
              for idx in range(args.actors)]
    
    print("[main] Initialized")

    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()

    print(
        f"[main] Completed in {(datetime.datetime.now() - start_time).seconds} seconds"
    )
    

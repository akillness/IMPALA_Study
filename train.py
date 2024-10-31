
import argparse


from proxy import *
from environment import CartPole,get_action_size

from model import IMPALA
from actor import actor
from learner import learner

# import threading, queue
from concurrent.futures import ThreadPoolExecutor

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
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=2,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=20,
                        help="the seed of random, default is 20")
    parser.add_argument("--game_name", type=str, default='CartPole-v1',
                        help="the name of atari game, default is CartPole-v1")
    # parser.add_argument("--game_name", type=str, default='breakout',
    #                     help="the name of atari game, default is CartPole-v1")
    parser.add_argument('--length', type=int, default=20,
                        help='Number of steps to run the agent')
    parser.add_argument('--total_steps', type=int, default=80000000,
                        help='Number of steps to run the agent')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Number of steps to run the agent')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate, default is 0.001")
    parser.add_argument("--entropy_cost", type=float, default=0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
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
    
    args = parser.parse_args()
    env_args = {'game_name': args.game_name, 'seed': args.seed, 'reward_clip': args.reward_clip}
    action_size = get_action_size(CartPole, env_args)
    # action_size = get_action_size(Atari, env_args)
    args.action_size = action_size    

    '''
    # Standard single-process
    '''
    
    # env_name = 'CartPole-v1'   
    experience_queue = queue.Queue()
    lock = threading.Lock()
    envs = [EnvThread(CartPole, env_args)
            for idx in range(args.actors)]
    
    sync_ps = SyncParameters(lock)
    model = IMPALA(action_size=args.action_size)
    sync_ps.push(model.state_dict())

    idx=0
    actor(idx, experience_queue, sync_ps, envs[idx], args)

    learner(model, experience_queue, sync_ps, args)

    # max workers : actors + learner
    # with ThreadPoolExecutor(max_workers=args.actors + 1) as executor:
    #     # actors
    #     thread_pool = [executor.submit(actor, idx, experience_queue, sync_ps, envs[idx], args) for idx in range(args.actors)]
    #     # learner
    #     thread_pool.append(executor.submit(learner, model, experience_queue, sync_ps, args))
        
    #     # synchronous learning and actors
    #     for thread in thread_pool:
    #         thread.result()
    
    """
    # Optional Section

    from proxy import EnvProcess
    from environment import CartPole, get_action_size

    import torch.multiprocessing as mp

    mp.set_start_method('spawn')
    
    # optional : using 4-actor other process 
    args.actors = 4

    experience_queue = mp.Queue(maxsize=1)
    lock = mp.Lock()
    sync_ps = SyncParameters(lock)
    model = IMPALA(action_size=args.action_size)
    sync_ps.push(model.state_dict())
    
    # environments of multi-process paired actors
    envs = [EnvProcess(CartPole, env_args)
            for idx in range(args.actors)]
    # learner
    learner = mp.Process(target=learner, args=(model, experience_queue, sync_ps, args))

    # actors of multi-process pool
    actors = [mp.Process(target=actor, args=(idx, experience_queue, sync_ps, envs[idx], args))
              for idx in range(args.actors)]
    
    # synchronous learning and actors
    learner.start()
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]
    learner.join()
    """
    
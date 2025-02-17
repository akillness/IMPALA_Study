
import argparse

import torch 

from proxy import *
from environment import Atari,CartPole,get_action_size

# from model import IMPALA

from impala import IMPALA

from single_agent import actor
from learner import learner

import torch.multiprocessing as mp

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


"""
실행환경
ㄴ pip install -r requirements.txt

추가 ) https://pytorch.org/ 이 링크에서 패키지 설치 필수.
ㄴex) > pip3 install torch torchvision torchaudio

예외 )
 "Close error" 발생시 numpy 버전을 1.24.4 , 1.26 등 2.x 버전보다 아래 설치

결과 ) Tensorboard 확인 가능, ./model/checkpoint.pt
ㄴ > tensorboard --logdir=./runs/ 

ㄴ reward 변동이 없는것으로 보아, 학습은 안되는것 같습니다... 
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=8,
                        help="the number of actors to start, default is 8")
    parser.add_argument("--seed", type=int, default=23,
                        help="the seed of random, default is 20")
    parser.add_argument("--game_name", type=str, default='breakout',#'CartPole-v1',
                        help="the name of atari game, default is CartPole-v1")
    parser.add_argument('--length', type=int, default=20,
                        help='Number of Trajectkories to get from the agent')
    parser.add_argument('--total_steps', type=int, default=80000000,
                        help='Number of steps to run the agent, default is 80000000')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of Batch size to set ')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor, default is 0.99")
    parser.add_argument("--entropy_cost", type=float, default=0.0006,#0.00025,
                        help="Entropy cost/multiplier, default is 0.00025")
    parser.add_argument("--baseline_cost", type=float, default=.5,
                        help="Baseline cost/multiplier, default is 0.5")
    parser.add_argument("--lr", type=float, default=0.0006,#0.00048,
                        help="Learning rate, default is 0.001")
    parser.add_argument("--decay", type=float, default=.99,
                        help="RMSProp optimizer decay, default is .99")
    parser.add_argument("--momentum", type=float, default=0,
                        help="RMSProp momentum, default is 0")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="RMSProp epsilon, default is 0.1")
    parser.add_argument("--global_gradient_norm", type=int, default=40,#40,
                        help="RMSProp gradient norm, default is 40")
    parser.add_argument("--verbose", type=int, default=2,
                        help="RMSProp print log flag, default is 0")
    parser.add_argument('--save_path', type=str, default="./model/impala.pt",
                        help='Set the path to save trained model parameters')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Set interval to save trained model parameters')
    parser.add_argument('--load_path', type=str, default="./model/impala.pt",
                        help='Set the path to load trained model parameters')
    parser.add_argument('--log_dir', type=str, default="./runs/",
                        help='Set the path to check learning state using tensorboard')    
    parser.add_argument('--reward_clip', type=str, default="abs_one",
                        help='Set clipping reward type, default is "abs_one" (soft_asymmetric,abs_one,no_clip)')
    
    args = parser.parse_args()
    args.game_name = 'CartPole-v1'
    
    env_args = {'game_name': args.game_name, 'seed': args.seed}
    # action_size = get_action_size(Atari, env_args)
    action_size = get_action_size(CartPole, env_args)
    args.action_size = action_size    
    
    learner_model = IMPALA(action_size=args.action_size)

    # Optional Section

    mp.set_start_method('spawn')
    
    terminate_event = mp.Event()
    # optional : using 4-actor other process 
    experience_queue = mp.Queue(maxsize=1)

    # sync for trajectories
    lock = mp.Lock()
    sync_ps = SyncParameters(lock)
    sync_ps.push(learner_model.state_dict())
    
    idx = 0
    actor(idx, experience_queue=experience_queue,learner_model=learner_model, sync_ps=sync_ps, args=args, terminate_event=terminate_event)

    # learner
    # learner = mp.Process(target=learner, args=(learner_model, experience_queue, sync_ps, args, terminate_event))
    
    # actors of multi-process pool
    # actors = [mp.Process(target=actor, args=(idx, experience_queue, sync_ps, args, terminate_event))
    #           for idx in range(args.actors)]
    
    # synchronous learning and actors
    # learner.start()
    
    # [actor.start() for actor in actors]
    # [actor.join() for actor in actors]
        
    # learner.join()

    # main 프로세스 종료
    print("All processes have been terminated. Exiting main process.")
    # """
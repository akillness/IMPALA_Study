envs = [EnvThread(CartPole, env_args)
            for idx in range(args.actors)]
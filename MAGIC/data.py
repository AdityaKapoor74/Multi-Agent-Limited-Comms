import sys
import gym

from env_wrappers import *

def init(env_name, args):
    if env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0', disable_env_checker=True)
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0', disable_env_checker=True)
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'grf':
        
        try:
            env = gym.make('GRFWrapper-v0', disable_env_checker=True)
        except TypeError:
            # Fallback for gym < 0.21.0 or when disable_env_checker isn't available
            env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    else:
        raise RuntimeError("wrong env name")

    return env

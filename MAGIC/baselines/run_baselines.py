import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
from comm import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
from trainer import Trainer

sys.path.append("..") 
import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer
# import gym

# gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# specifically for starcraft2
parser.add_argument("--is_smac", action='store_true', default=False)
parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
# parser.add_argument('--eval_map_name', type=str, default='3m', help="Which smac map to eval on")
# parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
parser.add_argument("--add_move_state", action='store_true', default=False)
parser.add_argument("--add_local_obs", action='store_true', default=False)
parser.add_argument("--add_distance_state", action='store_true', default=False)
parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
parser.add_argument("--add_agent_id", action='store_true', default=False)
parser.add_argument("--add_visible_state", action='store_true', default=False)
parser.add_argument("--add_xy_state", action='store_true', default=False)
parser.add_argument("--use_state_agent", action='store_false', default=True)
parser.add_argument("--use_centralized_state", action='store_false', default=False)
parser.add_argument("--use_stacked_frames", action='store_false', default=False)
parser.add_argument("--stacked_frames", action='store_false', default=3)
parser.add_argument("--use_mustalive", action='store_false', default=True)
parser.add_argument("--add_center_xy", action='store_false', default=True)
# parser.add_argument("--random_agent_order", action='store_true', default=False)
parser.add_argument("--sight_range", type=int, default=9)
parser.add_argument("--shoot_range", type=int, default=6)
parser.add_argument('--n_rollout_threads', type=int, default=16)
parser.add_argument('--sc2_path', type=str, default=os.path.expanduser("~/marl_comms/Multi-Agent-Limited-Comms/3rdparty/StarCraftII"),
                    help='Path to StarCraft II installation')


# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--transformer_comm', action='store_true', default=False,
                    help="enable transformer comm model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')
parser.add_argument('--use_comet', action='store_true', default=False,
                    help='whether to use Comet ML for live logging')
parser.add_argument("--use_wandb", action='store_true', default=False,
                    help='whether to use wandb for live logging')
parser.add_argument('--experiment_name', type=str, default="",
                    help='name of the Comet ML Experiment')
parser.add_argument('--num_messages', default=15, type=int,
                    help='number of messages over which the continuous messages would be binned')
parser.add_argument('--use_comms_channel', action='store_true', default=False,
                    help='whether to use the discretization comms channel')
parser.add_argument('--comms_penalty', default=0.0, type=float, 
                    help='value of the comms penalty that would be used for the comms loss')
parser.add_argument('--use_fake_quantization', action='store_true', default=False,
                   help='Use FakeQuantization instead of noise-based approach')
parser.add_argument('--quant_bits', type=int, default=8, 
                   help='Number of bits for quantization')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                    help='Device to run the model on (default: cuda)')

# Transformer Policy specific args
# parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for softmax (higher = more exploration)')
# parser.add_argument('--adaptive_entropy', type=bool, default=False, help='Adaptively adjust entropy coefficient')
# parser.add_argument('--entropy_coefficient', type=float, default=0.001, help='Entropy coefficient for policy loss')
# parser.add_argument('--min_entropy', type=float, default=0.1, help='Minimum entropy value per action dimension')
# parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Maximum gradient norm for clipping')
# parser.add_argument('--rescale_logits', type=bool, default=False, help='Whether action_out already has log_softmax applied')

# Other baselines
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for softmax (higher = more exploration)')
parser.add_argument('--adaptive_entropy', type=bool, default=False, help='Adaptively adjust entropy coefficient')
parser.add_argument('--entropy_coefficient', type=float, default=0.0, help='Entropy coefficient for policy loss')
parser.add_argument('--min_entropy', type=float, default=0.0, help='Minimum entropy value per action dimension')
parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Maximum gradient norm for clipping')
parser.add_argument('--rescale_logits', type=bool, default=False, help='Whether action_out already has log_softmax applied')



init_args_for_env(parser)
args = parser.parse_args()

if args.env_name == 'starcraft2':
    os.environ["SC2PATH"] = args.sc2_path

# Automatically adjust args.device based on CUDA availability.
if args.device == 'cuda' and not torch.cuda.is_available():
    print("CUDA not available; switching device to CPU.")
    args.device = 'cpu'

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0
    
    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
    
if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False
env = data.init(args.env_name, args, False)

# Add these imports at the top of your main script file
from absl import flags
import sys

# Add this right after your imports, before any environment initialization
# This initializes the absl flags system
def initialize_sc2_flags():
    # Parse absl flags with an empty list to initialize the flags system
    # Use sys.argv[0] as the program name to avoid warnings
    flags.FLAGS(sys.argv[:1])
    
    # You can also set any specific SC2 flags you need here, for example:
    # flags.DEFINE_string('sc2_run_config', None, 'SC2 run config')
    
# Call the function before creating the environment
if args.env_name == 'starcraft2':
    initialize_sc2_flags()

if args.env_name == 'starcraft2':
    # Get environment spaces through proper channels
    if args.n_rollout_threads == 1:
        # For ShareDummyVecEnv
        if args.use_centralized_state:
            print("Using centralized state")
            num_inputs_data = env.envs[0].get_state_size()
        else:
            print("Using local observation")
            num_inputs_data = env.envs[0].get_obs_size()
        
        args.num_actions = env.envs[0].get_total_actions()
        args.dim_actions = 1
    else:
        # For ShareSubprocVecEnv - use remote calls
        # First, let's get spaces info
        env.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = env.remotes[0].recv()
        
        # For multi-processing environments, we need to use the communication pipe
        if args.use_centralized_state:
            print("Using centralized state")
            # Send custom command to get state size
            env.remotes[0].send(('get_state_size', None))
            num_inputs_data = env.remotes[0].recv()
        else:
            print("Using local observation")
            # Send custom command to get obs size
            env.remotes[0].send(('get_obs_size', None))
            num_inputs_data = env.remotes[0].recv()
        
        # Get action info
        env.remotes[0].send(('get_total_actions', None))
        args.num_actions = env.remotes[0].recv()
        args.dim_actions = 1
    
    # Handle the returned structure - get the first element which is the total size
    if isinstance(num_inputs_data, (list, tuple)):
        num_inputs = num_inputs_data[0]
    else:
        num_inputs = num_inputs_data
else:
    num_inputs = env.observation_dim
    args.num_actions = env.num_actions

    args.dim_actions = env.dim_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]

args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    if isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [*args.num_actions, 2]
    else:
        args.num_actions = [args.num_actions, 2]
    args.dim_actions = args.dim_actions + 1

# Recurrence
if args.recurrent or args.rnn_type == 'LSTM':
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(args)

if args.gacomm:
    policy_net = GACommNetMLP(args, num_inputs).to(args.device)
elif args.commnet:
    if args.tarcomm:
        policy_net = TarCommNetMLP(args, num_inputs)
    else:
        policy_net = CommNetMLP(args, num_inputs)

if args.use_comet:
    from comet_ml import Experiment
    comet_experiment = Experiment(
        api_key="8U8V63x4zSaEk4vDrtwppe8Vg",
        project_name="multiagentcomms"
    )
    if args.experiment_name != "":
        comet_experiment.set_name(args.experiment_name)
if args.use_wandb:
    import wandb
    if args.experiment_name != "":
        wandb.init(project="multi-agent-comms", config=vars(args), name=args.experiment_name)
    else:
        wandb.init(project="multi-agent-comms", config=vars(args))

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()    
    
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)
if args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
else:
    model_dir = Path('./saved') / args.env_name / 'other'
if args.env_name == 'grf':
    model_dir = model_dir / args.scenario
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run 


# First, let's properly extract and print the rewards
def extract_reward_values(reward_data):
    """
    Extract scalar reward values from potentially deeply nested structures
    """
    # For numpy arrays, convert to list for consistent processing
    if isinstance(reward_data, np.ndarray):
        reward_data = reward_data.tolist()
    
    # Handle different nesting structures
    if isinstance(reward_data, list):
        # Check if we have a deeply nested structure [[[1.], [1.], ...]]
        if len(reward_data) == 1 and isinstance(reward_data[0], list):
            # Unwrap the outer list
            agents_rewards = reward_data[0]
            
            # Extract the scalar values
            result = []
            for agent_reward in agents_rewards:
                if isinstance(agent_reward, list) and len(agent_reward) == 1:
                    result.append(agent_reward[0])
                else:
                    result.append(agent_reward)
            return result
            
        # Handle structure like [[1.], [1.], ...]
        elif all(isinstance(item, list) for item in reward_data):
            return [item[0] if len(item) == 1 else item for item in reward_data]
            
        # Handle flat list
        else:
            return reward_data
    
    # Handle scalar case
    return [reward_data]

def run(num_epochs): 
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        stat['reward'] = extract_reward_values(stat['reward'])
        
        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        
        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))
        if 'comms_loss' in stat.keys():
            print('comms_loss: {:.4f}'.format(stat['comms_loss']))   


        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save(final=False, episode=ep)

        if args.save:
            save(final=True)

        if args.use_comet:
            for agent_id, reward in enumerate(stat['reward']):

                comet_experiment.log_metric(f'agent{agent_id}_reward', reward, epoch=epoch)

            for k, v in stat.items():
                if not isinstance(v, list) and not isinstance(v, np.ndarray):
                    comet_experiment.log_metric(k, v, epoch=epoch)
        if args.use_wandb:
            for agent_id, reward in enumerate(stat['reward']):
                    
                wandb.log({f'agent{agent_id}_reward': reward}, step=epoch)

            for k, v in stat.items():
                if not isinstance(v, list) and not isinstance(v, np.ndarray):
                    wandb.log({k: v}, step=epoch)


def save(final, episode=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        save_name = 'model_' + args.experiment_name +'.pt'
        torch.save(d, run_dir / save_name)
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(episode)))

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if args.use_comet:
    comet_experiment.end()
if args.use_wandb:
    wandb.finish()

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)

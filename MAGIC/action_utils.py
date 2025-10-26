import numpy as np
import torch
from torch.autograd import Variable


def parse_action_args(args):
    # Special handling for StarCraft2
    if args.env_name == 'starcraft2':
        # StarCraft has a discrete action space
        args.continuous = False
        
        # Make sure num_actions is a valid integer
        if isinstance(args.num_actions, list):
            # If it's already a list, keep the format but ensure values are integers
            args.naction_heads = [int(a) for a in args.num_actions]
        else:
            # If it's a single integer, make it a list with one element
            args.naction_heads = [int(args.num_actions)]
            
        # Debug print to verify correct type
        print(f"StarCraft2 action heads: {args.naction_heads}")
        return
    
    # Original code for other environments
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out):
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        return ret

# def translate_action(args, env, action):
#     print(args.num_actions)
#     if args.num_actions[0] > 0:
#         # environment takes discrete action
#         action = [x.squeeze().data.numpy() for x in action]
#         actual = action
#         return action, actual
#     else:
#         if args.continuous:
#             action = action.data[0].numpy()
#             cp_action = action.copy()
#             # clip and scale action to correct range
#             for i in range(len(action)):
#                 low = env.action_space.low[i]
#                 high = env.action_space.high[i]
#                 cp_action[i] = cp_action[i] * args.action_scale
#                 cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
#                 cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
#             return action, cp_action
#         else:
#             actual = np.zeros(len(action))
#             for i in range(len(action)):
#                 low = env.action_space.low[i]
#                 high = env.action_space.high[i]
#                 actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
#             action = [x.squeeze().data[0] for x in action]
#             return action, actual

def translate_action(args, env, action):
    """
    Translates actions from neural network format to environment format.
    Handles both single actions and batched actions (from multi-threading).
    
    Args:
        args: Command line arguments
        env: Environment instance
        action: Action(s) from neural network
    
    Returns:
        Tuple of (original action, processed action for environment)
    """
    # For debugging
    # print("Action shape before processing:", [x.shape for x in action])
    
    if args.num_actions[0] > 0:  # Discrete action space
        # Convert PyTorch tensors to NumPy arrays
        action = [x.squeeze().cpu().data.numpy() for x in action]
        
        # For discrete actions, we just pass them through
        # This works for both single and batched actions
        actual = action
        return action, actual
        
    else:  # Continuous action space
        if args.continuous:
            # Check if we're dealing with batched actions
            if hasattr(action, 'shape') and len(action.shape) > 2:
                # Batched case
                batch_size = action.shape[0]
                action_data = action.data.numpy()
                cp_action = action_data.copy()
                
                # Apply scaling and clipping to each action in the batch
                for b in range(batch_size):
                    for i in range(cp_action.shape[1]):
                        low = env.action_space.low[i]
                        high = env.action_space.high[i]
                        cp_action[b, i] = cp_action[b, i] * args.action_scale
                        cp_action[b, i] = max(-1.0, min(cp_action[b, i], 1.0))
                        cp_action[b, i] = 0.5 * (cp_action[b, i] + 1.0) * (high - low) + low
            else:
                # Single action case - original code
                action_data = action.data[0].numpy()
                cp_action = action_data.copy()
                
                # Apply scaling and clipping
                for i in range(len(action_data)):
                    low = env.action_space.low[i]
                    high = env.action_space.high[i]
                    cp_action[i] = cp_action[i] * args.action_scale
                    cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                    cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            
            return action_data, cp_action
            
        else:  # Discretized continuous action space
            # Check if we're dealing with batched actions
            is_batched = False
            if isinstance(action[0], torch.Tensor) and len(action[0].shape) > 1:
                is_batched = True
                
            if is_batched:
                # Get batch size from the first action tensor
                batch_size = action[0].shape[0]
                action_dim = len(action)
                
                # Create output arrays for batched processing
                actual = np.zeros((batch_size, action_dim))
                processed_action = []
                
                # Process each action dimension for all batches
                for i in range(action_dim):
                    low = env.action_space.low[i]
                    high = env.action_space.high[i]
                    
                    # Get data for this action dimension for all batches
                    action_data = action[i].data.numpy()
                    
                    for b in range(batch_size):
                        actual[b, i] = action_data[b] * (high - low) / (args.naction_heads[i] - 1) + low
                    
                    processed_action.append(action_data)
                
            else:
                # Original code for single actions
                actual = np.zeros(len(action))
                processed_action = []
                
                for i in range(len(action)):
                    low = env.action_space.low[i]
                    high = env.action_space.high[i]
                    actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
                    processed_action.append(action[i].squeeze().data[0])
            
            return processed_action, actual
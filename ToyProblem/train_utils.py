import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from CommunicatingGoalEnv import CommunicatingGoalEnv
from network import PopArt, SpeakerNetwork, ListenerActor, Critic, DDCLChannel


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env_with_config(config, env_seed=None):
    """Create environment with proper seed handling"""
    def _init():
        env = CommunicatingGoalEnv(
            grid_size=config["grid_size"],
            z_dim=config["z_dim"],
            goal_sampling_mode='non_uniform',
            goal_frequencies=config["goal_frequencies"]
        )
        if env_seed is not None:
            env.action_space.seed(env_seed)
        return env
    return _init


def train_mappo_agent(
    config,
    lambda_comms=None,
    goal_frequencies=None,
    enable_logging=True,
    wandb_run=None,
    seed=None,
    return_convergence_data=False
):
    """
    Core MAPPO training function with configurable parameters
    
    Args:
        config: Training configuration dictionary
        lambda_comms: Communication loss weight (overrides config if provided)
        goal_frequencies: Goal frequency distribution (overrides config if provided)
        enable_logging: Whether to enable wandb logging
        wandb_run: Existing wandb run for logging
        seed: Random seed (overrides config if provided)
        return_convergence_data: Whether to return convergence tracking data
    
    Returns:
        Tuple of (speaker_network, listener_actor, critic, ddcl_channel, convergence_data)
        convergence_data is None if return_convergence_data is False
    """
    
    # Set up parameters
    if seed is not None:
        set_seeds(seed)
    elif 'seed' in config:
        set_seeds(config['seed'])
    else:
        set_seeds(1)  # Default seed
    
    # Use provided lambda_comms or fall back to config
    effective_lambda_comms = lambda_comms if lambda_comms is not None else config.get("lambda_comms", 0.0)
    
    # Set up goal frequencies
    if goal_frequencies is not None:
        config = config.copy()
        config["goal_frequencies"] = goal_frequencies
    elif "goal_frequencies" not in config:
        # Default goal frequencies from MAPPO_hyperparam_sweep.py
        config = config.copy()
        config["goal_frequencies"] = {
            (0, 0): 200,
            (7, 7): 100,
            (3, 4): 50,
            (4, 3): 25,
            (1, 6): 12,
            (6, 1): 1
        }
    
    device = torch.device(config["device"])
    batch_size = config["num_envs"] * config["num_steps"]
    
    # Create environments
    env_fns = [make_env_with_config(config, 4 + i) for i in range(config["num_envs"])]
    envs = gym.vector.SyncVectorEnv(env_fns)
    
    # Initialize networks
    speaker_network = SpeakerNetwork(2, config["z_dim"]).to(device)
    listener_actor = ListenerActor(2 + config["z_dim"], 5).to(device)
    critic = Critic(2 + 2).to(device)  # Global state: listener_pos + goal_pos
    
    # Create DDCLChannel - handle different constructor signatures
    if hasattr(DDCLChannel, '__init__'):
        # Check if constructor takes lambda_comms parameter
        import inspect
        sig = inspect.signature(DDCLChannel.__init__)
        if 'lambda_comms' in sig.parameters:
            ddcl_channel = DDCLChannel(config["ddcl_delta"], effective_lambda_comms).to(device)
        else:
            ddcl_channel = DDCLChannel(config["ddcl_delta"]).to(device)
    else:
        ddcl_channel = DDCLChannel(config["ddcl_delta"]).to(device)
    
    popart = PopArt(input_shape=(), device=device)
    
    # Optimizers
    actor_optimizer = torch.optim.Adam(
        list(speaker_network.parameters()) + list(listener_actor.parameters()),
        lr=config["learning_rate"], eps=1e-5
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config["learning_rate"], eps=1e-5
    )
    
    # Storage buffers
    obs_speaker = torch.zeros((config["num_steps"], config["num_envs"], 2)).to(device)
    obs_listener = torch.zeros((config["num_steps"], config["num_envs"], 2)).to(device)
    z_vectors = torch.zeros((config["num_steps"], config["num_envs"], config["z_dim"])).to(device)
    actions_listener = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    logprobs_listener = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    rewards = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    dones = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    values = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    
    global_step = 0
    num_updates = config["total_timesteps"] // batch_size
    
    # Optional convergence tracking
    convergence_data = [] if return_convergence_data else None
    
    # Manual logging setup
    completed_ep_rewards = deque(maxlen=100)
    completed_ep_lengths = deque(maxlen=100)
    completed_ep_successes = deque(maxlen=100)
    last_log_step = 0
    current_episode_rewards = np.zeros(config["num_envs"])
    current_episode_lengths = np.zeros(config["num_envs"])
    
    # Initialize environment
    initial_obs, _ = envs.reset()
    next_obs_speaker = torch.Tensor(initial_obs[0]).to(device)
    next_obs_listener = torch.Tensor(initial_obs[1]).to(device)
    next_done = torch.zeros(config["num_envs"]).to(device)
    
    # Create progress bar description
    desc = f"Training λ={effective_lambda_comms:.0e}"
    if seed is not None:
        desc += f" seed={seed}"
    
    for update in tqdm(range(1, num_updates + 1), desc=desc):
        # Collect rollouts
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            obs_speaker[step] = next_obs_speaker
            obs_listener[step] = next_obs_listener
            dones[step] = next_done
            
            with torch.no_grad():
                # Speaker outputs deterministic encoding
                z = speaker_network(next_obs_speaker)
                
                # Pass through DDCL channel
                hat_z = ddcl_channel(z)
                
                # Listener action
                listener_input = torch.cat([next_obs_listener, hat_z], dim=1)
                listener_dist = listener_actor(listener_input)
                action_listener = listener_dist.sample()
                listener_logprob = listener_dist.log_prob(action_listener)
                
                # Store values
                z_vectors[step] = z
                actions_listener[step] = action_listener
                logprobs_listener[step] = listener_logprob
                
                # Value estimation
                global_state = torch.cat([next_obs_listener, next_obs_speaker], dim=1)
                values[step] = critic(global_state).squeeze()
            
            # Environment step
            actions_for_env = (z.cpu().numpy(), action_listener.cpu().numpy())
            next_obs_tuple, reward_scalar, terminated_scalar, truncated_scalar, infos = envs.step(actions_for_env)
            
            rewards[step] = torch.tensor(reward_scalar).to(device)
            next_done = torch.tensor(np.logical_or(terminated_scalar, truncated_scalar)).float().to(device)
            next_obs_speaker = torch.Tensor(next_obs_tuple[0]).to(device)
            next_obs_listener = torch.Tensor(next_obs_tuple[1]).to(device)
            
            # Manual logging
            current_episode_rewards += reward_scalar
            current_episode_lengths += 1
            dones_this_step = np.logical_or(terminated_scalar, truncated_scalar)
            
            for i, done in enumerate(dones_this_step):
                if done:
                    completed_ep_rewards.append(current_episode_rewards[i])
                    completed_ep_lengths.append(current_episode_lengths[i])
                    completed_ep_successes.append(1.0 if current_episode_rewards[i] > 0 else 0.0)
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0
            
            # Periodic logging
            if enable_logging and wandb_run and global_step - last_log_step >= config.get("log_frequency", 4096):
                last_log_step = global_step
                if len(completed_ep_rewards) > 0:
                    wandb.log({
                        "global_step": global_step,
                        "charts/episodic_reward": np.mean(completed_ep_rewards),
                        "charts/episodic_length": np.mean(completed_ep_lengths),
                        "charts/success_rate": np.mean(completed_ep_successes)
                    })
                    completed_ep_rewards.clear()
                    completed_ep_lengths.clear()
                    completed_ep_successes.clear()
        
        # PPO Update - Proper advantage calculation
        with torch.no_grad():
            unnormalized_values = popart.unnormalize(values)
            next_global_state = torch.cat([next_obs_listener, next_obs_speaker], dim=1)
            next_value_normalized = critic(next_global_state).reshape(1, -1)
            next_value = popart.unnormalize(next_value_normalized)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = unnormalized_values[t + 1]
                
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - unnormalized_values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
            
            returns = advantages + unnormalized_values
        
        # Flatten batch
        b_obs_s = obs_speaker.reshape((-1, 2))
        b_obs_l = obs_listener.reshape((-1, 2))
        b_z = z_vectors.reshape((-1, config["z_dim"]))
        b_logprobs_l = logprobs_listener.reshape(-1)
        b_actions_l = actions_listener.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Update PopArt and normalize returns
        popart.update_stats(b_returns)
        b_returns_normalized = popart.normalize(b_returns)
        
        # Training loop
        inds = np.arange(batch_size)
        for epoch in range(config["update_epochs"]):
            if config["minibatch_size"] < batch_size:
                np.random.shuffle(inds)
            
            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = inds[start:end]
                
                # Forward pass through Speaker to get z encoding
                z_current = speaker_network(b_obs_s[mb_inds])
                
                # Communication loss
                comm_loss = ddcl_channel.calculate_loss_from_z(z_current)
                
                # Listener policy update (with gradients flowing back to Speaker)
                hat_z = ddcl_channel(z_current)
                listener_input = torch.cat([b_obs_l[mb_inds], hat_z], dim=1)
                new_dist_l = listener_actor(listener_input)
                new_logprob_l = new_dist_l.log_prob(b_actions_l[mb_inds])
                
                ratio_l = torch.exp(new_logprob_l - b_logprobs_l[mb_inds])
                pg_loss1_l = -b_advantages[mb_inds] * ratio_l
                pg_loss2_l = -b_advantages[mb_inds] * torch.clamp(ratio_l, 1 - config["clip_epsilon"], 1 + config["clip_epsilon"])
                pg_loss_l = torch.max(pg_loss1_l, pg_loss2_l).mean()
                
                # Only listener entropy
                entropy_l = new_dist_l.entropy().mean()
                
                # Actor loss with communication penalty
                actor_loss = (pg_loss_l
                              - config["entropy_coef"] * entropy_l
                              + effective_lambda_comms * comm_loss)
                
                # Actor update
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(speaker_network.parameters()) + list(listener_actor.parameters()), 0.5)
                actor_optimizer.step()
                
                # Critic update
                new_global_state = torch.cat([b_obs_l[mb_inds], b_obs_s[mb_inds]], dim=1)
                new_value = critic(new_global_state).squeeze()
                v_loss = 0.5 * ((new_value - b_returns_normalized[mb_inds]) ** 2).mean() * config["value_coef"]
                
                critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()
        
        # Optional convergence tracking
        avg_comm_bits = ddcl_channel.calculate_total_bits_from_z(b_z).mean().item()
        if convergence_data is not None and update % 50 == 0:
            convergence_data.append({
                'lambda': effective_lambda_comms,
                'seed': seed,
                'update': update,
                'comm_bits': avg_comm_bits,
                'comm_loss': comm_loss.item()
            })
        
        # Log training metrics if enabled
        if enable_logging and wandb_run:
            wandb.log({
                "global_step": global_step,
                "losses/value_loss": v_loss.item(),
                "losses/listener_pg_loss": pg_loss_l.item(),
                "losses/comm_loss_term": comm_loss.item(),
                "rollout/communication_bits": avg_comm_bits,
                "rollout/z_magnitude": torch.norm(z_vectors, dim=-1).mean().item(),
                "rollout/listener_entropy": entropy_l.item()
            })
    
    envs.close()
    return speaker_network, listener_actor, critic, ddcl_channel, convergence_data
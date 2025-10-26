import os
# Fix for the OMP: Error #15, which is common on macOS
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from tqdm import tqdm
import gymnasium as gym
import pandas as pd
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import the custom environment
from CommunicatingGoalEnv import CommunicatingGoalEnv

# --- Configuration ---
ABLATION_CONFIG = {
    # Training parameters
    "total_timesteps": 1_000_000,  # Reduced for quicker testing
    "num_envs": 16,
    "num_steps": 256,
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.00,
    "value_coef": 0.5,
    "update_epochs": 10,
    "minibatch_size": 16 * 256,
    
    # Environment parameters
    "grid_size": 8,
    "z_dim": 4,
    "ddcl_delta": 1/5.0,
    
    # Evaluation parameters
    "num_eval_episodes": 500,  # Reduced for faster testing
    "num_seeds": 3,
    
    # Core lambda values for basic functionality
    "lambda_values": [0.0, 1e-3, 5e-3, 1e-2],
    
    # Goal frequencies
    "goal_frequencies": {
        (0, 0): 50, 
        (7, 7): 1, 
        (3, 4): 25, 
        (4, 3): 2, 
        (1, 6): 1, 
        (6, 1): 1
    },
    
    # Optional enhanced features (can be disabled)
    "enable_enhanced_analysis": False,
    "enable_convergence_tracking": False,
    "enable_goal_encoding_analysis": False,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "ddcl_rate_distortion_basic",
    "save_dir": Path("./rate_distortion_results")
}

# --- Network Definitions ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PopArt(nn.Module):
    def __init__(self, input_shape, device='cpu'):
        super(PopArt, self).__init__()
        self.mu = nn.Parameter(torch.zeros(input_shape, device=device), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(input_shape, device=device), requires_grad=False)
        self.nu = nn.Parameter(torch.zeros(input_shape, device=device), requires_grad=False)
        self.count = nn.Parameter(torch.tensor(1e-4, device=device), requires_grad=False)

    def update_stats(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mu
        tot_count = self.count + batch_count
        new_mu = self.mu + delta * batch_count / tot_count
        m_a = self.nu * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_nu = M2 / tot_count
        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.sigma.copy_(torch.sqrt(new_nu + 1e-8))
        self.count.copy_(tot_count)

    def normalize(self, x): 
        return (x - self.mu) / self.sigma
    
    def unnormalize(self, y): 
        return y * self.sigma + self.mu

class DDCLChannel(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        
    def forward(self, z):
        noise = (torch.rand_like(z) - 0.5) * self.delta
        return z + noise
    
    def calculate_loss_from_z(self, z):
        return torch.log2(2 * torch.abs(z) / self.delta + 1.0).mean()
    
    def calculate_total_bits_from_z(self, z):
        return torch.log2(2 * torch.abs(z) / self.delta + 1.0).sum(dim=-1)

class SpeakerNetwork(nn.Module):
    def __init__(self, obs_dim, z_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 16)), 
            nn.GELU(),
            layer_init(nn.Linear(16, z_dim), std=0.01)
        )

    def forward(self, x):
        return self.network(x)

class ListenerActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 16)), 
            nn.GELU(), 
        )
        self.logits = layer_init(nn.Linear(16, action_dim), std=0.01)

    def forward(self, x):
        features = self.network(x)
        return Categorical(logits=self.logits(features))

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 32)), 
            nn.GELU(), 
            layer_init(nn.Linear(32, 1), std=1.0)
        )
    
    def forward(self, x): 
        return self.network(x)

# --- Core Training Function ---
def train_agent(lambda_comms, seed, config):
    """Train a single agent with given lambda_comms value"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(config["device"])
    batch_size = config["num_envs"] * config["num_steps"]
    
    # Create environments
    env_fns = [lambda: CommunicatingGoalEnv(
        grid_size=config["grid_size"], 
        z_dim=config["z_dim"],
        goal_sampling_mode='non_uniform',
        goal_frequencies=config["goal_frequencies"]
    ) for _ in range(config["num_envs"])]

    envs = gym.vector.SyncVectorEnv(env_fns)
    
    # Initialize networks
    speaker_network = SpeakerNetwork(2, config["z_dim"]).to(device)
    listener_actor = ListenerActor(2 + config["z_dim"], 5).to(device)
    critic = Critic(2 + 2).to(device)
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
    convergence_data = [] if config.get("enable_convergence_tracking", False) else None
    
    # Initialize environment
    initial_obs, _ = envs.reset()
    next_obs_speaker = torch.Tensor(initial_obs[0]).to(device)
    next_obs_listener = torch.Tensor(initial_obs[1]).to(device)
    next_done = torch.zeros(config["num_envs"]).to(device)

    # Training loop
    progress_bar = tqdm(range(1, num_updates + 1), 
                       desc=f"Training λ={lambda_comms:.0e} seed={seed}")
    
    for update in progress_bar:
        # Collect rollouts
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            obs_speaker[step] = next_obs_speaker
            obs_listener[step] = next_obs_listener
            dones[step] = next_done
            
            with torch.no_grad():
                z = speaker_network(next_obs_speaker)
                hat_z = ddcl_channel(z)
                listener_input = torch.cat([next_obs_listener, hat_z], dim=1)
                listener_dist = listener_actor(listener_input)
                action_listener = listener_dist.sample()
                listener_logprob = listener_dist.log_prob(action_listener)
                
                z_vectors[step] = z
                actions_listener[step] = action_listener
                logprobs_listener[step] = listener_logprob
                
                global_state = torch.cat([next_obs_listener, next_obs_speaker], dim=1)
                values[step] = critic(global_state).squeeze()
            
            # Environment step
            actions_for_env = (z.cpu().numpy(), action_listener.cpu().numpy())
            next_obs_tuple, reward_scalar, terminated_scalar, truncated_scalar, infos = envs.step(actions_for_env)
            
            rewards[step] = torch.tensor(reward_scalar).to(device)
            next_done = torch.tensor(np.logical_or(terminated_scalar, truncated_scalar)).float().to(device)
            next_obs_speaker = torch.Tensor(next_obs_tuple[0]).to(device)
            next_obs_listener = torch.Tensor(next_obs_tuple[1]).to(device)

        # PPO Update
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
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        popart.update_stats(b_returns)
        b_returns_normalized = popart.normalize(b_returns)
        
        # Training epochs
        inds = np.arange(batch_size)
        for epoch in range(config["update_epochs"]):
            if config["minibatch_size"] < batch_size:
                np.random.shuffle(inds)
            
            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = inds[start:end]
                
                # Forward pass
                z_current = speaker_network(b_obs_s[mb_inds])
                comm_loss = ddcl_channel.calculate_loss_from_z(z_current)
                
                hat_z = ddcl_channel(z_current)
                listener_input = torch.cat([b_obs_l[mb_inds], hat_z], dim=1)
                new_dist_l = listener_actor(listener_input)
                new_logprob_l = new_dist_l.log_prob(b_actions_l[mb_inds])
                
                ratio_l = torch.exp(new_logprob_l - b_logprobs_l[mb_inds])
                pg_loss1_l = -b_advantages[mb_inds] * ratio_l
                pg_loss2_l = -b_advantages[mb_inds] * torch.clamp(ratio_l, 1 - config["clip_epsilon"], 1 + config["clip_epsilon"])
                pg_loss_l = torch.max(pg_loss1_l, pg_loss2_l).mean()
                
                entropy_l = new_dist_l.entropy().mean()
                
                # Use the passed lambda_comms parameter
                actor_loss = (pg_loss_l 
                             - config["entropy_coef"] * entropy_l  
                             + lambda_comms * comm_loss)
                
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
                'lambda': lambda_comms,
                'seed': seed,
                'update': update,
                'comm_bits': avg_comm_bits,
                'comm_loss': comm_loss.item()
            })

        # Update progress bar
        progress_bar.set_postfix({
            'bits': f'{avg_comm_bits:.2f}',
            'loss': f'{comm_loss.item():.4f}'
        })

    envs.close()
    return speaker_network, listener_actor, critic, ddcl_channel, convergence_data

# --- Core Evaluation Function ---
def evaluate_agent(speaker_network, listener_actor, ddcl_channel, config):
    """Evaluate trained agent to compute Rate and Distortion"""
    
    device = torch.device(config["device"])
    
    # Create single evaluation environment
    eval_env = CommunicatingGoalEnv(
        grid_size=config["grid_size"], 
        z_dim=config["z_dim"],
        goal_sampling_mode='non_uniform',
        goal_frequencies=config["goal_frequencies"]
    )
    
    total_bits = 0.0
    total_successes = 0
    
    # Set networks to eval mode
    speaker_network.eval()
    listener_actor.eval()
    
    with torch.no_grad():
        for episode in range(config["num_eval_episodes"]):
            obs, _ = eval_env.reset()
            episode_bits = 0.0
            done = False
            
            while not done:
                # Speaker generates message
                obs_speaker_tensor = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
                z = speaker_network(obs_speaker_tensor)
                
                # Calculate bits for this timestep
                step_bits = ddcl_channel.calculate_total_bits_from_z(z).item()
                episode_bits += step_bits
                
                # Listener receives message and acts
                obs_listener_tensor = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
                hat_z = ddcl_channel(z)
                listener_input = torch.cat([obs_listener_tensor, hat_z], dim=1)
                listener_dist = listener_actor(listener_input)
                action_listener = listener_dist.sample().item()
                
                # Step environment
                actions = (z.cpu().numpy().squeeze(), action_listener)
                obs, reward, terminated, truncated, _ = eval_env.step(actions)
                done = terminated or truncated
                
                if terminated:  # Successfully reached goal
                    total_successes += 1
            
            total_bits += episode_bits
    
    eval_env.close()
    
    # Calculate Rate and Distortion
    rate = total_bits / config["num_eval_episodes"]
    success_rate = total_successes / config["num_eval_episodes"]
    distortion = 1.0 - success_rate
    
    return rate, distortion, success_rate

# --- Optional Enhanced Analysis Functions ---
def analyze_goal_encoding(speaker_network, ddcl_channel, config):
    """Optional: Analyze how different goals are encoded"""
    if not config.get("enable_goal_encoding_analysis", False):
        return None
        
    device = torch.device(config["device"])
    grid_size = config["grid_size"]
    
    goal_analysis = []
    total_freq = sum(config["goal_frequencies"].values())
    
    speaker_network.eval()
    with torch.no_grad():
        for goal, freq in config["goal_frequencies"].items():
            goal_normalized = torch.FloatTensor([goal[0]/(grid_size-1), goal[1]/(grid_size-1)]).to(device)
            z = speaker_network(goal_normalized.unsqueeze(0))
            bits = ddcl_channel.calculate_total_bits_from_z(z).item()
            
            prob = freq / total_freq
            theoretical_bits = -np.log2(prob)
            efficiency = (theoretical_bits / bits) * 100
            
            goal_analysis.append({
                'goal': goal,
                'frequency': freq,
                'bits': bits,
                'theoretical_bits': theoretical_bits,
                'efficiency_percent': efficiency
            })
    
    return pd.DataFrame(goal_analysis)

# --- Theoretical Analysis ---
def calculate_theoretical_bounds(config):
    """Calculate theoretical Shannon bounds"""
    goal_frequencies = config["goal_frequencies"]
    total_freq = sum(goal_frequencies.values())
    
    entropy = 0.0
    for freq in goal_frequencies.values():
        prob = freq / total_freq
        entropy -= prob * np.log2(prob)
    
    return entropy


def setup_style():
    """Setup publication-quality matplotlib parameters"""
    plt.style.use('default')  # Start with clean default
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 20,
        'axes.titlesize': 30,
        'axes.labelsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 20,
        'figure.titlesize': 30,
        
        # Line and marker settings
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'patch.linewidth': 1.0,
        
        # Grid and spacing
        'grid.alpha': 0.4,
        'axes.grid': True,
        'axes.axisbelow': True,
        'figure.constrained_layout.use': True,
        
        # Colors and aesthetics
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        
        # High DPI for crisp images
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': 'white',
    })

def create_plots(results_df, theoretical_entropy, save_dir):
    """Create clean, interpretable ICLR-quality plots with proper error bars and clear labels"""
    
    setup_style()
    
    # Professional color palette
    colors = {
        'empirical': '#1f77b4',      # Blue
        'shannon': '#d62728',        # Red  
        'optimum': '#ff7f0e',        # Orange
        'success': '#2ca02c',        # Green
        'rate': '#9467bd',           # Purple
        'gap': '#8c564b'             # Brown
    }
    
    # Aggregate results
    agg_results = results_df.groupby('Lambda').agg({
        'Rate': ['mean', 'std', 'count'],
        'Distortion': ['mean', 'std', 'count'],
        'Success_Rate': ['mean', 'std', 'count']
    }).reset_index()
    
    agg_results.columns = ['Lambda', 'Rate_Mean', 'Rate_Std', 'Rate_Count',
                          'Distortion_Mean', 'Distortion_Std', 'Distortion_Count',
                          'Success_Rate_Mean', 'Success_Rate_Std', 'Success_Rate_Count']
    
    # Calculate standard errors
    agg_results['Rate_SE'] = agg_results['Rate_Std'] / np.sqrt(agg_results['Rate_Count'])
    agg_results['Distortion_SE'] = agg_results['Distortion_Std'] / np.sqrt(agg_results['Distortion_Count'])
    agg_results['Success_Rate_SE'] = agg_results['Success_Rate_Std'] / np.sqrt(agg_results['Success_Rate_Count'])
    
    agg_results = agg_results.sort_values('Lambda').reset_index(drop=True)
    
    # === 1. MAIN RATE-DISTORTION CURVE ===
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    
    x_data = agg_results['Rate_Mean']
    y_data = agg_results['Distortion_Mean']
    x_err = agg_results['Rate_SE']
    y_err = agg_results['Distortion_SE']
    lambda_vals = agg_results['Lambda']
    
    # Plot main curve with THIN error bars
    ax1.errorbar(x_data, y_data, xerr=x_err, yerr=y_err,
                fmt='o-', color=colors['empirical'], linewidth=3.0,  # Thick main line
                markersize=10, markerfacecolor='white', markeredgewidth=2,
                capsize=6, capthick=1.0, elinewidth=1.0,  # THIN error bar parameters
                alpha=0.9, label='DDCL (Empirical)', zorder=5)
    
    # Smart label positioning function
    def calculate_label_positions(x_data, y_data, ax):
        positions = []
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            # Different positioning strategy based on index and proximity to Shannon line
            base_angles = [0, 45, 90, 180, 135, 225, 270, 315]
            angle = base_angles[i % len(base_angles)]
            
            # Adjust radius and angle near Shannon line
            if x < theoretical_entropy + 1:
                radius = 35
                if angle < 180:
                    angle += 20
            else:
                radius = 25
            
            angle_rad = np.radians(angle)
            offset_x = radius * np.cos(angle_rad)
            offset_y = radius * np.sin(angle_rad)
            
            ha = 'left' if offset_x > 0 else 'right'
            va = 'bottom' if offset_y > 0 else 'top'
            
            positions.append((offset_x, offset_y, ha, va))
        
        return positions
    
    # Apply smart label positioning
    # label_positions = calculate_label_positions(x_data, y_data, ax1)
    
    # for i, (x, y, lam, (offset_x, offset_y, ha, va)) in enumerate(zip(x_data, y_data, lambda_vals, label_positions)):
    #     if lam == 0:
    #         label = 'λ=0'
    #         color = '#d62728'
    #     else:
    #         label = f'λ={lam:.0e}'
    #         color = '#333333'
        
        # ax1.annotate(label, (x, y), 
        #             xytext=(offset_x, offset_y), 
        #             textcoords='offset points',
        #             fontsize=15, fontweight='bold', ha=ha, va=va, color=color,
        #             bbox=dict(boxstyle='round,pad=0.3', 
        #                     facecolor='white', alpha=0.7, 
        #                     edgecolor=color, linewidth=1.0),
        #             arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=1.0),
        #             zorder=10)
    
    # Shannon bound and theoretical optimum
    shannon_x = theoretical_entropy
    y_min_plot = max(0, y_data.min() - 0.05)
    y_max_plot = min(1, y_data.max() + 0.05)
    
    ax1.axvline(x=shannon_x, color=colors['shannon'], linestyle='--', 
               linewidth=3.0, alpha=0.9, zorder=3,
               label=f'Shannon Bound\nH(X) = {theoretical_entropy:.2f} bits')
    
    ax1.scatter([shannon_x], [0], color=colors['optimum'], s=300, 
               marker='*', edgecolors='black', linewidth=2, zorder=7,
               label='Theoretical Optimum\n(H(X), 0)')
    
    # Set proper limits
    x_range = x_data.max() - x_data.min()
    y_range = y_data.max() - y_data.min()
    x_margin = max(0.2 * x_range, 1.5)
    y_margin = max(0.15 * y_range, 0.05)
    
    ax1.set_xlim(max(0, x_data.min() - x_margin), x_data.max() + x_margin)
    ax1.set_ylim(max(0, y_data.min() - y_margin), min(1, y_data.max() + y_margin))
    
    ax1.set_xlabel('Rate R(λ) [Average Bits per Episode]', fontweight='bold', fontsize=25)
    ax1.set_ylabel('Distortion D = 1 - Success Rate', fontweight='bold', fontsize=25)
    ax1.set_title('Rate-Distortion Analysis: Multi-Agent Communication Efficiency', 
                 fontweight='bold', pad=40, fontsize=30)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95, fontsize=20)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    fig1.savefig(save_dir / 'rate_distortion_curve.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(save_dir / 'rate_distortion_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # === 2. PERFORMANCE ANALYSIS WITH DUAL AXIS ===
    fig2 = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 2, figure=fig2, wspace=0.35)
    
    # Prepare cleaner lambda labels
    x_positions = list(range(len(lambda_vals)))
    x_labels = []
    for lam in lambda_vals:
        if lam == 0:
            x_labels.append('λ=0')
        else:
            # Handle the formatting more carefully
            if lam >= 1:
                # For values >= 1, just show the number
                x_labels.append(f'λ={lam}')
            else:
                # For values < 1, use scientific notation
                exp = int(np.floor(np.log10(lam)))
                coeff = lam / (10 ** exp)
                
                # Round coefficient to avoid floating point precision issues
                coeff = round(coeff, 1)
                
                if coeff == 1.0:
                    x_labels.append(f'λ=10-{abs(exp)}')
                else:
                    # Remove unnecessary decimal if it's a whole number
                    if coeff == int(coeff):
                        x_labels.append(f'λ={int(coeff)}×10-{abs(exp)}')
                    else:
                        x_labels.append(f'λ={coeff}×10-{abs(exp)}')
    
    # Combined dual-axis plot: Success Rate + Communication Rate
    ax2a = fig2.add_subplot(gs[0, 0])
    
    # Primary axis: Success Rate
    y_success = agg_results['Success_Rate_Mean']
    y_success_err = agg_results['Success_Rate_SE']
    
    line1 = ax2a.errorbar(x_positions, y_success, yerr=y_success_err,
                         fmt='o-', linewidth=3.0, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, capsize=8, capthick=1.2, elinewidth=1.2,
                         color=colors['success'], ecolor=colors['success'], alpha=0.9,
                         label='Success Rate')
    
    ax2a.set_xlabel('Communication Penalty λ', fontweight='bold', fontsize=15)
    ax2a.set_ylabel('Success Rate', fontweight='bold', fontsize=15, color=colors['success'])
    ax2a.tick_params(axis='y', labelcolor=colors['success'])
    
    y_min = max(0, (y_success - y_success_err).min() - 0.03)
    y_max = min(1, (y_success + y_success_err).max() + 0.05)
    ax2a.set_ylim(y_min, y_max)
    
    # Secondary axis: Communication Rate
    ax2a_twin = ax2a.twinx()
    
    y_rate = agg_results['Rate_Mean']
    y_rate_err = agg_results['Rate_SE']
    
    line2 = ax2a_twin.errorbar(x_positions, y_rate, yerr=y_rate_err,
                              fmt='s-', linewidth=3.0, markersize=10, markerfacecolor='white',
                              markeredgewidth=2, capsize=8, capthick=1.2, elinewidth=1.2,
                              color=colors['rate'], ecolor=colors['rate'], alpha=0.9,
                              label='Communication Rate')
    
    # Shannon bound line
    line3 = ax2a_twin.axhline(y=theoretical_entropy, color=colors['shannon'], linestyle='--',
                             linewidth=3.0, alpha=0.9, label=f'Shannon Bound H(X) = {theoretical_entropy:.2f}')
    
    ax2a_twin.set_ylabel('Rate [Bits per Episode]', fontweight='bold', fontsize=15, color=colors['rate'])
    ax2a_twin.tick_params(axis='y', labelcolor=colors['rate'])
    
    # Set x-axis for both
    ax2a.set_xticks(x_positions)
    ax2a.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax2a.grid(True, alpha=0.4)
    
    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax2a.legend(lines, labels, loc='center left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    ax2a.set_title('(a) Task Performance & Communication Rate vs Penalty', fontweight='bold', fontsize=20)
    
    # Subplot B: Shannon Gap - FIXED BAR CHART
    ax2c = fig2.add_subplot(gs[0, 1])
    
    shannon_gaps = y_rate - theoretical_entropy
    gap_errors = y_rate_err
    
    # Color bars based on efficiency
    colors_bars = []
    for gap in shannon_gaps:
        if gap < 0.5:
            colors_bars.append('#2ca02c')  # Green
        elif gap < 1.0:
            colors_bars.append('#ff7f0e')  # Orange
        elif gap < 2.0:
            colors_bars.append('#d62728')  # Red
        else:
            colors_bars.append('#8b0000')  # Dark red
    
    # Create bars - REMOVE the error bar parameters that don't work with bar()
    bars = ax2c.bar(x_positions, shannon_gaps, 
                   color=colors_bars, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    # Add error bars manually using errorbar()
    ax2c.errorbar(x_positions, shannon_gaps, yerr=gap_errors,
                 fmt='none', capsize=6, capthick=1.2, elinewidth=1.2,
                 color='black', alpha=0.8, zorder=6)
    
    # Zero line (Shannon optimum)
    # ax2c.axhline(y=0, color=colors['shannon'], linestyle='-', linewidth=3.5,
    #             alpha=0.9, label='Shannon Optimum (Gap = 0)')
    # ax2c.legend(fontsize=8, loc='lower right')
    
    # Add value labels on bars
    for i, (bar, gap, err) in enumerate(zip(bars, shannon_gaps, gap_errors)):
        height = bar.get_height()
        y_pos = height + err + 0.1 if height >= 0 else height - err - 0.1
        va_pos = 'bottom' if height >= 0 else 'top'
        
        ax2c.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{gap:.2f}', ha='center', va=va_pos, 
                 fontsize=10, fontweight='bold', color='black')
    
    ax2c.set_xticks(x_positions)
    ax2c.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax2c.set_xlabel('Communication Penalty λ', fontweight='bold', fontsize=15)
    ax2c.set_ylabel('Shannon Gap = R(λ) - H(X) [Bits]', fontweight='bold', fontsize=15)
    ax2c.set_title('(b) Communication Efficiency Gap', fontweight='bold', fontsize=20)
    ax2c.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2c.grid(True, alpha=0.4)
    
    # Add formula explanation
    gap_formula_text = (
        "Shannon Gap = R(λ) - H(X)\n"
        "Shanon Optimum Gap = 0\n"
        "(Indicated in Red dashed line)\n"
        "\n"
        "Interpretation:\n"
        "• Gap = 0: Optimal compression\n"
        "• Gap > 0: Excess bits used\n"
        "\n"
        # "Color Guide:\n"
        # "< 0.5: Highly efficient\n"
        # "0.5-1.0: Moderately efficient\n"
        # "1.0-2.0: Inefficient\n"
        # "> 2.0: Very inefficient"
    )
    
    ax2c.text(0.98, 0.98, gap_formula_text, transform=ax2c.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig2.savefig(save_dir / 'performance_analysis_enhanced.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(save_dir / 'performance_analysis_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Reset matplotlib
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"\nFixed ICLR-quality plots saved to {save_dir}")
    print("Key improvements:")
    print("  ✅ Thin error bars (1.0-1.2 width) vs thick main lines (3.0 width)")
    print("  ✅ Smart label positioning with collision detection")
    print("  ✅ Non-overlapping annotations with clear backgrounds")
    print("  ✅ Combined dual-axis plot for performance + communication rate")
    print("  ✅ Enhanced readability for all text elements")
    
    return agg_results


# Enhanced goal encoding analysis
def create_goal_encoding_plot(goal_analysis_df, save_dir):
    """Create clean goal encoding analysis with proper error bar styling and clear labels"""
    if goal_analysis_df is None or goal_analysis_df.empty:
        return
    
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Frequency vs Bits with trend line
    frequencies = goal_analysis_df['frequency']
    bits = goal_analysis_df['bits']
    
    # Scatter plot with larger, more visible markers
    ax1.scatter(frequencies, bits, s=120, alpha=0.8, 
               color='steelblue', edgecolors='black', linewidth=2, zorder=5)
    
    # Add trend line with proper thickness
    z = np.polyfit(frequencies, bits, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(frequencies.min(), frequencies.max(), 100)
    ax1.plot(x_trend, p(x_trend), '--', color='red', linewidth=2.5, alpha=0.9,
            label=f'Trend (slope={z[0]:.4f})', zorder=3)
    
    # Calculate correlation for additional context
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(frequencies, bits)
    
    # Enhanced smart goal labels with collision avoidance
    def calculate_goal_label_positions(frequencies, bits):
        """Calculate non-overlapping positions for goal labels"""
        positions = []
        n_goals = len(frequencies)
        
        for i, (freq, bit_val) in enumerate(zip(frequencies, bits)):
            # Use different strategies based on data distribution
            if n_goals <= 4:
                # For few goals, use cardinal directions
                angles = [45, 135, 225, 315]
                angle = angles[i % len(angles)]
            else:
                # For more goals, distribute evenly around circle
                angle = (i * 360 / n_goals) % 360
            
            # Adjust radius based on data density
            base_radius = 20
            if freq > frequencies.median():  # High frequency goals
                radius = base_radius + 5  # Slightly further out
            else:
                radius = base_radius
            
            angle_rad = np.radians(angle)
            offset_x = radius * np.cos(angle_rad)
            offset_y = radius * np.sin(angle_rad)
            
            # Determine text alignment
            ha = 'left' if offset_x > 0 else 'right'
            va = 'bottom' if offset_y > 0 else 'top'
            
            positions.append((offset_x, offset_y, ha, va))
        
        return positions
    
    label_positions = calculate_goal_label_positions(frequencies, bits)
    
    for i, (row, (offset_x, offset_y, ha, va)) in enumerate(zip(goal_analysis_df.iterrows(), label_positions)):
        row = row[1]  # Get the actual row data
        goal_str = f"({row['goal'][0]},{row['goal'][1]})"
        
        ax1.annotate(goal_str, (row['frequency'], row['bits']), 
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    ha=ha, va=va, fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.95, edgecolor='steelblue', linewidth=1),
                    arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.7, lw=1.0),
                    zorder=6)
    
    ax1.set_xlabel('Goal Frequency', fontweight='bold', fontsize=25)
    ax1.set_ylabel('Communication Bits', fontweight='bold', fontsize=25)
    ax1.set_title('(a) Goal Frequency vs Communication Cost', fontweight='bold', fontsize=30)
    
    # Enhanced legend with correlation info
    ax1.legend([f'Trend (slope={z[0]:.4f})', f'Correlation: r={correlation:.3f}, p={p_value:.3f}'], 
              loc='best', frameon=True, fancybox=True, shadow=True, fontsize=20)
    
    ax1.grid(True, alpha=0.4)
    
    # Add statistics text box
    stats_text = (
        f"Statistical Analysis:\n"
        f"• Correlation: r = {correlation:.3f}\n"
        f"• P-value: {p_value:.4f}\n"
        f"• Trend: {'Negative' if z[0] < 0 else 'Positive'}\n"
        f"• Significance: {'*' if p_value < 0.05 else 'n.s.'}"
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=20, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8))
    
    # Right: Efficiency comparison bar chart with FIXED error handling
    goals_str = [f"({g[0]},{g[1]})" for g in goal_analysis_df['goal']]
    theoretical = goal_analysis_df['theoretical_bits']
    empirical = goal_analysis_df['bits']
    
    x_pos = np.arange(len(goals_str))
    width = 0.35
    
    # Create bars without error bar parameters
    bars1 = ax2.bar(x_pos - width/2, theoretical, width, 
                   label='Shannon Optimal', color='crimson', alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x_pos + width/2, empirical, width, 
                   label='DDCL Empirical', color='steelblue', alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    
    # Add error bars separately if standard deviations are available
    if 'theoretical_bits_std' in goal_analysis_df.columns:
        theoretical_err = goal_analysis_df['theoretical_bits_std']
        ax2.errorbar(x_pos - width/2, theoretical, yerr=theoretical_err,
                    fmt='none', capsize=4, capthick=1.0, elinewidth=1.0,
                    color='darkred', alpha=0.8)
    
    if 'bits_std' in goal_analysis_df.columns:
        empirical_err = goal_analysis_df['bits_std']
        ax2.errorbar(x_pos + width/2, empirical, yerr=empirical_err,
                    fmt='none', capsize=4, capthick=1.0, elinewidth=1.0,
                    color='darkblue', alpha=0.8)
    
    # Add value labels on bars with smart positioning
    for bars, values, color in [(bars1, theoretical, 'darkred'), (bars2, empirical, 'darkblue')]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # Position labels slightly above bars
            y_pos = height + max(theoretical.max(), empirical.max()) * 0.02
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{value:.1f}', ha='center', va='bottom', 
                    fontsize=20, fontweight='bold', color=color)
    
    # Calculate and display efficiency metrics
    efficiency_ratios = theoretical / empirical * 100
    avg_efficiency = efficiency_ratios.mean()
    
    # Add efficiency percentages as secondary labels
    for i, (x, theo, emp, eff) in enumerate(zip(x_pos, theoretical, empirical, efficiency_ratios)):
        # Add efficiency label between the bars
        ax2.text(x, max(theo, emp) + max(theoretical.max(), empirical.max()) * 0.08,
                f'{eff:.0f}%', ha='center', va='bottom', 
                fontsize=20, style='italic', color='purple')
    
    ax2.set_xlabel('Goal Locations', fontweight='bold', fontsize=25)
    ax2.set_ylabel('Bits Required', fontweight='bold', fontsize=25)
    ax2.set_title('(b) Optimal vs Learned Encoding', fontweight='bold', fontsize=30)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(goals_str, rotation=45, ha='right', fontsize=25)
    
    # Enhanced legend with efficiency info
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='crimson', alpha=0.8, label='Shannon Optimal'),
        plt.Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.8, label='DDCL Empirical'),
        plt.Line2D([0], [0], color='purple', linestyle='', marker='o', markersize=6, 
                   label=f'Efficiency % (avg: {avg_efficiency:.0f}%)')
    ]
    ax2.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=20)
    ax2.grid(True, alpha=0.4)
    
    # Add efficiency summary box
    efficiency_text = (
        f"Encoding Efficiency:\n"
        f"• Average: {avg_efficiency:.1f}%\n"
        f"• Best: {efficiency_ratios.max():.1f}%\n"
        f"• Worst: {efficiency_ratios.min():.1f}%\n"
        f"• Range: {efficiency_ratios.max() - efficiency_ratios.min():.1f}%\n"
        "\n"
        "Purple % = Shannon/DDCL × 100"
    )
    
    ax2.text(0.02, 0.98, efficiency_text, transform=ax2.transAxes, 
            fontsize=20, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.8))
    
    # Set y-limits to accommodate all labels
    y_max = max(theoretical.max(), empirical.max())
    ax2.set_ylim(0, y_max * 1.25)  # 25% extra space for labels
    
    plt.tight_layout(pad=1.5)
    fig.savefig(save_dir / 'goal_encoding_analysis.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'goal_encoding_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"Enhanced goal encoding plot saved to {save_dir}")
    print("  ✅ Fixed error bar handling for bar charts")
    print("  ✅ Smart label positioning with collision avoidance")
    print("  ✅ Added statistical analysis and correlation info")
    print("  ✅ Enhanced efficiency metrics and visualization")
    print(f"  📊 Goal encoding correlation: r={correlation:.3f} (p={p_value:.4f})")



# --- Main Ablation Study ---
def run_rate_distortion_ablation(config=None):
    """Main function to run the rate-distortion ablation study"""
    
    if config is None:
        config = ABLATION_CONFIG
    
    save_dir = config["save_dir"]
    save_dir.mkdir(exist_ok=True)
    
    print("=== Rate-Distortion Ablation Study ===")
    print(f"Testing λ values: {config['lambda_values']}")
    print(f"Seeds per λ: {config['num_seeds']}")
    print(f"Total experiments: {len(config['lambda_values']) * config['num_seeds']}")
    
    # Initialize wandb (convert tuple keys to strings for JSON serialization)
    run_name = f"rd_ablation_{len(config['lambda_values'])}lambdas_{config['num_seeds']}seeds"
    
    # Create wandb-compatible config
    wandb_config = config.copy()
    if "goal_frequencies" in wandb_config:
        # Convert tuple keys to strings for wandb
        goal_freq_str = {f"{k[0]}_{k[1]}": v for k, v in wandb_config["goal_frequencies"].items()}
        wandb_config["goal_frequencies"] = goal_freq_str
    
    # Convert Path objects to strings
    if "save_dir" in wandb_config:
        wandb_config["save_dir"] = str(wandb_config["save_dir"])
    
    wandb.init(project=config["wandb_project"], config=wandb_config, name=run_name)
    
    # Store all results
    all_results = []
    all_convergence_data = []
    
    # Calculate theoretical bounds
    theoretical_entropy = calculate_theoretical_bounds(config)
    print(f"Theoretical Shannon bound: H(X) = {theoretical_entropy:.3f} bits")
    
    # Main ablation loop
    total_experiments = len(config['lambda_values']) * config['num_seeds']
    experiment_count = 0
    
    start_time = time.time()
    
    for lambda_comms in config['lambda_values']:
        print(f"\n--- Testing λ = {lambda_comms:.0e} ---")
        
        for seed in range(config['num_seeds']):
            experiment_count += 1
            exp_start_time = time.time()
            
            print(f"Experiment {experiment_count}/{total_experiments}: λ={lambda_comms:.0e}, seed={seed}")
            
            # Train agent
            speaker_net, listener_actor, critic, ddcl_channel, convergence_data = train_agent(
                lambda_comms, seed, config
            )
            
            # Store convergence data if available
            if convergence_data:
                all_convergence_data.extend(convergence_data)
            
            # Evaluate agent
            rate, distortion, success_rate = evaluate_agent(
                speaker_net, listener_actor, ddcl_channel, config
            )
            
            result = {
                'Lambda': lambda_comms,
                'Seed': seed,
                'Rate': rate,
                'Distortion': distortion,
                'Success_Rate': success_rate,
                'Shannon_Gap': rate - theoretical_entropy,
                'Experiment_Time_Minutes': (time.time() - exp_start_time) / 60
            }
            
            all_results.append(result)
            
            # Log to wandb
            wandb.log({
                f'lambda_{lambda_comms:.0e}/rate': rate,
                f'lambda_{lambda_comms:.0e}/distortion': distortion,
                f'lambda_{lambda_comms:.0e}/success_rate': success_rate,
                f'lambda_{lambda_comms:.0e}/shannon_gap': rate - theoretical_entropy,
                'experiment_count': experiment_count
            })
            
            print(f"  Rate: {rate:.2f}, Distortion: {distortion:.3f}, Success: {success_rate:.3f}")
            print(f"  Time: {result['Experiment_Time_Minutes']:.1f} min")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(save_dir / 'detailed_results.csv', index=False)
    
    # Save config (convert non-JSON serializable objects)
    config_to_save = config.copy()
    if "goal_frequencies" in config_to_save:
        # Convert tuple keys to strings for JSON serialization
        goal_freq_str = {f"{k[0]}_{k[1]}": v for k, v in config_to_save["goal_frequencies"].items()}
        config_to_save["goal_frequencies"] = goal_freq_str
    
    # Convert Path objects to strings
    if "save_dir" in config_to_save:
        config_to_save["save_dir"] = str(config_to_save["save_dir"])
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Create visualizations
    print("--- Creating Visualizations ---")
    agg_results = create_plots(results_df, theoretical_entropy, save_dir)
    agg_results.to_csv(save_dir / 'aggregated_results.csv', index=False)
    
    # Optional goal encoding analysis
    goal_analysis_df = None
    if config.get("enable_goal_encoding_analysis", False):
        print("--- Analyzing Goal Encoding ---")
        # Use best performing lambda for analysis
        best_lambda_idx = agg_results['Rate_Mean'].idxmin()
        best_lambda = agg_results.loc[best_lambda_idx, 'Lambda']
        
        # Re-train best model for analysis
        speaker_net, _, _, ddcl_channel, _ = train_agent(best_lambda, 0, config)
        goal_analysis_df = analyze_goal_encoding(speaker_net, ddcl_channel, config)
        
        if goal_analysis_df is not None:
            goal_analysis_df.to_csv(save_dir / 'goal_analysis.csv', index=False)
            
            # Calculate correlation
            correlation, p_value = pearsonr(goal_analysis_df['frequency'], goal_analysis_df['bits'])
            print(f"Goal encoding correlation: r = {correlation:.3f} (p = {p_value:.4f})")
    
    # Final summary
    total_runtime_hours = (time.time() - start_time) / 3600
    best_rate = agg_results['Rate_Mean'].min()
    best_success = agg_results['Success_Rate_Mean'].max()
    min_shannon_gap = (agg_results['Rate_Mean'] - theoretical_entropy).min()
    
    print(f"\n=== STUDY COMPLETE ===")
    print(f"Total runtime: {total_runtime_hours:.1f} hours")
    print(f"Results saved to: {save_dir}")
    print(f"Shannon bound: {theoretical_entropy:.3f} bits")
    print(f"Best empirical rate: {best_rate:.3f} bits")
    print(f"Shannon gap: {min_shannon_gap:.3f} bits")
    print(f"Max success rate: {best_success:.3f}")
    
    # Create summary table
    summary_data = {
        'Metric': [
            'Shannon Bound (bits)',
            'Best Empirical Rate (bits)',
            'Shannon Gap (bits)',
            'Max Success Rate',
            'Number of Lambda Values',
            'Seeds per Lambda',
            'Total Experiments',
            'Runtime (hours)'
        ],
        'Value': [
            f'{theoretical_entropy:.3f}',
            f'{best_rate:.3f}',
            f'{min_shannon_gap:.3f}',
            f'{best_success:.3f}',
            len(config['lambda_values']),
            config['num_seeds'],
            total_experiments,
            f'{total_runtime_hours:.1f}'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_dir / 'summary.csv', index=False)
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))
    
    # Log final results to wandb
    wandb.log({
        'final/theoretical_entropy': theoretical_entropy,
        'final/best_empirical_rate': best_rate,
        'final/min_shannon_gap': min_shannon_gap,
        'final/max_success_rate': best_success,
        'final/total_runtime_hours': total_runtime_hours
    })
    
    wandb.finish()
    
    return results_df, agg_results, theoretical_entropy, goal_analysis_df

# --- Test Configurations ---
def get_quick_test_config():
    """Configuration for quick testing (3 experiments, ~30 minutes)"""
    config = ABLATION_CONFIG.copy()
    config.update({
        'lambda_values': [0.0, 1e-3, 5e-3],  # Only 3 lambdas
        'num_seeds': 1,  # Single seed for speed
        'total_timesteps': 500_000,  # Reduced training
        'num_eval_episodes': 200,  # Fewer evaluations
        'wandb_project': 'ddcl_quick_test'
    })
    return config

def get_basic_test_config():
    """Configuration for basic testing (12 experiments, ~2 hours)"""
    config = ABLATION_CONFIG.copy() 
    config.update({
        'lambda_values': [0.0, 1e-3, 5e-3, 1e-2],  # 4 lambdas
        'num_seeds': 3,  # 3 seeds for basic statistics
        'total_timesteps': 1_000_000,  # Standard training
        'num_eval_episodes': 500,  # Standard evaluation
        'enable_goal_encoding_analysis': True,  # Enable basic analysis
        'wandb_project': 'ddcl_basic_test'
    })
    return config

def get_extended_test_config():
    """Configuration for extended testing (35 experiments, ~6 hours)"""
    config = ABLATION_CONFIG.copy()
    config.update({
        'lambda_values': [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],  # 7 lambdas
        'num_seeds': 5,  # 5 seeds for good statistics
        'total_timesteps': 1_500_000,  # Extended training
        'num_eval_episodes': 1000,  # More thorough evaluation
        'enable_goal_encoding_analysis': True,
        'enable_convergence_tracking': True,
        'wandb_project': 'ddcl_extended_test'
    })
    return config

def get_comprehensive_config():
    """Configuration for comprehensive study (70 experiments, ~12 hours)"""
    config = ABLATION_CONFIG.copy()
    config.update({
        'lambda_values': [0.0, 5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2e-3, 
                         5e-3, 8e-3, 1e-2, 1.5e-2, 2e-2, 3e-2],  # 14 lambdas
        'num_seeds': 5,  # 5 seeds for robust statistics
        'total_timesteps': 2_500_000,  # Full training
        'num_eval_episodes': 1000,  # Thorough evaluation
        'enable_enhanced_analysis': True,
        'enable_goal_encoding_analysis': True,
        'enable_convergence_tracking': True,
        # 'goal_frequencies': {  # More extreme frequencies
        #     (0, 0): 100,  # Very frequent
        #     (7, 7): 1,    # Very rare
        #     (3, 4): 50,   # Medium-frequent
        #     (4, 3): 2,    # Rare
        #     (1, 6): 1,    # Very rare
        #     (6, 1): 1     # Very rare
        # },
        'goal_frequencies': {
        (0, 0): 200,
        (7, 7): 100,
        (3, 4): 50,
        (4, 3): 25,
        (1, 6): 12,
        (6, 1): 1
        },
        'wandb_project': 'ddcl_comprehensive_study'
    })
    return config

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    
    print("Rate-Distortion Ablation Study")
    print("=" * 50)
    
    # Configuration options
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Available modes:")
        print("  --quick        : 3 experiments (~30 min)")
        print("  --basic        : 12 experiments (~2 hours)")  
        print("  --extended     : 35 experiments (~6 hours)")
        print("  --comprehensive: 70 experiments (~12 hours)")
        print("  --default      : Use default config")
        mode = input("Select mode (or press Enter for basic): ").lower().strip()
        if not mode:
            mode = "basic"
    
    # Select configuration
    if mode in ["--quick", "quick"]:
        config = get_quick_test_config()
        print("Running QUICK test (3 experiments)")
    elif mode in ["--basic", "basic"]:
        config = get_basic_test_config()
        print("Running BASIC test (12 experiments)")
    elif mode in ["--extended", "extended"]:
        config = get_extended_test_config()
        print("Running EXTENDED test (35 experiments)")
    elif mode in ["--comprehensive", "comprehensive"]:
        config = get_comprehensive_config()
        print("Running COMPREHENSIVE study (70 experiments)")
    else:
        config = ABLATION_CONFIG
        print("Running DEFAULT configuration")
    
    print(f"Lambda values: {config['lambda_values']}")
    print(f"Seeds per lambda: {config['num_seeds']}")
    print(f"Total experiments: {len(config['lambda_values']) * config['num_seeds']}")
    print(f"Training timesteps: {config['total_timesteps']:,}")
    print(f"Evaluation episodes: {config['num_eval_episodes']}")
    
    # Confirm before starting
    if mode not in ["--quick", "quick"]:
        confirm = input("\nProceed? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Run the study
    try:
        print("\nStarting ablation study...")
        results_df, agg_results, shannon_bound, goal_analysis = run_rate_distortion_ablation(config)
        print("\nAblation study completed successfully!")
        
        # Quick analysis
        print(f"\nQuick Analysis:")
        print(f"Shannon bound: {shannon_bound:.3f} bits")
        print(f"Best rate: {agg_results['Rate_Mean'].min():.3f} bits")
        print(f"Best success rate: {agg_results['Success_Rate_Mean'].max():.3f}")
        
        # Fixed: Use goal_analysis instead of goal_analysis_df
        if goal_analysis is not None and not goal_analysis.empty and 'frequency' in goal_analysis.columns and 'bits' in goal_analysis.columns:
            try:
                freq_bits_corr, p_val = pearsonr(goal_analysis['frequency'], goal_analysis['bits'])
                print(f"Goal encoding correlation: r = {freq_bits_corr:.3f} (p = {p_val:.4f})")
            except Exception as e:
                print(f"Could not calculate correlation: {e}")
                freq_bits_corr, p_val = None, None
        else:
            freq_bits_corr, p_val = None, None
            
    except KeyboardInterrupt:
        print("\nStudy interrupted by user.")
        # Added proper handling for script execution
        sys.exit(1)
    except Exception as e:
        print(f"\nError during study: {e}")
        import traceback
        traceback.print_exc()
        # Added proper handling for script execution
        sys.exit(1)
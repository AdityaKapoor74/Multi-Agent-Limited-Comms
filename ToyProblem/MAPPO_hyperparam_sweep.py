import os
# Fix for the OMP: Error #15, which is common on macOS. This is a standard workaround.
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import gymnasium as gym
from collections import deque
import pandas as pd
import itertools
import random

# Import the custom environment
from CommunicatingGoalEnv import CommunicatingGoalEnv

def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Hyperparameter Sweep Configuration ---
sweep_config = {
    "ddcl_delta": [1/5, 1/10, 1/15],
    "learning_rate": [1e-4, 3e-4, 5e-4], 
    "lambda_comms": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "update_epochs": [5, 10, 15]
}
# sweep_config = {
#     "ddcl_delta": [1/5],
#     "learning_rate": [5e-4], 
#     "lambda_comms": [4e-3],
#     "update_epochs": [10]
# }

# Fixed hyperparameters
base_config = {
    "total_timesteps": 2_500_000,
    "num_envs": 16,
    "num_steps": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "minibatch_size": 16 * 256, # Use 1 mini-batch as recommended by the MAPPO paper
    "grid_size": 8,
    "z_dim": 4,
    "wandb_project": "ddcl_optimal_coding_sweep",
    "log_frequency": 4096, # Log performance metrics every X global steps
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- PopArt Value Normalization Class ---
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

# --- DDCL Channel Implementation ---
class DDCLChannel(nn.Module):
    def __init__(self, delta, lambda_comms=1.0):
        super().__init__()
        self.delta = delta
        self.lambda_comms = lambda_comms
        
    def forward(self, z):
        # Proper DDCL noise addition with uniform distribution
        if self.lambda_comms == 0.0:
            return z
        noise = (torch.rand_like(z) - 0.5) * self.delta
        return z + noise
    
    def calculate_loss_from_z(self, z):
        # Expected message length penalty
        return torch.log2(2 * torch.abs(z) / self.delta + 1.0).mean()
    
    def calculate_total_bits_from_z(self, z):
        return torch.log2(2 * torch.abs(z) / self.delta + 1.0).sum(dim=-1)

# --- Network Initialization ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- Networks ---
class SpeakerNetwork(nn.Module):
    def __init__(self, obs_dim, z_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 16)), 
            nn.GELU(),
            # layer_init(nn.Linear(32, 32)), 
            # nn.GELU(),
            layer_init(nn.Linear(16, z_dim), std=0.01)
        )

    def forward(self, x):
        return self.network(x)  # Direct z output

class ListenerActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 16)), 
            nn.GELU(), 
            # layer_init(nn.Linear(64, 64)), 
            # nn.GELU()
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
            # layer_init(nn.Linear(64, 64)), 
            # nn.GELU(), 
            layer_init(nn.Linear(32, 1), std=1.0)
        )
    
    def forward(self, x): 
        return self.network(x)
    
# --- Environment Creation with Proper Closure Fix ---
def make_env(config, env_seed):
    """Create a single environment with proper seed handling"""
    def _init():
        env = CommunicatingGoalEnv(
            grid_size=config["grid_size"], 
            z_dim=config["z_dim"],
            goal_sampling_mode='non_uniform',
            goal_frequencies=config["goal_frequencies"]
        )
        env.action_space.seed(env_seed)
        return env
    return _init

'''
# --- Post-Training Analysis ---
def analyze_and_visualize(speaker_network, ddcl_channel, env_config, wandb_run):
    print("\n--- Starting Post-Training Analysis ---")
    
    # Set up matplotlib for high-quality plots suitable for papers
    plt.rcParams.update({
        'font.size': 20,           # Base font size
        'axes.titlesize': 30,      # Title font size
        'axes.labelsize': 25,      # Axis label font size
        'xtick.labelsize': 25,     # X-axis tick label size
        'ytick.labelsize': 25,     # Y-axis tick label size
        'legend.fontsize': 20,     # Legend font size
        'figure.titlesize': 30,    # Figure title size
        'lines.linewidth': 2,      # Line width
        'axes.linewidth': 1.5,     # Axes border width
        'font.family': 'serif',    # Font family for academic papers
        'text.usetex': False,      # Set to True if you have LaTeX installed
    })
    
    sns.set_theme(style="whitegrid", font_scale=1.3)
    grid_size = env_config["grid_size"]
    device = env_config["device"]
    
    # Generate all possible goal locations
    all_goals = np.array([[x, y] for y in range(grid_size) for x in range(grid_size)])
    # all_goals_normalized = torch.FloatTensor(all_goals / (grid_size - 1)).to(device)
    all_goals_normalized = torch.FloatTensor(all_goals).to(device)
    
    with torch.no_grad():
        # Get deterministic speaker output for analysis
        z_vectors = speaker_network(all_goals_normalized)
        comm_bits = ddcl_channel.calculate_total_bits_from_z(z_vectors).cpu().numpy()
        
        # Create heatmap of communication costs
        cost_grid = comm_bits.reshape(grid_size, grid_size)
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        heatmap = sns.heatmap(cost_grid, annot=True, fmt=".2f", cmap="magma_r", 
                             linewidths=0.5, ax=ax, annot_kws={'size': 14})
        ax.set_title("Learned Communication Bits per Goal Position", fontsize=30, pad=20)
        ax.set_xlabel("X Coordinate", fontsize=25)
        ax.set_ylabel("Y Coordinate", fontsize=25)
        
        # Improve colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Communication Bits', fontsize=18, rotation=270, labelpad=25)
        
        plt.tight_layout()
        
        # Save plot offline
        plt.savefig('communication_bits_heatmap.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig('communication_bits_heatmap.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Log to wandb
        wandb_run.log({"analysis/communication_bits_heatmap": wandb.Image(fig)})
        plt.close()
        print("Saved and logged communication bits heatmap (PDF and PNG).")
        
        # Handle goal frequencies properly
        if "goal_frequencies" in env_config:
            goal_freq_map = env_config["goal_frequencies"]
            analysis_data = []
            
            for i, goal in enumerate(all_goals):
                goal_tuple = (goal[0], goal[1])
                if goal_tuple in goal_freq_map:
                    analysis_data.append({
                        "goal": f"({int(goal_tuple[0])}, {int(goal_tuple[1])})",
                        "frequency": goal_freq_map[goal_tuple],
                        "cost": comm_bits[i]
                    })
            
            if analysis_data:
                df = pd.DataFrame(analysis_data).sort_values("frequency", ascending=False)
                freq_bins = pd.qcut(df['frequency'], q=3, labels=['Low', 'Medium', 'High'], 
                                  duplicates='drop')
                df['freq_category'] = freq_bins
                
                fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
                
                # Create bar plot with larger fonts
                bars = sns.barplot(x='goal', y='cost', data=df, hue='freq_category',
                                 palette={'Low': '#4575b4', 'Medium': '#fee090', 'High': '#d73027'},
                                 dodge=False, ax=ax)
                
                ax.set_title("Communication Bits per Goal Location\n(Sorted by Goal Frequency)", 
                           fontsize=30, pad=20)
                ax.set_xlabel("Goal Location (Sorted by Frequency)", fontsize=20)
                ax.set_ylabel("Communication Bits", fontsize=25)
                
                # Improve legend
                legend = ax.legend(title='Frequency Category', fontsize=18, title_fontsize=20,
                                 loc='upper left', frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right', fontsize=12)
                ax.tick_params(axis='y', labelsize=14)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                
                # Save plot offline
                plt.savefig('bits_vs_frequency_barchart.pdf', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.savefig('bits_vs_frequency_barchart.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                # Log to wandb
                wandb_run.log({"analysis/bits_vs_frequency_barchart": wandb.Image(fig)})
                plt.close()
                print("Saved and logged sorted bits vs. frequency bar chart (PDF and PNG).")
                
                # Calculate correlation
                correlation, p_value = pearsonr(df["frequency"], df["cost"])
                wandb_run.summary["analysis/correlation_freq_vs_cost"] = correlation
                wandb_run.summary["analysis/correlation_p_value"] = p_value
                print(f"Pearson Correlation between frequency and cost: {correlation:.4f} (p={p_value:.4f})")
                
                # Create scatter plot for correlation visualization
                fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
                
                scatter = ax.scatter(df["frequency"], df["cost"], 
                                   c=df['freq_category'].map({'Low': '#4575b4', 'Medium': '#fee090', 'High': '#d73027'}),
                                   s=100, alpha=0.7, edgecolors='black', linewidth=1)
                
                # Add trend line
                z = np.polyfit(df["frequency"], df["cost"], 1)
                p = np.poly1d(z)
                ax.plot(df["frequency"], p(df["frequency"]), "r--", alpha=0.8, linewidth=2)
                
                ax.set_title(f"Communication Bits vs Goal Frequency\n(Correlation: r = {correlation:.3f})", 
                           fontsize=30, pad=20)
                ax.set_xlabel("Goal Frequency", fontsize=25)
                ax.set_ylabel("Communication Bits", fontsize=25)
                
                # Create custom legend for categories
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#4575b4', label='Low Frequency'),
                                 Patch(facecolor='#fee090', label='Medium Frequency'),
                                 Patch(facecolor='#d73027', label='High Frequency')]
                ax.legend(handles=legend_elements, fontsize=18, title='Frequency Category', 
                         title_fontsize=20, loc='best')
                
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                
                # Save scatter plot offline
                plt.savefig('frequency_cost_correlation.pdf', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.savefig('frequency_cost_correlation.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                # Log to wandb
                wandb_run.log({"analysis/frequency_cost_correlation": wandb.Image(fig)})
                plt.close()
                print("Saved and logged frequency vs cost correlation plot (PDF and PNG).")
    
    # Reset matplotlib parameters to default after plotting
    plt.rcParams.update(plt.rcParamsDefault)
    
    print("--- Analysis complete. All plots saved offline and logged to wandb ---")
'''

def analyze_and_visualize(speaker_network, ddcl_channel, env_config, wandb_run):
    print("\n--- Starting Post-Training Analysis ---")
    
    # Set up matplotlib for high-quality plots suitable for papers
    plt.rcParams.update({
        'font.size': 16,           # Base font size
        'axes.titlesize': 20,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 14,     # X-axis tick label size
        'ytick.labelsize': 14,     # Y-axis tick label size
        'legend.fontsize': 14,     # Legend font size
        'figure.titlesize': 20,    # Figure title size
        'lines.linewidth': 2,      # Line width
        'axes.linewidth': 1.5,     # Axes border width
        'font.family': 'sans-serif', # More universally available
        'text.usetex': False,      # Set to True if you have LaTeX installed
    })
    
    sns.set_theme(style="whitegrid", font_scale=1.0)
    grid_size = env_config["grid_size"]
    device = env_config["device"]
    
    # Generate all possible goal locations in proper order
    # Create goals in (x,y) format matching environment expectations
    all_goals = []
    for y in range(grid_size):
        for x in range(grid_size):
            all_goals.append([x, y])
    all_goals = np.array(all_goals)
    all_goals_tensor = torch.FloatTensor(all_goals).to(device)
    
    with torch.no_grad():
        # Get deterministic speaker output for analysis
        z_vectors = speaker_network(all_goals_tensor)
        comm_bits = ddcl_channel.calculate_total_bits_from_z(z_vectors).cpu().numpy()
        
        # Create heatmap of communication costs
        # Reshape to match grid: rows are y-coordinates (top to bottom), cols are x-coordinates
        cost_grid = comm_bits.reshape(grid_size, grid_size)
        
        # Create masks for goal vs non-goal positions
        goal_mask = np.zeros((grid_size, grid_size), dtype=bool)
        goal_freq_grid = np.zeros((grid_size, grid_size))
        goal_prob_grid = np.zeros((grid_size, grid_size))
        
        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            goal_freq_map = env_config["goal_frequencies"]
            total_frequency = sum(goal_freq_map.values())
            
            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    goal_mask[y, x] = True  # Note: y first for grid indexing
                    goal_freq_grid[y, x] = freq
                    goal_prob_grid[y, x] = freq / total_frequency
        
        # Create the heatmap with proper orientation
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        # Create a modified colormap that fades non-goal positions
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as patches
        
        # Base heatmap with all positions
        heatmap = sns.heatmap(
            cost_grid, 
            annot=False,  # We'll add custom annotations
            cmap="viridis_r",
            linewidths=0.5, 
            ax=ax, 
            cbar_kws={'label': 'Communication Bits'},
            square=True,
            xticklabels=range(grid_size),
            yticklabels=range(grid_size),
            alpha=0.3  # Make everything semi-transparent initially
        )
        
        # Overlay highlighted goal positions
        for y in range(grid_size):
            for x in range(grid_size):
                if goal_mask[y, x]:
                    # Highlight goal positions with full opacity
                    rect = patches.Rectangle((x, y), 1, 1, 
                                           linewidth=3, edgecolor='red', 
                                           facecolor=plt.cm.viridis_r(cost_grid[y, x] / cost_grid.max()),
                                           alpha=1.0)
                    ax.add_patch(rect)
                    
                    # Add custom annotation with bits and frequency info
                    bits_text = f'{cost_grid[y, x]:.2f}'
                    freq_text = f'f={int(goal_freq_grid[y, x])}'
                    prob_text = f'p={goal_prob_grid[y, x]:.3f}'
                    
                    # Multi-line annotation
                    annotation = f'{bits_text}\n{freq_text}\n{prob_text}'
                    
                    ax.text(x + 0.5, y + 0.5, annotation, 
                           ha='center', va='center', 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8, edgecolor='red'))
                else:
                    # Add faded annotation for non-goal positions
                    ax.text(x + 0.5, y + 0.5, f'{cost_grid[y, x]:.2f}', 
                           ha='center', va='center', 
                           fontsize=8, alpha=0.6, color='gray')
        
        ax.set_title("Communication Bits: Goal Positions vs Non-Goal Positions", fontsize=18, pad=15)
        ax.set_xlabel("X Coordinate", fontsize=16)
        ax.set_ylabel("Y Coordinate", fontsize=16)
        
        # Add legend explaining the annotations
        legend_text = ("Red boxes: Goal positions\n"
                      "Top: Communication bits\n"
                      "Middle: Frequency (f)\n"
                      "Bottom: Probability (p)\n"
                      "Faded: Non-goal positions")
        ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Invert y-axis to match typical coordinate convention (origin at bottom-left)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot offline
        plt.savefig('communication_bits_heatmap.pdf', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig('communication_bits_heatmap.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Log to wandb
        wandb_run.log({"analysis/communication_bits_heatmap": wandb.Image(fig)})
        plt.close()
        print("Saved and logged communication bits heatmap (PDF and PNG).")
        
        # Create a separate, cleaner visualization focusing just on goal positions
        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
            
            # Left plot: Communication bits for all positions with goal highlights
            im1 = ax1.imshow(cost_grid, cmap='viridis_r', alpha=0.3)
            ax1.set_title("Communication Bits (All Positions)", fontsize=16)
            
            # Right plot: Goal frequencies
            freq_grid = np.full((grid_size, grid_size), np.nan)
            goal_freq_map = env_config["goal_frequencies"]
            total_frequency = sum(goal_freq_map.values())
            
            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    freq_grid[y, x] = freq / total_frequency  # Convert to probability
            
            # Mask for showing only goal positions
            masked_freq = np.ma.masked_where(np.isnan(freq_grid), freq_grid)
            im2 = ax2.imshow(masked_freq, cmap='Reds', vmin=0, vmax=masked_freq.max())
            ax2.set_title("Goal Sampling Probabilities", fontsize=16)
            
            # Add annotations and highlights for both plots
            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    prob = freq / total_frequency
                    
                    # Left plot: highlight goal with communication cost
                    rect1 = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                            linewidth=3, edgecolor='red', 
                                            facecolor='none')
                    ax1.add_patch(rect1)
                    ax1.text(x, y, f'{cost_grid[y, x]:.2f}\n({freq})', 
                            ha='center', va='center', fontweight='bold', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    # Right plot: show probability
                    ax2.text(x, y, f'{prob:.3f}\n(f={freq})', 
                            ha='center', va='center', fontweight='bold', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Format both plots
            for ax in [ax1, ax2]:
                ax.set_xlim(-0.5, grid_size-0.5)
                ax.set_ylim(-0.5, grid_size-0.5)
                ax.set_xticks(range(grid_size))
                ax.set_yticks(range(grid_size))
                ax.set_xlabel("X Coordinate", fontsize=14)
                ax.set_ylabel("Y Coordinate", fontsize=14)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
            
            # Add colorbars
            plt.colorbar(im1, ax=ax1, label='Communication Bits', shrink=0.8)
            plt.colorbar(im2, ax=ax2, label='Sampling Probability', shrink=0.8)
            
            plt.suptitle("Goal Position Analysis", fontsize=18, y=1.02)
            plt.tight_layout()
            
            # Save goal-focused visualization
            plt.savefig('goal_analysis_comparison.pdf', dpi=300, bbox_inches='tight')
            plt.savefig('goal_analysis_comparison.png', dpi=300, bbox_inches='tight')
            wandb_run.log({"analysis/goal_analysis_comparison": wandb.Image(fig)})
            plt.close()
            print("Saved and logged goal analysis comparison.")
        
        # Handle goal frequencies analysis - only for goals that have defined frequencies
        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            goal_freq_map = env_config["goal_frequencies"]
            analysis_data = []
            
            # Only analyze goals that have defined frequencies
            for goal_tuple, frequency in goal_freq_map.items():
                # Find the index of this goal in our all_goals array
                goal_idx = None
                for i, goal in enumerate(all_goals):
                    if goal[0] == goal_tuple[0] and goal[1] == goal_tuple[1]:
                        goal_idx = i
                        break
                
                if goal_idx is not None:
                    analysis_data.append({
                        "goal_x": int(goal_tuple[0]),
                        "goal_y": int(goal_tuple[1]),
                        "goal_label": f"({int(goal_tuple[0])}, {int(goal_tuple[1])})",
                        "frequency": frequency,
                        "cost": comm_bits[goal_idx]
                    })
            
            if len(analysis_data) >= 3:  # Need at least 3 points for meaningful analysis
                df = pd.DataFrame(analysis_data).sort_values("frequency", ascending=False)
                
                # Create frequency categories using actual data distribution
                if len(df) >= 3:
                    # Use quantiles if we have enough data points
                    try:
                        freq_bins = pd.qcut(df['frequency'], q=min(3, len(df)), 
                                          labels=['Low', 'Medium', 'High'][:min(3, len(df))], 
                                          duplicates='drop')
                    except ValueError:
                        # If qcut fails due to duplicate edges, use simple thresholds
                        freq_median = df['frequency'].median()
                        freq_75 = df['frequency'].quantile(0.75)
                        freq_bins = pd.cut(df['frequency'], 
                                         bins=[0, freq_median, freq_75, float('inf')],
                                         labels=['Low', 'Medium', 'High'],
                                         include_lowest=True)
                else:
                    freq_bins = ['Medium'] * len(df)  # Default category for small datasets
                
                df['freq_category'] = freq_bins
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
                
                # Create color palette
                colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
                bar_colors = [colors.get(cat, '#95a5a6') for cat in df['freq_category']]
                
                bars = ax.bar(range(len(df)), df['cost'], color=bar_colors, alpha=0.8, edgecolor='black')
                
                ax.set_title("Communication Bits per Goal Location\n(Sorted by Goal Frequency)", 
                           fontsize=18, pad=15)
                ax.set_xlabel("Goal Location (Sorted by Frequency)", fontsize=16)
                ax.set_ylabel("Communication Bits", fontsize=16)
                
                # Set x-axis labels
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['goal_label'], rotation=45, ha='right')
                
                # Create legend
                unique_categories = df['freq_category'].unique()
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(cat, '#95a5a6'), 
                                               label=f'{cat} Frequency') for cat in unique_categories]
                ax.legend(handles=legend_elements, loc='upper right')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                
                # Save plot offline
                plt.savefig('bits_vs_frequency_barchart.pdf', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.savefig('bits_vs_frequency_barchart.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                # Log to wandb
                wandb_run.log({"analysis/bits_vs_frequency_barchart": wandb.Image(fig)})
                plt.close()
                print("Saved and logged sorted bits vs. frequency bar chart (PDF and PNG).")
                
                # Calculate correlation
                correlation, p_value = pearsonr(df["frequency"], df["cost"])
                wandb_run.summary["analysis/correlation_freq_vs_cost"] = correlation
                wandb_run.summary["analysis/correlation_p_value"] = p_value
                print(f"Pearson Correlation between frequency and cost: {correlation:.4f} (p={p_value:.4f})")
                
                # Scatter plot for correlation visualization
                fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
                
                # Create scatter plot with category colors
                for category in unique_categories:
                    mask = df['freq_category'] == category
                    if mask.any():
                        ax.scatter(df[mask]["frequency"], df[mask]["cost"], 
                                 c=colors.get(category, '#95a5a6'), 
                                 label=f'{category} Frequency',
                                 s=100, alpha=0.7, edgecolors='black', linewidth=1)
                
                # Add trend line if we have enough points
                if len(df) > 2:
                    z = np.polyfit(df["frequency"], df["cost"], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df["frequency"].min(), df["frequency"].max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
                
                ax.set_title(f"Communication Bits vs Goal Frequency\n(Correlation: r = {correlation:.3f})", 
                           fontsize=18, pad=15)
                ax.set_xlabel("Goal Frequency", fontsize=16)
                ax.set_ylabel("Communication Bits", fontsize=16)
                
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)
                
                plt.tight_layout()
                
                # Save scatter plot offline
                plt.savefig('frequency_cost_correlation.pdf', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.savefig('frequency_cost_correlation.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                # Log to wandb
                wandb_run.log({"analysis/frequency_cost_correlation": wandb.Image(fig)})
                plt.close()
                print("Saved and logged frequency vs cost correlation plot (PDF and PNG).")
                
            else:
                print(f"Insufficient data for frequency analysis: only {len(analysis_data)} goals with defined frequencies")
        else:
            print("No goal frequencies defined - skipping frequency analysis")
        
        # Additional analysis: Show distribution of communication costs
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.hist(comm_bits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Communication Bits Across All Goals", fontsize=18)
        ax.set_xlabel("Communication Bits", fontsize=16)
        ax.set_ylabel("Number of Goals", fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {comm_bits.mean():.2f}\nStd: {comm_bits.std():.2f}\nMin: {comm_bits.min():.2f}\nMax: {comm_bits.max():.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('communication_bits_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('communication_bits_distribution.png', dpi=300, bbox_inches='tight')
        wandb_run.log({"analysis/communication_bits_distribution": wandb.Image(fig)})
        plt.close()
        print("Saved and logged communication bits distribution.")
    
    # Reset matplotlib parameters to default after plotting
    plt.rcParams.update(plt.rcParamsDefault)
    
    print("--- Analysis complete. All plots saved offline and logged to wandb ---")
    

# --- Single Training Run Function ---
def run_single_experiment(config):

    set_seeds(1)

    device = torch.device(config["device"])
    batch_size = config["num_envs"] * config["num_steps"]
    
    # Define goal frequencies properly
    goal_frequencies = {
        (0, 0): 200, 
        (7, 7): 100, 
        (3, 4): 50, 
        (4, 3): 25, 
        (1, 6): 12, 
        (6, 1): 1
    }
    config["goal_frequencies"] = goal_frequencies
    
    # Create non-uniform goal sampling environments
    # env_fns = [lambda: CommunicatingGoalEnv(
    #     grid_size=config["grid_size"], 
    #     z_dim=config["z_dim"],
    #     goal_sampling_mode='non_uniform',
    #     goal_frequencies=config["goal_frequencies"]
    # ) for _ in range(config["num_envs"])]

    # envs = gym.vector.SyncVectorEnv(env_fns)

    # Create non-uniform goal sampling environments with proper closure fix
    env_fns = [make_env(config, 4+i) for i in range(config["num_envs"])]
    envs = gym.vector.SyncVectorEnv(env_fns)
    
    # Initialize networks
    speaker_network = SpeakerNetwork(2, config["z_dim"]).to(device)
    listener_actor = ListenerActor(2 + config["z_dim"], 5).to(device)
    critic = Critic(2 + 2).to(device)  # Global state: listener_pos + goal_pos
    ddcl_channel = DDCLChannel(config["ddcl_delta"], config["lambda_comms"]).to(device)
    popart = PopArt(input_shape=(), device=device)
    
    # Single optimizer for stable training
    actor_optimizer = torch.optim.Adam(
        list(speaker_network.parameters()) + list(listener_actor.parameters()), 
        lr=config["learning_rate"], eps=1e-5
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), 
        lr=config["learning_rate"], eps=1e-5
    )
    
    # Storage buffers - only need listener action storage for PPO
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

    for update in tqdm(range(1, num_updates + 1), desc=f"Training (lr={config['learning_rate']}, delta={config['ddcl_delta']}, lambda={config['lambda_comms']}, epochs={config['update_epochs']})"):
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
            if global_step - last_log_step >= config["log_frequency"]:
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

        # Proper advantage calculation
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
        
        # Training loop with single optimizer for stability
        inds = np.arange(batch_size)
        for epoch in range(config["update_epochs"]):
            if config["minibatch_size"] < batch_size:
                np.random.shuffle(inds)
            
            for start in range(0, batch_size, config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = inds[start:end]
                
                # Forward pass through Speaker to get z encoding
                z_current = speaker_network(b_obs_s[mb_inds])
                
                # Communication loss (encourages efficient encoding) - INCREASED PENALTY
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
                
                # Speaker learns from Listener's gradients + HIGHER communication cost penalty
                actor_loss = (pg_loss_l  # Listener PPO loss (gradients flow to Speaker)
                             - config["entropy_coef"] * entropy_l  # Only listener entropy
                             + config["lambda_comms"] * comm_loss)  # INCREASED penalty
                
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
        
        # Log training metrics
        wandb.log({
            "global_step": global_step,
            "losses/value_loss": v_loss.item(),
            "losses/listener_pg_loss": pg_loss_l.item(),
            "losses/comm_loss_term": comm_loss.item(),
            "rollout/communication_bits": ddcl_channel.calculate_total_bits_from_z(b_z).mean().item(),
            "rollout/z_magnitude": torch.norm(z_vectors, dim=-1).mean().item(),
            "rollout/listener_entropy": entropy_l.item()
        })
    
    # Post-training analysis
    analyze_and_visualize(speaker_network, ddcl_channel, config, wandb.run)

    envs.close()

# --- Main Sweep Script ---
if __name__ == "__main__":
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(
        sweep_config["ddcl_delta"],
        sweep_config["learning_rate"], 
        sweep_config["lambda_comms"],
        sweep_config["update_epochs"]
    ))
    
    print(f"Starting hyperparameter sweep with {len(param_combinations)} combinations")
    
    for i, (ddcl_delta, learning_rate, lambda_comms, update_epochs) in enumerate(param_combinations):
        print(f"\n=== Run {i+1}/{len(param_combinations)} ===")
        print(f"ddcl_delta: {ddcl_delta}")
        print(f"learning_rate: {learning_rate}")
        print(f"lambda_comms: {lambda_comms}")
        print(f"update_epochs: {update_epochs}")
        
        # Create config for this run
        config = base_config.copy()
        config.update({
            "ddcl_delta": ddcl_delta,
            "learning_rate": learning_rate,
            "lambda_comms": lambda_comms,
            "update_epochs": update_epochs
        })
        
        # Create run name
        run_name = f"sweep_delta_{ddcl_delta:.3f}_lr_{learning_rate:.0e}_lambda_{lambda_comms:.0e}_epochs_{update_epochs}"
        
        # Initialize wandb run
        wandb_run = wandb.init(
            project=config["wandb_project"], 
            config=config, 
            name=run_name,
            reinit=True
        )
        
        try:
            # Run single experiment
            run_single_experiment(config)
            print(f"Completed run {i+1}/{len(param_combinations)} successfully")
            
        except Exception as e:
            print(f"Error in run {i+1}/{len(param_combinations)}: {str(e)}")
            
        finally:
            # Clean up wandb run
            wandb_run.finish()
    
    print("\n--- Hyperparameter Sweep Finished ---")
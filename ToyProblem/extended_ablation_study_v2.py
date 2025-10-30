import os

from train_utils import train_mappo_agent

# Fix for the OMP: Error #15, which is common on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import wandb
from scipy.stats import pearsonr
import pandas as pd
import json
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

# Import the custom environment
from CommunicatingGoalEnv import CommunicatingGoalEnv

# Import utility modules
from plotting_utils import create_plots
from analysis_utils import analyze_goal_encoding, calculate_theoretical_bounds
from test_configs import get_quick_test_config, get_basic_test_config, get_extended_test_config, \
    get_comprehensive_config

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
    "ddcl_delta": 1 / 5.0,

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
    "enable_wandb": False,
    "wandb_project": "ddcl_rate_distortion_basic",
    "save_dir": Path("./results/rate_distortion_results")
}


# --- Core Training Function ---
def train_agent(lambda_comms, seed, config):
    """Train a single agent with given lambda_comms value"""

    # Use shared training function with extended ablation configuration
    return train_mappo_agent(
        config=config,
        lambda_comms=lambda_comms,
        seed=seed,
        enable_logging=False,  # Extended ablation doesn't use wandb logging during training
        wandb_run=None,
        return_convergence_data=config.get("enable_convergence_tracking", False)
    )


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

    # Initialize wandb only if enabled
    if config.get("enable_wandb", True):
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

            # Log to wandb if enabled
            if config.get("enable_wandb", True):
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

    # Log final results to wandb if enabled
    if config.get("enable_wandb", True):
        wandb.log({
            'final/theoretical_entropy': theoretical_entropy,
            'final/best_empirical_rate': best_rate,
            'final/min_shannon_gap': min_shannon_gap,
            'final/max_success_rate': best_success,
            'final/total_runtime_hours': total_runtime_hours
        })

    # Finish wandb logging if enabled
    if config.get("enable_wandb", True):
        wandb.finish()

    return results_df, agg_results, theoretical_entropy, goal_analysis_df


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

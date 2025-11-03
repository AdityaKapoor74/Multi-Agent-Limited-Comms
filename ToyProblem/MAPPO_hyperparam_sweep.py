import os

from analysis_utils import analyze_and_visualize
from train_utils import train_mappo_agent

# Fix for the OMP: Error #15, which is common on macOS. This is a standard workaround.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import wandb
import itertools

sweep_config = {
    "ddcl_delta": [1 / 5, 1 / 10, 1 / 15],
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "lambda_comms": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "update_epochs": [5, 10, 15]
}

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
    "minibatch_size": 16 * 256,  # Use 1 mini-batch as recommended by the MAPPO paper
    "grid_size": 8,
    "z_dim": 4,
    "wandb_project": "ddcl_optimal_coding_sweep",
    "log_frequency": 4096,  # Log performance metrics every X global steps
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# --- Single Training Run Function ---
def run_single_experiment(config):
    # Define goal frequencies for MAPPO hyperparam sweep
    goal_frequencies = {
        (0, 0): 200,
        (7, 7): 100,
        (3, 4): 50,
        (4, 3): 25,
        (1, 6): 12,
        (6, 1): 1
    }

    # Train using shared training function
    speaker_network, listener_actor, critic, ddcl_channel, _ = train_mappo_agent(
        config=config,
        goal_frequencies=goal_frequencies,
        enable_logging=True,
        wandb_run=wandb.run,
        seed=1
    )

    # Post-training analysis
    analyze_and_visualize(speaker_network, ddcl_channel, config, wandb.run)


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
        print(f"\n=== Run {i + 1}/{len(param_combinations)} ===")
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
            print(f"Completed run {i + 1}/{len(param_combinations)} successfully")

        except Exception as e:
            print(f"Error in run {i + 1}/{len(param_combinations)}: {str(e)}")

        finally:
            # Clean up wandb run
            wandb_run.finish()

    print("\n--- Hyperparameter Sweep Finished ---")

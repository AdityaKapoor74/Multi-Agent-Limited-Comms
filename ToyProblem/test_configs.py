from pathlib import Path

import torch

# Base configuration - duplicated here to avoid circular imports
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
    "enable_wandb": False,  
    "wandb_project": "ddcl_rate_distortion_basic",
    "save_dir": Path("./results/rate_distortion_results")
}


def get_quick_test_config():
    """Configuration for quick testing (3 experiments, ~30 minutes)"""
    config = ABLATION_CONFIG.copy()
    config.update({
        'lambda_values': [0.0, 1e-3, 5e-3],  # Only 3 lambdas
        'num_seeds': 1,  # Single seed for speed
        'total_timesteps': 500_000,  # Reduced training
        'num_eval_episodes': 200,  # Fewer evaluations
        'enable_wandb': False,  # Disable wandb for quick testing
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
        'enable_wandb': False,  # Disable wandb for basic testing
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
        'enable_wandb': False,  # Disable wandb for extended testing
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
        'enable_wandb': False,  # Disable wandb for comprehensive testing
        'wandb_project': 'ddcl_comprehensive_study'
    })
    return config
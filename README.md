# Differentiable Discrete Communication Learning (DDCL)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01554-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.01554)

<!-- Placeholder for paper overview and description -->
This repository contains the official implementation for the paper **"Learning What to Say and How Precisely: Efficient Communication via Differentiable Discrete Communication Learning"**. 

## Overview

Our codebase builds on top of two foundational codebases:
1. [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://github.com/marlbenchmark/on-policy) - Our `MAPPO_Transformer` implementation is based on the MAPPO framework. The transformer policy implementation can be found in `onpolicy/algorithms/utils/transformer_encoder.py`, which also has the DDCL and FakeQuant extensions integrated.
2. [Multi-Agent Graph Communication and Teaming](https://github.com/CORE-Robotics-Lab/MAGIC) - We integrate DDCL with scheduling-based algorithms implemented in the MAGIC codebase, extending their graph-attention communication approach with our novel differentiable discrete communication learning methodology.

## Supported Environments

Our implementation focuses on three popular and challenging multi-agent environments that require sophisticated coordination and communication strategies: Traffic Junction, Predator Prey, and Google Research Football. Additionally, we provide ToyProblem, a controlled testing environment specifically designed for systematic analysis and validation of DDCL.

### Traffic Junction
Traffic Junction environments simulate urban intersection scenarios where multiple autonomous agents must coordinate their movements to avoid collisions while efficiently navigating through shared spaces. These environments test the ability of agents to communicate spatial intentions and negotiate right-of-way in real-time, making them ideal testbeds for learning concise yet informative communication protocols. 

### Predator Prey  
The Predator Prey environments create cooperative hunting scenarios where a team of predator agents must collaborate to capture faster prey agents in bounded environments. Success requires sophisticated coordination strategies, spatial reasoning, and dynamic role assignment, all of which benefit significantly from learned communication that can convey tactical information and coordinate synchronized actions.

### Google Research Football
Google Research Football provides a complex, competitive multi-agent environment that mirrors real-world football gameplay. Teams of agents must learn to pass, position, and coordinate their actions against opposing teams, requiring both strategic planning and reactive decision-making. The environment's complexity makes it particularly suitable for evaluating the scalability and effectiveness of communication learning approaches.

### ToyProblem
To build intuition for how our generalized DDCL framework enables agents to learn efficient communication, we first conduct a qualitative analysis in a controlled, interpretable environment. This environment is designed to enable systematic study of rate-distortion trade-offs in learned communication, and allows us to investigate the structure of the learned protocol itself.

The central component is `CommunicatingGoalEnv`, a multi-agent gym-style environment implementing a Speaker-Listener communication paradigm:

**Agent Roles:**
- **Speaker Agent**: A stationary agent that observes the goal location and must communicate this information through a continuous communication vector `z` of configurable dimensionality
- **Listener Agent**: A mobile agent that receives the Speaker's message and must navigate to the goal location in an N×N grid

**Environment Characteristics:**
- **Grid-based Navigation**: Configurable grid sizes (default 8×8 to 10×10) with discrete movement actions
- **Sparse Cooperative Rewards**: +1.0 when Listener reaches goal, -0.01 time penalty to encourage efficiency
- **Communication Constraints**: Continuous communication vectors with DDCL channel processing for discrete protocol emergence
- **Goal Distribution Control**: Configurable goal frequencies enabling non-uniform sampling for realistic scenarios

## Installation 
We strongly recommend using dedicated virtual environments to ensure consistent dependency management and avoid conflicts with other projects.

We used `pytorch_ubuntu` docker image for our PyTorch installations, but installation via package manager (conda / pip / uv) can also be done. For `MAPPO` codebase, we used PyTorch version `2.6.0` while for `MAGIC` codebase, we used PyTorch `2.4.1`, in accordance with the versions recommended by the corresponding official repositories.

If you face any issues in installing GRF, please refer to the official repository for more details: https://github.com/google-research/football

First, please clone this repository 
```
git clone --branch=iclr https://github.com/agent-lab/Multi-Agent-Limited-Comms.git
cd Multi-Agent-Limited-Comms
```

The directory structure looks like this
```
.
├── LICENSE
├── MAGIC #contains code for MAGIC/IC3Net/GAComm/TarMAC experiments
├── on-policy #contains code for MAPPO_Transformer experiments
├── README.md
├── setup.py
└── ToyProblem # contains code for DDCL analysis
```

### GRF for MAPPO

```
conda create -n marl_comms_mappo_grf python=3.9.2
conda activate marl_comms_mappo_grf

apt-get update

apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

python -m pip install --upgrade pip setuptools psutil wheel

pip3 install setuptools==65.5.0

pip3 install psutil==7.0.0

git clone https://github.com/google-research/football.git
cd football
python3 -m pip install .

pip3 install six

pip3 install visdom

pip3 install setproctitle

pip3 install torch

pip3 install wandb

pip3 install imageio

pip3 install tensorboardX
```

### GRF for MAGIC Codebase
```
conda create -n marl_comms_magic_grf python=3.9.2
conda activate marl_comms_magic_grf

sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip

git clone https://github.com/chrisyrniu/football.git
cd football
pip install .

cd envs/grf-envs
python setup.py develop
```

If pygame throws error then edit the setup.py file in `football` directory and comment the line `pygame==1.9.6`
```
install_requires=[
# 'pygame==1.9.6',
'opencv-python',
'scipy',
'gym>=0.11.0',
'absl-py',
]
```

### Traffic Junction and Predator Prey 
```
conda create -n marl_comms_tjpp python=3.9.2
conda activate marl_comms_tjpp

cd envs/ic3net-envs
python setup.py develop

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install absl-py==2.3.1

pip3 install numpy==1.23.5
```


### Core Package Installation

Install the on-policy package in development mode, which allows for easy modification and experimentation with the core algorithms:

```bash
# Navigate to the repository root and install the on-policy package
cd onpolicy
pip install -e .
```


## Usage

### MAPPO
All core algorithmic code for MAPPO resides within the `on-policy` folder. The `on-policy/onpolicy/algorithms/` subfolder houses algorithm implementation, with our DDCL extensions building upon the MAPPO foundation.

Environment wrapper implementations for Traffic Junction, Predator Prey, and Google Research Football can be found in the `on-policy/onpolicy/envs/` subfolder. Training orchestration and policy update logic are contained within the `on-policy/onpolicy/runner/` folder, with specialized runners designed for each supported environment type.

The `config.py` file serves as the central configuration hub, containing relevant hyperparameter settings and environment configurations. Refer to this file for detailed flag descriptions and default values.

Executable training scripts can be found in the `on-policy/onpolicy/scripts/` folder. These scripts follow a consistent naming convention: `train_algo_environment.sh`, making it straightforward to identify appropriate training configurations for specific experimental setups.

Training procedures follow a standardized approach across all environments, with environment-specific scripts handling the particular requirements of each domain. Here we demonstrate the training process using Google Research Football as an example:

```bash
# Navigate to the scripts directory
cd onpolicy/scripts/train_football_scripts

# Make the training script executable
chmod +x train_football_3v1.sh

# Execute training with default parameters
./train_football_3v1.sh
```

Training results and model checkpoints are automatically stored in the `results` subdirectory, organized by algorithm, environment, and experimental timestamp for easy identification. By default, all experiments assume a shared policy architecture where a single neural network is utilized by all agents in the environment. 


### MAGIC / GAComm / TarMAC / IC3Net

For running `MAGIC` experiments on GRF environment, here is a sample command
```
cd MAGIC
sh grf_scripts/train_grf_magic_comms.sh
```
We provide different scripts for different hyperparameter settings. For trying another environment, replace `grf` with the environment of choice (eg: `tj`, `pp`, etc.)

For running `GAComm` or `TarMAC` or `IC3Net` - here is a sample command for running `IC3Net` on `Predatory-Prey Hard` task.
```
cd MAGIC/baselines
sh pp_hard_scripts/train_pp_hard_ic3net.sh 
```
For trying another environment, replace `pp` with the environment of choice (eg: `tj`, `grf`, etc.). Similarly, for other algorithms, replace `ic3net` with the algorithm of choice (`gacomm`, `tarmac`, etc.)

The scripts with `fakequant` in their names indicate that they have Fake Quantization enabled. Similarly, scripts with `comms` indicate DDCL enabled.




### ToyProblem

**Basic Usage Example:**
```python
# Navigate to ToyProblem directory
cd ToyProblem

python extended_ablation_study_v2.py

python MAPPO_hyperparam_sweep.py
```

## Visualization and Monitoring

We utilize Weights & Biases as the default visualization platform for comprehensive experiment tracking and analysis. To use Weights & Biases integration, register for an account and authenticate your local installation following the official [Weights & Biases documentation](https://docs.wandb.ai/).

For users who prefer Tensorboard visualization, adding the `--use_wandb` flag to command line arguments or training scripts will redirect logging output to Tensorboard instead of Weights & Biases

<!-- ## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{ddcl2025,
  title={Learning What to Say and How Precisely: Efficient Communication via Differentiable Discrete Communication Learning},
  author={[Author Names - Placeholder]},
  journal={[Journal/Conference - Placeholder]},
  year={2025}
}
``` -->

## License 
This project is licensed under the MIT License - see the LICENSE file for details.

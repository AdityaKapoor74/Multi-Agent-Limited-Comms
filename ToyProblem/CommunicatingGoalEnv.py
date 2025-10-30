"""
Implements a multi-agent gym-style environment for testing Differentiable Discrete 
Communication Learning (DDCL). This environment is specifically designed to test an agent's 
ability to learn an efficient, sparse communication protocol.
"""
import collections

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CommunicatingGoalEnv(gym.Env):
    """
    A multi-agent environment where a 'Speaker' agent must communicate a goal location to a 
    'Listener' agent in an N x N grid.

    **Objective:**
    The Listener must navigate to the goal. The Speaker must learn to communicate the goal's
    location efficiently. The environment is designed such that communication is necessary
    for solving the task. The sparse reward structure and the communication cost (handled by the 
    training algorithm) should incentivize the agents to learn a sparse and optimal communication protocol.

    **Agent Roles:**
    - Agent 0 (Speaker): A stationary agent that observes the goal location. Its only action is 
      to produce a continuous communication vector `z`.
    - Agent 1 (Listener): A mobile agent that observes its own location. It must interpret the 
      Speaker's message to navigate to the goal.

    **State Space:**
    The global state is defined by the Listener's position and the Goal's position.

    **Observation Space:**
    - Speaker: Observes the goal's (x, y) coordinates, normalized to [0, 1].
    - Listener: Observes its own (x, y) coordinates, normalized to [0, 1].

    **Action Space:**
    - Speaker: A continuous vector `z` of shape `(z_dim,)` with values in [-1, 1].
    - Listener: A discrete action space with 5 actions (Stay, North, South, East, West).

    **Reward Structure:**
    A sparse, cooperative reward:
    - +1.0 for both agents when the Listener reaches the goal.
    - -0.01 for both agents for every other timestep (time penalty).

    **Episode Termination:**
    An episode ends when the Listener reaches the goal or the `max_steps` limit is reached.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, grid_size=10, max_steps=50, z_dim=10, dynamic_goal=False, goal_move_prob=0.05,
                 goal_sampling_mode='uniform', goal_frequencies=None):
        """
        Initializes the environment.

        Args:
            grid_size (int): The size of the square grid (N in an N x N grid).
            max_steps (int): The maximum number of steps per episode before truncation.
            z_dim (int): The dimensionality of the Speaker's communication vector `z`.
            dynamic_goal (bool): If True, the goal can move randomly during an episode.
            goal_move_prob (float): The probability of the goal moving at each step if dynamic_goal is True.
            goal_sampling_mode (str): Determines how goals are sampled at the start of an episode.
                                      Can be 'uniform' or 'non_uniform'.
            goal_frequencies (dict): A dictionary mapping goal tuples (x, y) to probability weights.
                                     Required if `goal_sampling_mode` is 'non_uniform'.
        """
        super(CommunicatingGoalEnv, self).__init__()

        # --- Environment Parameters ---
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.z_dim = z_dim
        self.dynamic_goal = dynamic_goal
        self.goal_move_prob = goal_move_prob
        self.goal_sampling_mode = goal_sampling_mode
        self.n_agents = 2

        # --- Agent Constants ---
        self.SPEAKER_IDX = 0
        self.LISTENER_IDX = 1

        # --- Episode State Variables ---
        self.listener_pos = None
        self.goal_pos = None
        self.current_step = 0

        # Define agent observation and action spaces
        self._define_spaces()

        # Pre-process frequencies for efficient non-uniform sampling if specified
        if self.goal_sampling_mode == 'non_uniform':
            if goal_frequencies is None:
                raise ValueError("`goal_frequencies` must be provided for 'non_uniform' sampling mode.")

            # Sort items for consistent ordering and convert to lists for sampling
            sorted_goals = sorted(goal_frequencies.items())
            self.goal_indices = [np.ravel_multi_index(goal, (grid_size, grid_size)) for goal, _ in sorted_goals]
            total_weight = sum(w for _, w in sorted_goals)
            self.goal_probs = [w / total_weight for _, w in sorted_goals]

    def _define_spaces(self):
        """Helper function to define observation and action spaces."""
        # --- Observation Spaces ---
        speaker_obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        listener_obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Tuple([speaker_obs_space, listener_obs_space])

        # --- Action Spaces ---
        speaker_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.z_dim,), dtype=np.float32)
        listener_action_space = spaces.Discrete(5)  # 0: Stay, 1: N, 2: S, 3: E, 4: W
        self.action_space = spaces.Tuple([speaker_action_space, listener_action_space])

    def _get_obs(self):
        """
        Computes the observations for both agents based on the current state.
        Coordinates are normalized to the range [0, 1].

        Returns:
            tuple: A tuple of observations for (Speaker, Listener).
        """
        # speaker_obs = self.goal_pos.astype(np.float32) / (self.grid_size - 1)
        # listener_obs = self.listener_pos.astype(np.float32) / (self.grid_size - 1)
        speaker_obs = self.goal_pos.astype(np.float32)
        listener_obs = self.listener_pos.astype(np.float32)
        return (speaker_obs, listener_obs)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new episode and returns the initial observations.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting (not used here).

        Returns:
            tuple: A tuple containing the initial observations and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0

        if self.goal_sampling_mode == 'non_uniform':
            goal_idx = self.np_random.choice(self.goal_indices, p=self.goal_probs)
            self.goal_pos = np.array(np.unravel_index(goal_idx, (self.grid_size, self.grid_size)))
        else:
            self.goal_pos = self.np_random.integers(0, self.grid_size, size=2)

        while True:
            self.listener_pos = self.np_random.integers(0, self.grid_size, size=2)
            if not np.array_equal(self.listener_pos, self.goal_pos):
                break

        return self._get_obs(), {}

    def step(self, actions):
        """
        Executes one timestep in the environment.

        Args:
            actions (tuple): A tuple of actions, one for each agent (Speaker, Listener).

        Returns:
            tuple: A tuple containing (observations, reward, terminated, truncated, info).
        """
        self.current_step += 1
        listener_action = actions[self.LISTENER_IDX]

        if listener_action == 1:
            self.listener_pos[1] -= 1
        elif listener_action == 2:
            self.listener_pos[1] += 1
        elif listener_action == 3:
            self.listener_pos[0] += 1
        elif listener_action == 4:
            self.listener_pos[0] -= 1

        self.listener_pos = np.clip(self.listener_pos, 0, self.grid_size - 1)

        if self.dynamic_goal and self.np_random.random() < self.goal_move_prob:
            while True:
                new_goal_pos = self.np_random.integers(0, self.grid_size, size=2)
                if not np.array_equal(new_goal_pos, self.listener_pos):
                    self.goal_pos = new_goal_pos
                    break

        terminated = np.array_equal(self.listener_pos, self.goal_pos)
        # reward = 1.0 if terminated else -0.01 * (1 + self.current_step/self.max_steps)
        reward = 1.0 if terminated else -0.01
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Renders the environment for visualization."""
        if mode != 'human':
            super().render(mode=mode)
            return

        grid = np.full((self.grid_size, self.grid_size), '_', dtype=str)
        grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
        grid[self.listener_pos[1], self.listener_pos[0]] = 'L'

        print("\n" + "=" * (self.grid_size * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print(f"Step: {self.current_step}, Goal: {self.goal_pos}, Listener: {self.listener_pos}")
        print("=" * (self.grid_size * 2 + 1))


if __name__ == '__main__':
    print("--- Running Non-Uniform Goal Sampling Example ---")
    # freqs = { (0, 0): 10, (4, 4): 10, (1, 1): 1, (3, 3): 1 }
    freqs = {
        (0, 0): 20,
        (7, 7): 20,
        (3, 4): 10,
        (4, 3): 10,
        (1, 6): 1,
        (6, 1): 1
    }
    env_non_uniform = CommunicatingGoalEnv(grid_size=8, goal_sampling_mode='non_uniform', goal_frequencies=freqs)

    goal_counts = collections.defaultdict(int)
    for _ in range(1000):
        obs, info = env_non_uniform.reset()
        goal_counts[tuple(env_non_uniform.goal_pos)] += 1

    print("Goal sampling counts after 1000 resets:")
    for goal, count in sorted(goal_counts.items()):
        print(f"  Goal {goal}: sampled {count} times")

    env_non_uniform.close()

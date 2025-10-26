from collections import namedtuple
from inspect import getfullargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc', 'comms_loss', 'comms_bits'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode = []
        # Use args flag to identify SMAC environment
        using_smac = self.args.is_smac if hasattr(self.args, 'is_smac') else False
        
        # Handle environment reset
        if using_smac:
            obs, share_obs, available_actions = self.env.reset()
            state = torch.tensor(np.array(share_obs), dtype=torch.float64) if self.args.use_centralized_state else torch.tensor(np.array(obs), dtype=torch.float64)
            # Store available_actions for the first step
            if isinstance(available_actions, list):
                tensor_avail_actions = [torch.tensor(aa, dtype=torch.float64) for aa in available_actions]
            else:
                tensor_avail_actions = torch.tensor(available_actions, dtype=torch.float64)
                
            # Store available_actions for the first step
            info = {'avail_actions': tensor_avail_actions}
        else:
            reset_args = getfullargspec(self.env.reset).args
            if 'epoch' in reset_args:
                state = self.env.reset(epoch)
            else:
                state = self.env.reset()
            info = dict()
        
        should_display = self.display and self.last_step
        if should_display:
            self.env.display()
        
        stat = dict()
        switch_t = -1
        
        prev_hid = torch.zeros(state.shape[0], self.args.nagents, self.args.hid_size)
        
        # Initialize tracking metrics for SMAC
        if using_smac:
            stat['success'] = 0
            # Initialize counters for tracking alive units
            total_allies_alive = 0
            total_enemies_alive = 0
            num_timesteps = 0
        
        for t in range(self.args.max_steps):
            misc = dict()
            
            # Initialize comm_action if needed (existing code)
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros((state.shape[0], self.args.nagents), dtype=int)
            
            # Pass state and info (including available_actions for SMAC) to policy network
            # Recurrent policy handling (existing code)
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                
                if self.args.gacomm or self.args.transformer_comm:
                    action_out, value, prev_hid, comm_density, (comms_loss, comms_bits) = self.policy_net(x, info)
                else:
                    action_out, value, prev_hid, (comms_loss, comms_bits) = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.args.gacomm or self.args.transformer_comm:
                    action_out, value, comm_density, (comms_loss, comms_bits) = self.policy_net(x, info)
                else:
                    action_out, value, (comms_loss, comms_bits) = self.policy_net(x, info)
            
            # Tracking for gacomm (existing code) 
            if self.args.gacomm or self.args.transformer_comm:
                stat['density1'] = stat.get('density1', 0) + comm_density[0]
                stat['density2'] = stat.get('density2', 0) + comm_density[1]
            
            # Select action
            action = select_action(self.args, action_out)
            
            action, actual = translate_action(self.args, self.env, action)

            if self.args.n_rollout_threads == 1 and using_smac:
                # SMAC-specific action translation
                actual = [np.array([a]) for a in actual]
            
            # Environment step
            if using_smac:
                next_obs, next_share_obs, reward, done, info_dict, next_available_actions = self.env.step(actual[0])
                next_state = torch.tensor(np.array(next_share_obs), dtype=torch.float64) if self.args.use_centralized_state else torch.tensor(np.array(next_obs), dtype=torch.float64)
                # Update next_available_actions for next step
                if isinstance(next_available_actions, list):
                    tensor_next_avail_actions = [torch.tensor(aa, dtype=torch.float64) for aa in next_available_actions]
                else:
                    tensor_next_avail_actions = torch.tensor(next_available_actions, dtype=torch.float64)
                    
                info = {'avail_actions': tensor_next_avail_actions}
            else:
                next_state, reward, done, info_dict = self.env.step(actual)
                # For non-SMAC, info resets between iterations
                info = dict()
            
            # Store comm_action in info for next step (existing code)
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm'] = stat.get('enemy_comm', 0) + info['comm_action'][self.args.nfriendly:]
            
            # Handle alive mask consistently for both environments
            if 'alive_mask' in info_dict:
                misc['alive_mask'] = info_dict['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            if using_smac:
                # SMAC-specific mask extraction for allies
                misc['alive_mask'] = 1 - np.array(done).astype(float)  # Use agent-specific done values
                
                # Count alive allies based on misc['alive_mask']
                num_allies_alive = np.sum(misc['alive_mask'])
                
                # Try to extract enemy alive info from info_dict
                num_enemies_alive = 0
                
                # Extract from different possible info formats
                if isinstance(info_dict, dict):
                    # Check for direct enemy_alive_count
                    if 'enemy_alive_count' in info_dict:
                        num_enemies_alive = info_dict['enemy_alive_count']
                    # Try to extract from observation - the obs may contain enemy status
                    elif 'enemy_alive' in info_dict:
                        enemy_alive = info_dict['enemy_alive']
                        num_enemies_alive = np.sum(enemy_alive)
                    # Or try to get from share_obs which might contain enemy info
                    elif next_share_obs is not None:
                        # This extraction depends on the exact shape and format of share_obs
                        # In many SMAC envs, the enemy alive info is in the observation
                        # Try to extract based on common observation format
                        try:
                            if isinstance(next_share_obs, np.ndarray) and next_share_obs.ndim > 2:
                                # Check if enemy alive info might be in a specific part of the obs
                                # This will need adjustment based on your specific environment
                                enemy_info = next_share_obs[0, -self.args.n_enemies:, 0] if hasattr(self.args, 'n_enemies') else []
                                num_enemies_alive = np.sum(enemy_info > 0)
                        except:
                            # If extraction fails, estimate based on map info
                            if hasattr(self.args, 'n_enemies'):
                                num_enemies_alive = self.args.n_enemies  # Initial count
                
                # Increment total counters
                total_allies_alive += num_allies_alive
                total_enemies_alive += num_enemies_alive
                num_timesteps += 1
                
                # Store current values in stats
                stat['current_allies_alive'] = num_allies_alive
                stat['current_enemies_alive'] = num_enemies_alive
            else:
                # Original alive mask code
                if 'alive_mask' in info_dict:
                    misc['alive_mask'] = info_dict['alive_mask'].reshape(reward.shape)
                else:
                    misc['alive_mask'] = np.ones_like(reward)
            
            # Store available_actions in misc for SMAC
            if using_smac:
                misc['avail_actions'] = next_available_actions  # Store for the transition
            
            # Reward handling (existing code)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            
            # done = done or t == self.args.max_steps - 1
            if isinstance(done, (list, np.ndarray, torch.Tensor)):
                # Choose the appropriate termination condition
                # For SMAC, typically end the episode when all agents are done
                done_flag = False
                if isinstance(done, np.ndarray):
                    done_flag = done.all() or t == self.args.max_steps - 1
                elif isinstance(done, torch.Tensor):
                    done_flag = done.all().item() or t == self.args.max_steps - 1
                else:  # list
                    done_flag = all(done) or t == self.args.max_steps - 1
            else:
                done_flag = done or t == self.args.max_steps - 1
            
            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)
            
            stat['comms_loss'] = stat.get('comms_loss', 0) + comms_loss.item()
            stat['comms_bits'] = stat.get('comms_bits', 0) + comms_bits.item()

            if using_smac:
                # Track success for this episode (did we win the battle?)
                battle_won = False
                
                # Handle different info formats
                if isinstance(info_dict, list) and len(info_dict) > 0:
                    for item in info_dict:
                        if isinstance(item, dict):
                            if 'battle_won' in item:
                                battle_won = item['battle_won']
                                stat['battles_won'] = item['battle_won']
                            elif 'battles_won' in item:
                                battle_won = item['battles_won']
                                stat['battles_won'] = item['battles_won']
                            
                            if 'battle_game' in item:
                                stat['battles_game'] = item['battle_game']
                            elif 'battles_game' in item:
                                stat['battles_game'] = item['battles_game']
                elif isinstance(info_dict, dict):
                    if 'battle_won' in info_dict:
                        battle_won = info_dict['battle_won']
                        stat['battles_won'] = info_dict['battle_won']
                    elif 'battles_won' in info_dict:
                        battle_won = info_dict['battles_won']
                        stat['battles_won'] = info_dict['battles_won']
                    
                    if 'battle_game' in info_dict:
                        stat['battles_game'] = info_dict['battle_game']
                    elif 'battles_game' in info_dict:
                        stat['battles_game'] = info_dict['battles_game']
                
                # If this is the final step and the battle was won, mark success
                if done_flag and battle_won:
                    stat['success'] = 1
                        
                # Calculate win rate if possible
                if 'battles_game' in stat and stat['battles_game'] > 0:
                    stat['win_rate'] = stat.get('battles_won', 0) / stat['battles_game']
            
            if done_flag:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info_dict:
                    episode_mini_mask = 1 - info_dict['is_completed'].reshape(-1)
            
            if should_display:
                self.env.display()
            
            # Create transition with all necessary information
            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, comms_loss, comms_bits)
            episode.append(trans)
            state = next_state
            
            if done_flag:
                break

        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        # Calculate average allies and enemies alive
        if using_smac and num_timesteps > 0:
            stat['avg_allies_alive'] = total_allies_alive / num_timesteps
            stat['avg_enemies_alive'] = total_enemies_alive / num_timesteps

        # take mean of the number of bits and loss
        stat['comms_loss'] = stat['comms_loss'] / stat['num_steps']
        stat['comms_bits'] = stat['comms_bits'] / stat['num_steps']

        # Check if the environment provides terminal success information
        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            
            # For SMAC, terminal reward can indicate success
            if using_smac and not stat.get('success', 0) and np.any(reward > 0):
                stat['success'] = 1
            
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        # For SMAC, ensure success is being tracked
        if using_smac and 'success' not in stat:
            # As a fallback, if we didn't catch success during the episode,
            # use the win_rate if available
            if 'win_rate' in stat and stat['win_rate'] > 0:
                stat['success'] = 1
            else:
                stat['success'] = 0

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)

        return (episode, stat)
    

    def compute_grad(self, batch):
        stat = dict()
        using_smac = self.args.is_smac if hasattr(self.args, 'is_smac') else False
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.from_numpy(np.array(batch.reward))
        episode_masks = torch.from_numpy(np.array(batch.episode_mask))
        episode_mini_masks = torch.from_numpy(np.array(batch.episode_mini_mask))
        actions = torch.from_numpy(np.array(batch.action))
        if actions.ndim == 4:
            actions = actions.permute(0, 2, 3, 1).reshape(-1, n, dim_actions)
        else:
            actions = actions.transpose(1, 2).reshape(-1, n, dim_actions)

        if using_smac:
            rewards = rewards.reshape(-1, n)
            episode_masks = episode_masks.reshape(-1, n)
            episode_mini_masks = episode_mini_masks.reshape(-1, n)

        batch_size = rewards.shape[0]

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(self.policy_net.device)

        coop_returns = torch.Tensor(batch_size, n).to(self.policy_net.device)
        ncoop_returns = torch.Tensor(batch_size, n).to(self.policy_net.device)
        returns = torch.Tensor(batch_size, n).to(self.policy_net.device)
        advantages = torch.Tensor(batch_size, n).to(self.policy_net.device)
        values = values.view(batch_size, n).to(self.policy_net.device)

        prev_coop_return = 0
        prev_ncoop_return = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i].to(self.policy_net.device) + self.args.gamma * prev_coop_return * episode_masks[i].to(self.policy_net.device)
            ncoop_returns[i] = rewards[i].to(self.policy_net.device) + self.args.gamma * prev_ncoop_return * episode_masks[i].to(self.policy_net.device) * episode_mini_masks[i].to(self.policy_net.device)

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            # CHANGE 1: Add temperature scaling to action distributions
            temperature = getattr(self.args, 'temperature', 1.0)  # Default to 1.0 if not specified
            
            # Apply temperature scaling to the action logits before softmax
            log_p_a = []
            for i in range(dim_actions):
                # Rescale logits before softmax (assuming action_out contains logits)
                # We need to undo the log_softmax and apply temperature scaling
                if hasattr(self.args, 'rescale_logits') and self.args.rescale_logits:
                    # If action_out already has log_softmax applied
                    probs = action_out[i].exp()
                    logits = torch.log(probs + 1e-10)  # Convert back to logits
                    scaled_logits = logits / temperature
                    log_p_a.append(F.log_softmax(scaled_logits, dim=-1).view(-1, num_actions[i]))
                else:
                    # If action_out contains raw logits
                    log_p_a.append(F.log_softmax(action_out[i] / temperature, dim=-1).view(-1, num_actions[i]))
            
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()

        if self.args.use_comms_channel:
            comms_loss = torch.Tensor(batch.comms_loss).sum()
            action_loss = action_loss + self.args.comms_penalty * comms_loss

        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # CHANGE 2: Enhanced entropy calculation with more diagnostics
            entropy = 0
            max_probs = []
            
            for i in range(len(log_p_a)):
                probs = log_p_a[i].exp()
                entropy_term = -(log_p_a[i] * probs).sum()
                entropy += entropy_term
                
                # Diagnostic: track maximum probability for each action dimension
                max_prob = probs.max(dim=-1)[0].mean()
                max_probs.append(max_prob.item())
            
            # Monitor entropy and probability distributions
            stat['entropy'] = entropy.item()
            stat['max_probs'] = max_probs  # Add max probabilities to stats
            
            # CHANGE 3: Stronger entropy regularization with minimum value
            # Ensure entropy coefficient doesn't become too small
            entropy_coeff = max(getattr(self.args, 'entropy_coefficient', 0.01), 0.0)
            if hasattr(self.args, 'adaptive_entropy') and self.args.adaptive_entropy:
                # Optional: Make entropy coefficient adaptive based on current entropy
                # If entropy is too low, increase the coefficient
                entropy_value = entropy.item() / (len(log_p_a) * batch_size)  # Normalize by actions and batch
                if entropy_value < getattr(self.args, 'min_entropy', 0.1):
                    # Increase entropy coefficient if entropy is too low
                    entropy_coeff = max(entropy_coeff, 0.05)
            
            # Apply entropy regularization with enhanced coefficient
            loss -= entropy_coeff * entropy
            stat['entropy_coeff'] = entropy_coeff  # Track coefficient
        
        loss.backward()

        # CHANGE 4: Add gradient diagnostics
        before_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                before_norm += param_norm.item() ** 2
        before_norm = before_norm ** 0.5
        stat['grad_norm_before_clip'] = before_norm
        
        # CHANGE 5: Add gradient clipping
        if self.args.max_grad_norm != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), 
                getattr(self.args, 'max_grad_norm', 0.5)
            )
            stat['grad_norm_after_clip'] = grad_norm.item()
        
        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat
    
    def eval_batch(self, epoch):
        _, stat = self.run_batch(epoch)
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

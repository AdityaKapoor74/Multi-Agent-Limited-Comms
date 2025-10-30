import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical



# ------ Network Definitions --------

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

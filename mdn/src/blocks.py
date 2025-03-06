from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseType(Enum):
    FULL = auto()
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim, noise_type=NoiseType.DIAGONAL, fixed_noise_level=None):
        super(MixtureDensityNetwork, self).__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.FULL: int(dim_out * (dim_out + 1) / 2) * n_components,
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, dim_out * n_components + num_sigma_channels)
        )

        self.upper_triangular_mask = (torch.triu(torch.ones(dim_out, dim_out)) == 1).expand(n_components, -1, -1)

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        if self.noise_type is NoiseType.FULL:
            sigma_mat = torch.empty((len(x), self.n_components, self.dim_out, self.dim_out)).to(x)
            sigma_mat[self.upper_triangular_mask.expand(len(x),-1,-1,-1)] = sigma
            sigma_mat.T[self.upper_triangular_mask.expand(len(x),-1,-1,-1)] = sigma
            sigma = torch.exp(sigma_mat + eps)
            return log_pi, mu, sigma
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        if self.noise_type is NoiseType.FULL:
            diff = (y.unsqueeze(1) - mu).unsqueeze(-1)
            z_score = diff.transpose(-1,-2)@torch.linalg.solve(sigma, diff).squeeze()
            normal_loglik = -0.5 * z_score - torch.logdet(sigma)
        else:
            z_score = (y.unsqueeze(1) - mu) / sigma
            normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                -torch.sum(torch.log(sigma), dim=-1)
            )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x, samples_per_input=1):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand([*x.shape[:-1], samples_per_input]).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs).unsqueeze(-1)
        rand_pi = torch.clamp(rand_pi, 0, self.n_components-1)

        rand_mu = torch.take_along_dim(mu, indices=rand_pi, dim=1)
        rand_sigma = torch.take_along_dim(sigma, indices=rand_pi, dim=1)
        samples = rand_mu + rand_sigma * torch.randn_like(rand_mu)
        samples = samples.permute(-2, *tuple(range(len(samples.shape)-2)), -1)

        # rand_pi = torch.searchsorted(cum_pi, rvs)
        # rand_normal = torch.randn_like(mu) * sigma + mu
        # samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)

        return samples

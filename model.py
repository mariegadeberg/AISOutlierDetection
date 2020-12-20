import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.distributions import ContinuousBernoulli
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from torch.autograd import Variable

class VRNN(nn.Module):

    def __init__(self, input_shape, latent_shape, mean_, splits, len_data):
        super(VRNN, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.mean = mean_
        self.splits = splits
        self.len_data = len_data

        self.phi_x = nn.Sequential(nn.Linear(self.input_shape, self.latent_shape),
                                   nn.ReLU())

        self.phi_z = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU())

        self.prior = nn.Sequential(nn.Linear(self.latent_shape, 2*self.latent_shape),
                                   nn.ReLU())

        self.encoder = nn.Sequential(nn.Linear(self.latent_shape+self.latent_shape, 2*self.latent_shape),
                                     nn.ReLU())
                                     #nn.BatchNorm1d(2*self.latent_shape),

        self.decoder = nn.Sequential(nn.Linear(self.latent_shape+self.latent_shape, self.input_shape),
                                     nn.ReLU())

        self.rnn = nn.LSTM(self.latent_shape + self.latent_shape, self.latent_shape, batch_first=True)

        self.register_buffer('out', torch.zeros(1, self.latent_shape))
        self.register_buffer('h', torch.zeros(1, self.latent_shape))
        self.register_buffer('c', torch.zeros(1, 1, self.latent_shape))

        #self.bn = nn.BatchNorm1d(self.latent_shape)
        #self.bn.weight.requires_grad = False

    def _prior(self, h, sigma_min=0.0, raw_sigma_bias=0.5):
        hidden = self.prior(h)
        mu, log_sigma = hidden.chunk(2, dim=-1)

        sigma = log_sigma.exp()
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.maximum(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)
        log_sigma = torch.log(sigma)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def posterior(self, hidden, x, prior_mu):
        encoder_input = torch.cat([hidden, x], dim=1)
        hidden = self.encoder(encoder_input)
        #hidden = hidden.unsqueeze(1)
        mu, log_sigma = hidden.chunk(2, dim=-1)
        mu = mu + prior_mu
        #mu = self.bn(mu)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def generative(self, z_enc, h):
        px_logits = self.decoder(torch.cat([z_enc, h], dim=2))
        px_logits = px_logits + self.mean
        #px_logits = px_logits.view(-1, self.input_shape) + self.mean
        #print(self.mean)
        #print(px_logits.shape)
        #k+=1
        return Bernoulli(logits=px_logits)

    def get_bce(self, log_px, x):
        log_px_splits = torch.split(log_px, self.splits, dim=1)
        x_splits = torch.split(x, self.splits, dim=1)
        loss = []
        for log_px, x in zip(log_px_splits, x_splits):
            loss.append(binary_cross_entropy_with_logits(log_px, x, reduction='sum'))

        #print(loss.shape)
        #bce = torch.cat(loss, dim=1)
        #bce.sum(dim=1)
        bce = torch.stack(loss).sum() / x.size(0)
        return bce

    def get_kl_analytic(self, qz, pz):
        kld_element = (2 * torch.log(pz.sigma) - 2 * torch.log(qz.sigma) +
                       (qz.sigma**2 + (qz.mu - pz.mu)**2) /
                       pz.sigma**2 - 1)
        return 0.5 * torch.sum(kld_element)



    def forward(self, inputs, beta):

        batch_size = inputs.size(0)

        #kl_weight = batch_size/self.len_data

        out = self.out.expand(batch_size, *self.out.shape[1:]).contiguous()
        h = self.h.expand(1, batch_size, self.c.shape[-1]).contiguous()
        c = self.c.expand(1, batch_size, self.c.shape[-1]).contiguous()

        acc_loss = 0
        loss_list = []
        kl_list = []
        log_px_list = []
        h_out = []
        mu_prior = []
        mu_post = []
        z_out = 0
        #for x in inputs:
        for t in range(inputs.size(1)):
            x = inputs[:, t, :]

            #Embed input
            x_hat = self.phi_x(x)
            #Create prior distribution
            pz = self._prior(out)

            #Create approximate posterior
            qz = self.posterior(out, x_hat, prior_mu=pz.mu)

            #Sample and embed z from posterior
            z = qz.rsample()
            if 10 > 1:
                z_ = [z]
                for i in range(10-1):
                    z_.append(qz.rsample())
                z = torch.stack(z_, dim=2).permute(0, 2, 1)

            z_hat = self.phi_z(z)

            #Decode z_hat
            out = out.unsqueeze(1).expand(-1, 10, -1)
            px = self.generative(z_hat, out)

            #Update h from LSTM

            rnn_input = torch.cat([x_hat.unsqueeze(1).expand(-1, 10, -1), z_hat], dim=2)
            #rnn_input = rnn_input.unsqueeze(1)
            out, (h, c) = self.rnn(rnn_input, (h, c))
            out = out.squeeze()

            h_out.append(out.mean(dim=1))

            #Calulating loss
            log_px = px.log_prob(x).sum(dim=1)
            log_pz = pz.log_prob(z).sum(dim=1)
            log_qz = qz.log_prob(z).sum(dim=1)

            kl = log_qz - log_pz
            #kl = self.get_kl_analytic(qz, pz)
            #log_px_bce = self.get_bce(px.log_prob(x), x)
            #elbo_beta = log_px_bce - self.beta * kl.mean()

            elbo_beta = log_px - beta * kl

            #acc_loss += -torch.mean(elbo_beta)
            acc_loss += elbo_beta
            #acc_loss += -torch.mean(torch.logsumexp(elbo_beta,0))

            #iwae_elbo = log_px - kl_weight * kl
            #weight = torch.nn.functional.softmax(iwae_elbo, dim=-1)
            #acc_loss += -torch.mean(torch.sum(weight * iwae_elbo, dim=-1), dim=0)

            loss_list.append(-elbo_beta)
            #loss_list.append(acc_loss)
            kl_list.append(kl)
            log_px_list.append(px.logits)
            mu_prior.append(pz.mu.sum(dim=1))
            mu_post.append(qz.mu.sum(dim=1))

        with torch.no_grad():
            diagnostics = {'loss_list': torch.stack(loss_list).cpu().numpy(),
                           'log_px': torch.stack(log_px_list).cpu().numpy(),
                           'kl': torch.stack(kl_list).cpu().numpy(),
                           'h': torch.stack(h_out).cpu().numpy(),
                           "mu_prior": torch.stack(mu_prior).cpu().numpy(),
                           "mu_post": torch.stack(mu_post).cpu().numpy()}

        return -torch.mean(acc_loss/inputs.size(1)), diagnostics


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        dist = torch.distributions.Normal(self.mu, self.sigma)
        return dist.log_prob(z)






'''

def test_normal_distribution():
    """a few safety checks for your implementation"""
    N = 1000000
    ones = torch.ones(torch.Size((N,)))
    mu = 1.224 * ones
    sigma = 0.689 * ones
    dist = ReparameterizedDiagonalGaussian(mu, sigma.log())
    z = dist.sample()

    # Expected value E[N(0, 1)] = 0
    expected_z = z.mean()
    diff = (expected_z - mu.mean()) ** 2
    assert diff < 1e-3, f"diff = {diff}, expected_z = {expected_z}"

    # Variance E[z**2 - E[z]**2]
    var_z = (z ** 2 - expected_z ** 2).mean()
    diff = (var_z - sigma.pow(2).mean()) ** 2
    assert diff < 1e-3, f"diff = {diff}, var_z = {var_z}"

    # log p(z)
    from torch.distributions import Normal
    base = Normal(loc=mu, scale=sigma)
    diff = ((base.log_prob(z) - dist.log_prob(z)) ** 2).mean()
    assert diff < 1e-3, f"diff = {diff}"


test_normal_distribution()

n_samples = 10000
mu = torch.tensor([[0, 1]])
sigma = torch.tensor([[0.5, 3]])
ones = torch.ones((1000, 2))
p = ReparameterizedDiagonalGaussian(mu=mu * ones, log_sigma=(sigma * ones).log())
samples = p.sample()
data = pd.DataFrame({"x": samples[:, 0], "y": samples[:, 1]})
g = sns.jointplot(
    data=data,
    x="x", y="y",
    kind="hex",
    ratio=10
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle(r"$\mathcal{N}(\mathbf{y} \mid \mu, \sigma)$")
plt.show()

'''
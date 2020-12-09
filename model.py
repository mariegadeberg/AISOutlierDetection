import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from torch.autograd import Variable

class VRNN(nn.Module):

    def __init__(self, input_shape, latent_shape, beta):
        super(VRNN, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.beta = beta

        self.phi_x = nn.Sequential(nn.Linear(self.input_shape, self.latent_shape),
                                   nn.ReLU())

        self.phi_z = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU())

        self.prior = nn.Sequential(nn.Linear(self.latent_shape, 2*self.latent_shape),
                                   nn.ReLU())

        self.encoder = nn.Sequential(nn.Linear(self.latent_shape+self.latent_shape, 2*self.latent_shape),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(self.latent_shape+self.latent_shape, self.input_shape),
                                     nn.ReLU())

        self.rnn = nn.LSTM(self.latent_shape + self.latent_shape, self.latent_shape, batch_first=True)

        self.register_buffer('out', torch.zeros(1, 1, self.latent_shape))
        self.register_buffer('h', torch.zeros(1, 1, self.latent_shape))
        self.register_buffer('c', torch.zeros(1, 1, self.latent_shape))

    def _prior(self, h):
        hidden = self.prior(h)
        mu, log_sigma = hidden.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def posterior(self, hidden, x):
        encoder_input = torch.cat([hidden, x], dim=2)
        hidden = self.encoder(encoder_input)
        #hidden = self.encoder(torch.cat([hidden, x]))
        mu, log_sigma = hidden.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def generative(self, z_enc, h):
        px_logits = self.decoder(torch.cat([z_enc, h], dim=2))
        return Bernoulli(logits=px_logits)

    def forward(self, inputs):

        batch_size = inputs.size(0)

        #out = torch.zeros(1, 1, self.latent_shape)
        #h = torch.zeros(1, 1, self.latent_shape)
        #c = torch.zeros(1, 1, self.latent_shape)

        out = self.out.expand(batch_size, *self.out.shape[1:]).contiguous()
        h = self.h.expand(1, batch_size, self.h.shape[-1]).contiguous()
        c = self.c.expand(1, batch_size, self.c.shape[-1]).contiguous()

        acc_loss = 0
        loss_list = []
        kl_list = []
        log_px_list = []
        h_out = []

        z_out = 0

        #for x in inputs:
        for t in range(inputs.size(1)):
            x = inputs[:, t, :].unsqueeze(1)

            #Embed input
            x_hat = self.phi_x(x)
            #Create prior distribution
            pz = self._prior(out)

            #Create approximate posterior
            qz = self.posterior(out, x_hat)

            #Sample and embed z from posterior
            z = qz.rsample()
            z_hat = self.phi_z(z)

            #Decode z_hat
            px = self.generative(z_hat, out)

            #Update h form LSTM
            #rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = torch.cat([x_hat, z_hat], dim=2)
            #rnn_input = rnn_input.unsqueeze(1)
            out, (h, c) = self.rnn(rnn_input, (h, c))

            h_out.append(out.mean(axis=2))

            #Calulating loss
            log_px = px.log_prob(x).sum(axis=2)
            log_pz = pz.log_prob(z).sum(axis=2)
            log_qz = qz.log_prob(z).sum(axis=2)

            kl = log_qz - log_pz
            elbo_beta = log_px - self.beta * kl

            acc_loss += -elbo_beta.mean()

            loss_list.append(-elbo_beta)
            kl_list.append(kl)
            log_px_list.append(log_px)

        with torch.no_grad():
            diagnostics = {'loss_list': torch.stack(loss_list).cpu().numpy(),
                           'log_px': torch.stack(log_px_list).cpu().numpy(),
                           'kl': torch.stack(kl_list).cpu().numpy(),
                           'h': torch.stack(h_out).cpu().numpy()}

        return acc_loss/len(inputs[0]), diagnostics


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
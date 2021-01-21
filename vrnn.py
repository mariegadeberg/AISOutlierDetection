import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch.nn.functional import binary_cross_entropy_with_logits
#from torch.distributions import ContinuousBernoulli
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from torch.autograd import Variable
import math

class VRNN(nn.Module):

    def __init__(self, input_shape, latent_shape, mean_logits, mean_, splits, len_data, gamma, bn_switch):
        super(VRNN, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.mean_logits = mean_logits
        self.mean_ = mean_
        self.splits = splits
        self.len_data = len_data
        self.gamma = gamma
        self.bn_switch = bn_switch

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
        #torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        #torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)

        self.register_buffer('out', torch.zeros(1, self.latent_shape))
        self.register_buffer('h', torch.zeros(1, self.latent_shape))
        self.register_buffer('c', torch.zeros(1, 1, self.latent_shape))

        if self.bn_switch:
            self.bn = nn.BatchNorm1d(self.latent_shape)
            self.bn.weight.requires_grad = False
            self.bn.weight.fill_(self.gamma)

    def _prior(self, h, sigma_min=0.0, raw_sigma_bias=0.5):
        hidden = self.prior(h)
        mu, log_sigma = hidden.chunk(2, dim=-1)

        #sigma = log_sigma.exp()
        #sigma_min = torch.full_like(sigma, sigma_min)
        #sigma = torch.maximum(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)
        #log_sigma = torch.log(sigma)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def posterior(self, hidden, x, prior_mu,  sigma_min=0.0, raw_sigma_bias=0.5):
        encoder_input = torch.cat([hidden, x], dim=1)
        hidden = self.encoder(encoder_input)
        #hidden = hidden.unsqueeze(1)
        mu, log_sigma = hidden.chunk(2, dim=-1)

        #sigma = log_sigma.exp()
        #sigma_min = torch.full_like(sigma, sigma_min)
        #sigma = torch.maximum(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)
        #log_sigma = torch.log(sigma)

        mu = mu + prior_mu
        if self.bn_switch:
            mu = self.bn(mu)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def generative(self, z_enc, h):
        px_logits = self.decoder(torch.cat([z_enc, h], dim=1))
        px_logits = px_logits + self.mean_logits
        return Bernoulli(logits=px_logits)

    def get_bce(self, log_px, x):
        log_px_splits = torch.split(log_px, self.splits, dim=1)
        x_splits = torch.split(x, self.splits, dim=1)
        loss = []
        for log_px, x in zip(log_px_splits, x_splits):
            loss.append(binary_cross_entropy_with_logits(log_px, x, reduction='sum'))

        bce = torch.stack(loss).sum() / x.size(0)
        return bce

    def get_kl_analytic(self, qz, pz):
        kld_element = (2 * torch.log(pz.sigma) - 2 * torch.log(qz.sigma) +
                       (qz.sigma**2 + (qz.mu - pz.mu)**2) /
                       pz.sigma**2 - 1)
        return 0.5 * torch.sum(kld_element)

    def forward(self, inputs, beta):

        batch_size = inputs.size(0)

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
            z_hat = self.phi_z(z)

            #Decode z_hat
            px = self.generative(z_hat, out)

            #Update h from LSTM
            rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = rnn_input.unsqueeze(1)
            out, (h, c) = self.rnn(rnn_input, (h, c))
            out = out.squeeze()

            #h_out.append(out.mean(dim=1))

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
            log_px_list.append(log_px)
            mu_prior.append(pz.mu.sum(dim=1))
            mu_post.append(qz.mu.sum(dim=1))

        with torch.no_grad():
            diagnostics = {'loss_list': torch.stack(loss_list).cpu().numpy(),
                           'log_px': torch.stack(log_px_list).cpu().numpy(),
                           'kl': torch.stack(kl_list).cpu().numpy(),
                           #'h': torch.stack(h_out).cpu().numpy(),
                           "mu_prior": torch.stack(mu_prior).cpu().numpy(),
                           "mu_post": torch.stack(mu_post).cpu().numpy()}

        loss = torch.mean(acc_loss/inputs.size(1))

        return -loss, diagnostics

    def calc_mi(self, inputs):

        batch_size = inputs.size(0)

        out = self.out.expand(batch_size, *self.out.shape[1:]).contiguous()
        h = self.h.expand(1, batch_size, self.c.shape[-1]).contiguous()
        c = self.c.expand(1, batch_size, self.c.shape[-1]).contiguous()

        neg_entropy = 0
        log_qz = 0

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            x_hat = self.phi_x(x)

            # Create prior distribution
            pz = self._prior(out)

            # Create approximate posterior
            qz = self.posterior(out, x_hat, prior_mu=pz.mu)

            mu = qz.mu
            logsigma = torch.log(qz.sigma)

            z = qz.rsample()
            z_hat = self.phi_z(z)

            rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = rnn_input.unsqueeze(1)
            out, (h, c) = self.rnn(rnn_input, (h, c))
            out = out.squeeze()

            neg_entropy += (-0.5 * self.latent_shape * math.log(2 * math.pi) - 0.5 * (1 + 2 * logsigma).sum(-1)).mean()

            var = logsigma.exp()**2

            z = z.unsqueeze(1)
            mu = mu.unsqueeze(0)
            logsigma = logsigma.unsqueeze(0)

            dev = z - mu

            log_density = -0.5 * ((dev ** 2 ) / var).sum(dim=-1) - 0.5 * (self.latent_shape * math.log(2 * math.pi) + (2*logsigma).sum(dim=-1))

            log_qz1 = torch.logsumexp(log_density, dim=1) - math.log(batch_size)
            log_qz += log_qz1.mean(-1)

        mi = (neg_entropy / inputs.size(1)) - (log_qz / inputs.size(1))

        return mi

    def encode_stats(self, inputs):
        batch_size = inputs.size(0)

        out = self.out.expand(batch_size, *self.out.shape[1:]).contiguous()
        h = self.h.expand(1, batch_size, self.c.shape[-1]).contiguous()
        c = self.c.expand(1, batch_size, self.c.shape[-1]).contiguous()

        mu = torch.zeros(batch_size, self.latent_shape)

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            x_hat = self.phi_x(x)

            # Create prior distribution
            pz = self._prior(out)

            # Create approximate posterior
            qz = self.posterior(out, x_hat, prior_mu=pz.mu)

            mu += qz.mu.cpu().numpy()

            z = qz.rsample()
            z_hat = self.phi_z(z)

            rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = rnn_input.unsqueeze(1)
            out, (h, c) = self.rnn(rnn_input, (h, c))
            out = out.squeeze()

        return mu / inputs.size(1)







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


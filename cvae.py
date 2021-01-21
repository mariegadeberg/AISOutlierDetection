import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from torch.distributions import Bernoulli
import math
from scripts_main.vrnn import ReparameterizedDiagonalGaussian


class CVAE(nn.Module):

    def __init__(self, latent_features):
        super(CVAE, self).__init__()

        self.latent_features = latent_features

        self.init_kernel = 16

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, #input is bz x ch x 201 x 402
                                               out_channels=self.init_kernel,
                                               kernel_size=(3, 4),
                                               stride=2,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=self.init_kernel, #input is bz x 16 x 101 x 201
                                               out_channels=self.init_kernel*2,
                                               kernel_size=(3, 3),
                                               stride=2,
                                               padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*2),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=self.init_kernel*2, #input is bz x 32 x 51 x 101
                                               out_channels=self.init_kernel*4,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*4),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=self.init_kernel*4, #input is bz x 64 x 26 x 51
                                               out_channels=self.init_kernel*8,
                                               kernel_size=(4, 3),
                                               stride=2,
                                               padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*8),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=self.init_kernel*8, #input is bz x 128 x 13 x 26
                                               out_channels=self.init_kernel*16,
                                               kernel_size=(3,4),
                                               stride=2,
                                               padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*16),
                                     nn.ReLU(), #output is bz x 256 x 7 x 13
                                     flatten(), #flatten to bz x 23296
                                     nn.Linear(in_features=256*7*13, out_features=512),
                                     nn.ReLU(),
                                     nn.Linear(in_features=512, out_features=2*self.latent_features) #2 * latent features to split for mu and std
                                     )

        self.decoder = nn.Sequential(nn.Linear(in_features=self.latent_features, out_features=512),
                                     nn.ReLU(),
                                     nn.Linear(in_features=512, out_features=256*7*13),
                                     unflatten(), #get back to shape bz x 256 x 7 x 13
                                     nn.ConvTranspose2d(in_channels=self.init_kernel*16,
                                                        out_channels=self.init_kernel*8,
                                                        kernel_size = (3, 4),
                                                        stride=2,
                                                        padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*8),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=self.init_kernel*8, #input is bz x 128 x 13 x 26
                                                        out_channels=self.init_kernel*4,
                                                        kernel_size = (4, 3),
                                                        stride=2,
                                                        padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*4),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=self.init_kernel*4, #input is bz x 64 x 26 x 51
                                                        out_channels=self.init_kernel*2,
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1),
                                     #nn.BatchNorm2d(self.init_kernel*2),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=self.init_kernel*2,
                                                        out_channels=self.init_kernel,
                                                        kernel_size=(3, 3),
                                                        stride=2,
                                                        padding=1),
                                     #nn.BatchNorm2d(self.init_kernel),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=self.init_kernel,
                                                        out_channels=1,
                                                        kernel_size=(3,4),
                                                        stride=2,
                                                        padding=1))

        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * latent_features])))

    def prior(self, bz):
        prior_params = self.prior_params.expand(bz, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def approximate_posterior(self, x):
        h = self.encoder(x)
        mu, log_sigma = h.chunk(2, dim=-1)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def generative(self, z):
        px_logits = self.decoder(z)
        return Bernoulli(logits=px_logits)

    def reduce(self, x):
        """for each datapoint: sum over all dimensions"""
        return x.view(x.size(0), -1).sum(dim=1)

    def forward(self, x):

        #Define approximate posterior
        qz = self.approximate_posterior(x)

        #Define prior
        pz = self.prior(x.size(0))

        #Sample z
        z = qz.rsample()

        #Define generative model
        px = self.generative(z)

        #evaluate log probabilities
        log_px = self.reduce(px.log_prob(x))
        log_qz = self.reduce(qz.log_prob(z))
        log_pz = self.reduce(pz.log_prob(z))

        kl = log_qz - log_pz
        elbo = log_px - kl

        loss = -elbo.mean()

        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px': log_px, 'kl': kl}

        return loss, diagnostics

    def sample(self, x):
        qz = self.approximate_posterior(x)

        # Define prior
        pz = self.prior(x.size(0))

        # Sample z
        z = qz.rsample()

        # Define generative model
        px = self.generative(z)

        logits = px.logits
        log_px = self.reduce(px.log_prob(x))

        return logits, log_px

    def calc_mi(self, x):

        batch_size = x.size(0)

        # Define approximate posterior
        qz = self.approximate_posterior(x)

        # Sample z
        z = qz.rsample()

        mu = qz.mu
        logsigma = torch.log(qz.sigma)

        # Create approximate posterior

        neg_entropy = (-0.5 * self.latent_features * math.log(2 * math.pi) - 0.5 * (1 + 2 * logsigma).sum(-1)).mean()

        var = logsigma.exp()**2

        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logsigma = logsigma.unsqueeze(0)

        dev = z - mu
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (self.latent_features * math.log(2 * math.pi) + (2*logsigma).sum(dim=-1))

        log_qz = torch.logsumexp(log_density, dim=1) - math.log(batch_size)

        mi = (neg_entropy - log_qz.mean(-1)).item()

        return mi

    def encode_stats(self, x):
        x = x.unsqueeze(1)
        qz = self.approximate_posterior(x)

        mu = qz.mu

        return mu

    def sample_corrupt(self, x_c, x_t):
        qz = self.approximate_posterior(x_c)

        # Sample z
        z = qz.rsample()

        # Define generative model
        px = self.generative(z)

        logits = px.logits
        log_px = self.reduce(px.log_prob(x_t))
        #log_px = px.log_prob(x_t)

        return logits, log_px

class flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class unflatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 7, 13)


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
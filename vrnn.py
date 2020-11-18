import torch
from torch import nn, optim
from torch.autograd import Variable

class VRNNcell(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VRNNcell, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.phi_x = nn.Sequential(nn.Linear(input_size, latent_size*2),
                                   nn.ReLU)
        self.encoder = Encoder(latent_size, latent_size)
        self.phi_z = nn.Sequential(nn.Linear(latent_size, latent_size*2),
                                   nn.ReLU)
        self.decoder = Decoder(input_size, latent_size)
        self.prior = Prior(latent_size, latent_size)
        self.rnn = nn.LSTM(input_size + latent_size, latent_size)

    def forward(self, inputs, targets, state):
        rnn_state, prev_latent_encoded = state

        inputs_encoded = self.phi_x(inputs)
        targets_encoded = self.phi_x(targets)

        rnn_inputs = torch.cat([inputs_encoded, prev_latent_encoded], dim=1)
        rnn_out, new_rnn_state = self.rnn(rnn_inputs, rnn_state)

        latent_dist_prior = self.prior(rnn_out)
        latent_dist_q = self.encoder(rnn_out, targets_encoded, mu_prior = latent_dist_prior.loc)

        latent_state = latent_dist_q.sample()
        latent_encoded = self.phi_z(latent_state)

        latent_state_prior = latent_dist_prior.sample()
        latent_prior_encoded = self.phi_z(latent_state_prior)

        log_q_z = latent_dist_q.log_prob(latent_state).sum()
        log_p_z = latent_dist_prior.log_prob(latent_state).sum()

        analytic_kl = torch.distributions.kl.kl_divergence(latent_dist_q, latent_dist_prior).sum

        generative_dist = self.decoder(latent_encoded, rnn_out)

        log_p_x_given_z = generative_dist.log_prob(targets).sum()

        latent_encoded_return = (latent_prior_encoded, latent_encoded)

        return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
                (new_rnn_state, latent_encoded_return), rnn_out)

    def init_hidden(self):




class Prior():
    def __init__(self, size, hidden_layer_sizes, sigma_min = 0.0, raw_sigma_bias = 0.25, ):
        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.parameterizer = nn.Linear(size, hidden_layer_sizes + 2*size)

    def condition(self, tensor_list):
        inputs = torch.cat(tensor_list)
        outs = self.parameterizer(inputs)
        mu, sigma = torch.split(outs, 2, dim = 1)
        sigma = torch.maximum(torch.nn.Softplus(sigma + self.raw_sigma_bias), self.sigma_min)
        return mu, sigma

    def __call__(self, *args, **kwargs):
        mu, sigma = self.condition(args, **kwargs)
        return torch.distributions.normal.Normal(log=mu, scale=sigma)


class Decoder():
    def __init__(self, size, hidden_layer_sizes, bias_init = 0.0):
        self.bias_init = bias_init
        self.parameterizer = nn.Linear(size, hidden_layer_sizes + 2 * size)

    def condition(self, tensor_list):
        inputs = torch.cat(tensor_list)
        return self.parameterizer(inputs) + self.bias_init

    def __call__(self, *args):
        p = self.condition(args)
        return torch.distributions.bernoulli.Bernoulli(logits = p)


class Encoder(Prior):
    def condition(self, tensor_list, mu_prior):
        mu, sigma = super(Encoder, self).condition(tensor_list)
        return mu + mu_prior, sigma





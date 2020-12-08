import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch import nn

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def get_weights(w_dict, model):
    for name in w_dict:
        for n, p in model.named_parameters():
            if n == name:
                w_ave = p.grad.abs().mean()
                w_dict[name].append(w_ave)
    return w_dict

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


'''
def test_func(w_dict, named_parameters, i):
    for name in w_dict.keys():
        print(f"name from dict :{name}")
        i += 1
        ls = w_dict[name]
        for n, p in named_parameters:
            print(f"name from model {n}")
            if n == name:
                print(f"appending for {name}")
                w_ave = i
                ls.append(w_ave)
        w_dict[name] = ls

    return w_dict

def test_func(w_dict, named_parameters, i):
    for name in w_dict.keys():
        print(f"name from dict :{name}")
        i += 1
        w_dict[name].append([i for n, p in named_parameters if n == name])

    return w_dict






w_dict = {"phi_x.0.weight": [],
         "phi_z.0.weight": [],
         "prior.0.weight": [],
         "encoder.0.weight": [],
         "decoder.0.weight": [],
         "rnn.weight_ih_l0": [],
         "rnn.weight_hh_l0": []}

w_dict = test_func(w_dict, model, 0)
'''
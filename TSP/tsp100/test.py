
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from critic_network import CriticNetwork
from options import get_options
from test_function import validate, get_inner_model
from baselines import CriticBaseline
from attention_model import AttentionModel
from utils import torch_load_cpu, load_model, maybe_cuda_model, load_problem
import math


# 
# ## Test instance with changed graph size, seed, steps

# In[2]:


opts=get_options()
opts.graph_size=100 ########################################################change
opts.eval_only=True
opts.seed=1234
opts.steps=1000
opts.load_path='outputs/tsp_100/run/epoch-198.pt'



# In[3]:


# Set the random seed
torch.manual_seed(opts.seed)

# Figure out what's the problem
problem = load_problem(opts.problem)
val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset)

# In[4]:


# Load data from load_path
load_data = {}
assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
load_path = opts.load_path if opts.load_path is not None else opts.resume
if load_path is not None:
    print('  [*] Loading data from {}'.format(load_path))
    load_data = load_data = torch_load_cpu(load_path)

# Initialize model
model_class = AttentionModel
model = maybe_cuda_model(
    model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ),
    opts.use_cuda
)

# Overwrite model parameters by parameters to load
model_ = get_inner_model(model)
model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

# Initialize baseline
baseline = CriticBaseline(
            maybe_cuda_model(
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                ),
                opts.use_cuda
            )
        )
# Load baseline from data, make sure script is called with same type of baseline
if 'baseline' in load_data:
    baseline.load_state_dict(load_data['baseline'])


if opts.eval_only:


    best = validate(model, val_dataset, opts)

    print('Best solutions: {} +- {}'.format(
    best.mean().item(), torch.std(best) / math.sqrt(len(best))))



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
from test_function import get_inner_model, validate
from baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from attention_model import AttentionModel
from utils import torch_load_cpu, load_model, maybe_cuda_model, load_problem
import math


# ## Test instance with changed graph size, seed, steps

# In[2]:

opts=get_options()
opts.graph_size=100 ########################################################change
opts.eval_only=True
opts.seed=1234
opts.steps=5000
opts.load_path='outputs/cvrp_100/run/epoch-170.pt'

# In[3]:


torch.manual_seed(opts.seed)
problem = load_problem(opts.problem)
val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset)
#torch.save(val_dataset, 'myval_CVRP_100.pt')
#val_dataset = torch.load('myval_CVRP_50.pt')
# val_dataset=val_dataset[500:1000]


# In[4]:


from itertools import combinations, permutations
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

o=torch.tensor([i for i in range(128)]) ######change the number of dummy depots
oo = o
CL=list(combinations(o, 2))
lis=[]
for i in CL:
    ooo = oo.clone()
    ooo[i[0]:i[1]+1]=torch.flip(oo[i[0]:i[1]+1],[0])
    lis.append(ooo)
dic = torch.stack(lis,0)
dic=dic.cuda()
CL = torch.tensor(CL).cuda()


# Overwrite model parameters by parameters to load
model_ = get_inner_model(model)
model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    
    
l=[]
if opts.eval_only:
    lis=[]
    for i in range(1):

        total_cost, return_return, best = validate(model, val_dataset, dic, CL, opts)

        l.append(best)



#     print('Improving: {} +- {}'.format(
#     total_cost.mean().item(), torch.std(total_cost) / math.sqrt(len(total_cost))))    

#     print('Best Improving: {} +- {}'.format(
#     return_return.mean().item(), torch.std(return_return) / math.sqrt(len(return_return))))
        print('Best solutions: {} +- {}'.format(
        best.mean().item(), torch.std(best) / math.sqrt(len(best))))


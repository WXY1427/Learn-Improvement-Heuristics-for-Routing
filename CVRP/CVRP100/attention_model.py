import torch
import time
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import math
from graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel


# def set_decode_type(model, decode_type):
#     if isinstance(model, DataParallel):
#         model = model.module
#     model.set_decode_type(decode_type)


class AttentionModel(nn.Module):

    VEHICLE_CAPACITY = 1.  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=1,   ##############  8
                 fix_norm_factor=False):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.norm_factor = 1 / math.sqrt(embedding_dim) if fix_norm_factor else math.sqrt(embedding_dim)   #??????????????
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp:
            step_context_dim = embedding_dim + 1  # Embedding of last node + remaining_capacity
            node_dim = 7  # x, y, demand

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)           ###############?????????????????????
        else:  # TSP
            assert problem.NAME == 'tsp', "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, rec, input_info, pos_enc, exc, che_mask, dic, cl, action, test):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        
        to_attention = self.init_embed(input_info)+pos_enc#+depot_embed[:, None, :].expand(*pos_enc.size()) 
        
        if action is None:
            
            soft_max, exchange, act = self.embedder(to_attention, test, exc, che_mask, action) 
            
            # rec: bs, graph_size; exchange: n_heads, bs, 2
            exchange_once = exchange[0,:,:]+1       

    
            loc_of_first = torch.nonzero(rec.long() == exchange_once[:,0][:,None].long())
        
            loc_of_second = torch.nonzero(rec.long() == exchange_once[:,1][:,None].long())
        
            exchange_for_now = torch.cat((loc_of_first[:,1][:,None], loc_of_second[:,1][:,None]),1)
        
            # return (length_now-length_pre), None, length_now, dataset, rec_new from get_costs()
            now_length, rec_new, che_mask  = self.problem.get_costs(input, rec, dic, cl, exchange_for_now)
            
        else:
            soft_max = self.embedder(to_attention, test, exc, che_mask, action)    
            


        # exchange: n_heads, batch_size, 2； soft_max: n_heads, batch_size
        if action is None:
            if test:
                #for max selection
                ll = soft_max[0,:]  
            else:
                #for sample
                ll = soft_max.squeeze() 
    
            return ll, now_length, rec_new, exchange[0], act, che_mask
        else:
            if test:
                #for max selection
                ll = soft_max[0,:]  
            else:
                #for sample
                ll = soft_max.squeeze() 
    
            return ll      


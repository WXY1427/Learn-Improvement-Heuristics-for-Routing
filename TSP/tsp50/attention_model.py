import torch
import time
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import math
from graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
import numpy as np


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

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

        self.problem = problem
        self.n_heads = n_heads

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp:
            step_context_dim = embedding_dim + 1  # Embedding of last node + remaining_capacity
            node_dim = 7  # x, y, demand

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)           
        else:  # TSP
            assert problem.NAME == 'tsp', "Unsupported problem: {}".format(problem.NAME)

            node_dim = 2  # x, y            

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

    def forward(self, input, rec, input_info, pos, embed, exc, test, pre_length=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
            
        embedding, soft_max, exchange, entropy = self.embedder(self.init_embed(input_info)+pos, embed, test, exc)            

        # rec: bs, graph_size; exchange: n_heads, bs, 2
        exchange_once = exchange[0,:,:]

        loc_of_first = torch.nonzero(rec.long() == exchange_once[:,0][:,None])
        
        loc_of_second = torch.nonzero(rec.long() == exchange_once[:,1][:,None])
        
        exchange_for_now = torch.cat((loc_of_first[:,1][:,None], loc_of_second[:,1][:,None]),1)

        # return (length_now-length_pre), None, length_now, dataset, rec_new from get_costs()
        cost, now_length, rec_new  = self.problem.get_costs(input, exchange_for_now, rec, pre_length)

        # exchange: n_heads, batch_size, 2； soft_max: n_heads, batch_size
        if test:
            ll = soft_max[0,:]  #for max selection
        else:
            ll = soft_max.squeeze() #for sample

        return cost, ll, now_length, rec_new, embedding, exchange[0], entropy
    
    def _emdedding(self, input, rec):
        
        ########## input: batch_size, graph, 2
        
        bs, gs = rec.size()
        
        seq_tensor_index = torch.cat((rec.long()[:,-1][:,None], rec.long()), 1)
        
        node_2_cor = []
        
        for i in range(gs):
            
            cor = torch.nonzero(rec.long() == i)
            
            pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
            mid = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
            cor_indice = torch.cat((pre[:,None], mid[:,None]),1)
            
            cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))  
            
            cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
            node_2_cor.append(cor_single_fla)
            
        return torch.stack(node_2_cor, 1)
    

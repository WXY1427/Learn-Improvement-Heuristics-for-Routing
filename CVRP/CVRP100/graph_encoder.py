import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        # Calculate queries, (n_heads, batch_size, n_query, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
            
        

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)   
       

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out
    
    
class MultiHeadAttention_to_attn(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention_to_attn, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, exchange, che_mask, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        gs = graph_size//2
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        compatibility = F.tanh(compatibility) * 10.
            
        mask_dia = torch.tril(torch.ones(graph_size,graph_size)).view(1,1,graph_size,graph_size).expand_as(compatibility)
        compatibility[mask_dia.byte()] = -np.inf
        

        tt=compatibility[1-mask_dia.byte()].clone() 
        tt[1-che_mask.byte()] = -np.inf
        
        compatibility[1-mask_dia.byte()] = tt
        compatibility[:,:,-28:,:] = -np.inf
        
        if exchange is not None:
            compatibility[0][torch.arange(batch_size), exchange[:,1], exchange[:,0]] = -np.inf            
            compatibility[0][torch.arange(batch_size), exchange[:,0], exchange[:,1]] = -np.inf
        
        compatibility[:,:,:,100] = -np.inf    
            
        im = compatibility.view(self.n_heads, batch_size, -1)
        assert (im[0] > -1e10).data.any(dim=-1).all(), print(im[0][1-(im[0] > -1e10).data.any(dim=-1).byte()])        
        im_l = F.log_softmax(im,dim = -1)                               # for sample
        im_s = F.softmax(im,dim = -1)                               # for sample   
                 

        return im_l, im_s

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
        
class one_step_to_get_attn(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(one_step_to_get_attn, self).__init__(
                MultiHeadAttention_to_attn(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
            )
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
#         self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        
        self.current_best = nn.Linear(1, embed_dim)
        
        self.rnn = nn.Linear(embed_dim, embed_dim)        
        
        self.project_graph = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.project_node = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))
        
#         self.one_attn_pre = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)       
        
        self.one_attn = MultiHeadAttention_to_attn(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
#         self.one_attn = one_step_to_get_attn(n_heads, embed_dim, feed_forward_hidden, normalization)
        
        self.heads = n_heads
        
#         self.second_attn = second_step_to_get_attn(n_heads, embed_dim, feed_forward_hidden, normalization)

    def forward(self, x, test, exchange, che_mask, action, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        bs, gs, in_d = x.size()

        h_em = self.layers(x)
    
        graph_embed = h_em.max(1)[0]

        fixed_context = self.project_graph(graph_embed)[:, None, :]    ######## batchsize, 1, embed_dim
        
        node_feature = self.project_node(h_em) ######## batchsize, gs, embed_dim
        
        fusion = node_feature + fixed_context.expand_as(node_feature)

        ### attn: (n_heads, batch_size, n_query, graph_size)
        att, att_s = self.one_attn(fusion, exchange, che_mask)  
        
        if action is None:
            if test:   
                atten = att_s.view(self.heads, bs, gs, gs)   # for max selection
                ########### n_heads, batch_size
                softmax_max = atten.max(-1)[0].max(-1)[0]
            
                ############ exchange: n_heads, batch_size, 2
                row = atten.max(-1)[0].max(-1)[1].unsqueeze(-1)
                col = atten.max(-1)[1].gather(2, row)
                exc = torch.cat((row,col),-1)
    
            else:
                ## sample
                inde = att_s.squeeze().multinomial(1)
                softmax_max = att.squeeze().gather(1,inde)
                while not (softmax_max > -1e10).data.all():
                    inde = att_s.squeeze().multinomial(1)
                    softmax_max = att.squeeze().gather(1,inde)
                col = inde % gs
                row = inde // gs
                exc = torch.cat((row,col),-1)
                exc = exc[None,:,:]
            assert (softmax_max > -1e10).data.all(), print(softmax_max[softmax_max<-1e10])
            return (         
                softmax_max,
                exc,
                inde
            )
        else:
            if test:   
                atten = att_s.view(self.heads, bs, gs, gs)   # for max selection
                ########### n_heads, batch_size
                softmax_max = atten.max(-1)[0].max(-1)[0]
            
                ############ exchange: n_heads, batch_size, 2
                row = atten.max(-1)[0].max(-1)[1].unsqueeze(-1)
                col = atten.max(-1)[1].gather(2, row)
                exc = torch.cat((row,col),-1)
    
            else:
                ## sample
                softmax_max = att.squeeze().gather(1,action)           
            
        
            assert (softmax_max > -1e10).data.all(), print(softmax_max[softmax_max<-1e10])
            return (         
                softmax_max
            )

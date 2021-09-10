from torch import nn
from graph_encoder import MultiHeadAttentionLayer
import torch
from problems import TSP, CVRP, SDVRP
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

class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,   
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()
        
        self.init_embed = nn.Linear(input_dim, embedding_dim)

        self.project_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.project_node = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads=1, embed_dim=embedding_dim)
            for _ in range(n_layers)
        ))
        
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs, rec, i, p):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """ 
        if i is None:       
            input_info, pos_enc = self._emdedding(inputs, rec)
            
        
            return input_info, pos_enc
        else:
            input_info, pos_enc = i, p
            
            hb = self.init_embed(input_info)+pos_enc#+depot_embed[:, None, :].expand(*pos_enc.size())
            
            h = self.encoder(hb)
        
            graph_embed = h.mean(1)

            fixed_context = self.project_graph(graph_embed)[:, None, :]    ######## batchsize, 1, embed_dim
        
            node_feature = self.project_node(h) ######## batchsize, gs, embed_dim
        
            fusion = node_feature + fixed_context.expand_as(node_feature)
        
            return self.value_head(fusion.mean(dim=1))
    
    def _emdedding(self, input, seq_tensor):
        
        bs, graph_2 = seq_tensor.size()
        
        loc_with_depot = input['loc'].view(-1, graph_2, 2) ########## batch_size, graph+1, 2
        
        enc = position_encoding_init(graph_2, 128)      
        
        enc = enc.cuda()
        
        enc_b = enc.expand(bs,graph_2,128)

        seq_tensor_index = torch.cat((seq_tensor[:,-1][:,None], seq_tensor, seq_tensor[:,0][:,None]),1)
        
        cor_3_demand = []
        
        pos_enc = []
        
        for i in range(1, graph_2+1):
            
            cor = torch.nonzero(seq_tensor == i)        
            
            pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
            mid = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
            las = seq_tensor_index[cor[:,0], cor[:,1]+2]    
           
            dem = input['demand'].gather(1, mid[:,None]-1) 
            
            cor_indice = torch.cat((pre[:,None]-1, mid[:,None]-1, las[:,None]-1),1)
            
            cor_single = loc_with_depot.gather(1, cor_indice[..., None].expand(*cor_indice.size(), loc_with_depot.size(-1)))  
            
            cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
            cor_with_demand = torch.cat((cor_single_fla, dem),1)
            
            cor_3_demand.append(cor_with_demand)

            single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

            pos_enc.append(single_pos.squeeze())
            
        return torch.stack(cor_3_demand, 1), torch.stack(pos_enc, 1)
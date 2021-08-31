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
        
        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads=1, embed_dim=embedding_dim)
            for _ in range(n_layers)
        ))
        
        self.project_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.project_node = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs, rec):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """      
               
        input_info, pos_enc = self._emdedding(inputs, rec)
            
        h = self.encoder(self.init_embed(input_info)+pos_enc)

        graph_embed = h.mean(1)

        fixed_context = self.project_graph(graph_embed)[:, None, :]    ######## batchsize, 1, embed_dim
        
        node_feature = self.project_node(h) ######## batchsize, gs, embed_dim
        
        fusion = node_feature + fixed_context.expand_as(node_feature)

        return self.value_head(fusion.mean(dim=1)), input_info, pos_enc
    
    def _emdedding(self, input, rec):
        
        ########## input: batch_size, graph, 2
        bs, gs = rec.size()
        
        enc = position_encoding_init(gs, 128)      
        
        enc = enc.cuda()
        
        enc_b = enc.expand(bs,gs,128)
        
        seq_tensor_index = rec.long()
    
        node_2_cor = []
        
        pos_enc = []
        
        for i in range(gs):
            
            cor = torch.nonzero(rec.long() == i)
            
            pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
            cor_indice = pre[:,None]    
     
            cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))    
    
            cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
            node_2_cor.append(cor_single_fla)
            
            single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

            pos_enc.append(single_pos.squeeze())

        return torch.stack(node_2_cor, 1), torch.stack(pos_enc, 1)
    
    

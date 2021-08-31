import os
import time
from tqdm import tqdm
import torch
import math

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel

# from attention_model import set_decode_type
from log_utils import log_values
import torch.nn.functional as F
import numpy as np
from insertion import insertion
from problems import TSP, CVRP, SDVRP

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def emded(input, rec):
        
        ########## input: batch_size, graph, 2
    bs, gs = rec.size()
        
    enc = position_encoding_init(gs, 128)      
        
    enc = enc.cuda()
        
    enc_b = enc.expand(bs,gs,128)
        
#         seq_tensor_index = torch.cat((rec.long()[:,-1][:,None], rec.long()), 1)
    seq_tensor_index = rec.long()
    
    node_2_cor = []
        
    pos_enc = []
        
    for i in range(gs):
            
        cor = torch.nonzero(rec.long() == i)
            
        pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
#             mid = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
#             cor_indice = torch.cat((pre[:,None], mid[:,None]),1)
        cor_indice = pre[:,None]    
    
#             cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))  
        cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))    
    
        cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
        node_2_cor.append(cor_single_fla)
            
        single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

        pos_enc.append(single_pos.squeeze())

    return torch.stack(node_2_cor, 1), torch.stack(pos_enc, 1)

def make_var(val, cuda=False, **kwargs):
    if isinstance(val, dict):
        return {k: make_var(v, cuda, **kwargs) for k, v in val.items()}
    var = Variable(val, **kwargs)
    if cuda:
        var = var.cuda()
    return var


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    model.eval()
    dataset = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0)
    batch_to_link = []
    return_m = []
    best_solution =[]
    for batch_id_e, batch_e in enumerate(dataset):

#         rec_v = insertion(batch_e, met='nearest')
#         rec_v = rec_v.cuda()
        batch_e = make_var(batch_e, opts.use_cuda)

        rec_v = torch.linspace(0, opts.graph_size-1, steps=opts.graph_size).expand(opts.eval_batch_size, opts.graph_size)
        rec_v = make_var(rec_v, opts.use_cuda)

#         rec_v = torch.load('ICLR_greedy_50.pt')
#         rec_v = make_var(rec_v, opts.use_cuda)
        current_length = None
        exchange = None
#         total_return = []
        length_current = []
        em = torch.zeros(opts.eval_batch_size, opts.graph_size, opts.embedding_dim).cuda()
        s_time = time.time()
        
        rec_reward=[]
        seg = opts.graph_size//4            
        rec_best = rec_v.clone() 
        
        for T in range(opts.steps):
            
            input_info, pos_enc = emded(batch_e, rec_v)
            
            pre_length, log_likelihood, current_length, rec_v, em, exchange, entropy = model(batch_e, rec_v, input_info, pos_enc, em, exchange, test = False, pre_length=current_length)
            
#             cost = pre_length-current_length
            
#             cost_total = cost_total + cost.detach()
            
#             total_return.append(cost_total)
            
            length_current.append(current_length)
        
        
            if T==0:
                now_best=pre_length
            pre_best = now_best.clone() 
            now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]        
            reward = pre_best - now_best 
            rec_reward.append(reward)                
            rec_best[reward>0]=rec_v[reward>0]
            
                
#             if T%10==0:
        
#                 rec_mem = torch.stack(rec_reward,0)            
#                 ite = (rec_mem[-10:]<0.00001).sum(0)
#                 for j, k in enumerate(ite.tolist()):
#                     if k>9:
                                                  
#                         rec_v[j] = rec_best[j].clone()
#                         rec_reward[-1][j]=1.
#                 input_info, pos_enc = emded(batch_e, rec_v)            
#                 current_length, _, _, _, _, _, _ = model(batch_e, rec_v, input_info, pos_enc, em, exchange, test = False, pre_length=None)
#                 pre_best = now_best 

#                 now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
#                 reward = pre_best - now_best 
#                 rec_reward.append(reward)                
#                 rec_best[reward>0]=rec_v[reward>0]            
#                 length_current.append(current_length)                    
        
            
        duration = time.time() - s_time
        print("Test, took {} s".format(time.strftime('%H:%M:%S', time.gmtime(duration))))
        
        current_m = torch.stack(length_current, 0)
        best_solution.append(current_m.min(0)[0])
        
#         return_total = torch.stack(total_return, 0) 
#         return_max = return_total.max(0)[0]
#         return_m.append(return_max)
#         batch_to_link.append(cost_total)
#         total_cost = torch.cat(batch_to_link,0)
#         return_return = torch.cat(return_m,0)
        best = torch.cat(best_solution,0)
#     torch.save(current_m,'exp3_our_50result(10000)_2.pt')
    return best










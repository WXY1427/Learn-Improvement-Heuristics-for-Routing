from torch.utils.data import Dataset
import torch
import os
import pickle
# import addings

import time
from torch.utils.data import DataLoader
import json
from itertools import combinations
import math
from scipy.spatial import distance_matrix
import models.data_utils.data_utils as data_utils
import vrpDatagen
from itertools import combinations, permutations
import numpy as np




class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, rec, dic, cl, exchange=None):  ###### exchange instead of pi: n_heads, batch_size, 2
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param exchange: (n_heads, batch_size, 2) meaning exchange two nodes;
        :return: (batch_size) reward: improving of the length
        """

#         exchange_one_head = exchange  ############ batch_size, 2 
        
        batch_size, graph_size = dataset['demand'].size()

#         loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)   ########## batch_size, graph+1, 2
        loc_with_depot = dataset['loc']   ########## batch_size, graph+1, 2   
    
        comb_num, g2s = dic.size()

        ########## length of previous
        if exchange is None:  

            pi = rec-1 

            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))      
 
            length_now = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
    
            rn = rec.clone()

            rec_rep = rn[:,None,:].expand(batch_size, comb_num, graph_size).reshape(-1,graph_size)
             
            # 2-opt move in advance
            mov_set = rec_rep.gather(1, dic.repeat(batch_size, 1))
                
            che_mask = CVRP.seq_tensor2(dataset['demand'], mov_set)
            
            cor = rec_rep.gather(1, cl.repeat(batch_size, 1)).sort(1)[0]
            
            order = (cor[:,0]*(g2s*2)+cor[:,1]).view(batch_size, -1).sort(1)[1]
            
            cm = che_mask.view(batch_size, -1).gather(1,order)
        

            return length_now, rec, cm.flatten()
        else:
            ############# change the record    

            rec_new = rec.cpu()
            exchange_sort = exchange.sort(1)[0].cpu()
#             for i in range(batch_size):
#                 inter = rec_new[i][exchange_sort[i][0]:exchange_sort[i][1]+1].clone()
#                 rec_new[i][exchange_sort[i][0]:exchange_sort[i][1]+1] = torch.flip(inter,[0])
                
            for i in range(batch_size):
                niu = rec_new[i]
                cc=exchange_sort[i]
                aa=cc[0]
                bb=cc[1]
                inter = niu[aa:bb+1].clone()              
                niu[aa:bb+1] = torch.flip(inter,[0])                
            rec_new= rec_new.cuda()
        
         
            pi = rec_new-1

            d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))            
            
            length_now = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)             
            
            rn = rec_new.clone()            
  
            rec_rep = rn[:,None,:].expand(batch_size, comb_num, graph_size).reshape(-1,graph_size)
 
    
    

            # 2-opt move in advance
            mov_set = rec_rep.gather(1, dic.repeat(batch_size, 1)) 
 
          
            che_mask = CVRP.seq_tensor2(dataset['demand'], mov_set)

            
         
            cor = rec_rep.gather(1, cl.repeat(batch_size, 1)).sort(1)[0]
            
            order = (cor[:,0]*(g2s*2)+cor[:,1]).view(batch_size, -1).sort(1)[1]
            
            cm = che_mask.view(batch_size, -1).gather(1,order)
 

            return length_now, rec_new, cm.flatten()
        
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)  
    
    @staticmethod
    def seq_tensor(input, rec, capacity=1.):
#         input = {k: v.cpu() for k, v in input.items()}
        depot_loc = CVRP.addings(input, rec)
#         depot_loc = depot_loc.cuda()
#         print(depot_loc)
        bs, gs = input['demand'].size()
        matrix_i = torch.ones(bs,2*gs).long().cuda()
        matrix_acc = torch.zeros(bs,2*gs).long().cuda()
        counter = 0
        for i in range(gs-1):
#             print(depot_loc[:,i])
            matrix_i[torch.arange(bs).cuda(), depot_loc[:,i]+1+counter]=0 
            counter+=1   
        matrix_z_i = matrix_i.clone()
        for i in range(2*gs):
            matrix_acc[:,i]=matrix_i[:,:i+1].sum(1).clone()
        
        matrix_acc[matrix_acc>gs]=0
        matrix_acc[matrix_z_i==0]=0       
        dic = torch.cat((torch.zeros(bs,1).cuda(), rec),1)        
        
        return dic.gather(1, matrix_acc).long() 
    
    @staticmethod
    def seq_tensor2(input, rec):
      
        bs, g2s = rec.size()
        sz = input.size()[0]
      
        inp_dem = input[:,None,:].expand(sz, bs//sz, g2s).reshape(-1,g2s)

        bb = inp_dem.gather(1, rec-1)
        b=bb.cumsum(1)
 
        dep_ins = torch.nonzero(rec>50)[:,1] 
 
        
         
        dep_insert = dep_ins.reshape(bs,-1)                         

        dep_insert_g = torch.cat((dep_insert, (g2s-1)*torch.ones(bs,1).long().cuda()),1)
        c=b.gather(1, dep_insert_g) #select relevant terms
        c[:,1:]=c[:,1:]-c[:,:-1]        
        tr=c
        
        tr[:,0]=tr[:,-1]=tr[:,0]+tr[:,-1]
        proceed_depot = (tr<=1.0 + 1e-5).all(dim=1)                         
         
        return proceed_depot      

    @staticmethod
    def addings(batch, rec): ######## given opts and part number

        bs, gs = batch['demand'].size()
        parts = torch.ceil(batch['demand'].sum(1)).long().tolist()
        dem_now = batch['demand'].gather(1,rec.long()-1)
        dem_now_cpu = dem_now.cpu()
        loc_now = batch['loc'].gather(1,(rec.long()-1)[...,None].expand_as(batch['loc']))
        loc_now_cpu = loc_now.cpu()
        dep_cor = batch['depot'].cpu()
        lis = []
#         saving_v_lis = []
        saving_i_lis = []
        for d in range(bs):

            dij = torch.from_numpy(distance_matrix(loc_now_cpu[d], loc_now_cpu[d]))
            did = torch.from_numpy(distance_matrix(loc_now_cpu[d], dep_cor[d][None,:]))
            did_sum = did+did.t()
            ind = torch.arange(gs-1).cuda()
            sav = did_sum[ind, ind+1]-dij[ind, ind+1]

#             saving = dict()
#             for i in range(gs-1):
#                 j = i+1
#                 saving[(i, j)] = CVRP.computeSaving(dep_cor[d], loc_now[d][i], loc_now[d][j])

#             saving_lis = sorted(saving.items(), key=lambda kv: kv[1])

            saving_lis_v, saving_lis_i = sav.sort(0)
#             saving_v_lis.append(saving_lis_v)
            saving_i_lis.append(saving_lis_i)        
            
#         mat_v = torch.stack(saving_v_lis, 0)
        mat_i = torch.stack(saving_i_lis, 0)
        
#         b=time.time()
        depot_loc = CVRP.depot(mat_i, dem_now_cpu, parts[d]-1)
#         print(time.time()-b)

        return depot_loc.cuda()
    
#     @staticmethod    
#     def depot(saving, saving_lis, dem, parts):
#         gs = len(dem)
#         adding_number = len(saving_lis) 
#     #     for i in range(adding_number-2):
#     #         for j in range(i+1,adding_number-1):
#         for i in list(combinations([c[0] for c in saving_lis], parts)):
#             print(i)
#             ads = []
#     #         adding = 0
#             for j in list(i):
#                 ads.append(j[0])
#     #             adding+=saving[j]
#             ads = sorted(ads)
#             lis = []
#             for l in range(parts-1):
#                 lis.append(dem[ads[l]+1:ads[l+1]+1].sum())
#             lis.append(dem[:ads[0]+1].sum())
#             lis.append(dem[ads[-1]+1:].sum())
                
#             if all(e <= 1. for e in lis):
#                 print('pass')
#                 return ads+[gs]*(gs-len(ads))
#         return CVRP.depot(saving, saving_lis, dem, parts+1) 

    @staticmethod    
    def depot(saving_lis_i, dem, parts):
        bs, gs = dem.size()

#         adding_number = gs-1
#         ads = []
#         for i in [c[0] for c in saving_lis]:

        matrix_one = torch.ones(bs,gs)
        for i in range(0,gs-1):

#             ads.append(saving_lis[1][i])
    #             adding+=saving[j]
            ads = saving_lis_i[:,0:i+1].sort(1)[0]
#             print(ads)
            lis = []
            for l in range(i+1-1):
#                 w=time.time()
#                 demc=dem.cpu()
                

                mat_one = torch.ones(bs,gs)
                mat_one[torch.arange(bs), ads[:,l].squeeze()]=0

                matt_one = torch.ones(bs,gs)
                matt_one[torch.arange(bs), ads[:,l+1].squeeze()+1]=0                
                ite = torch.cat((torch.flip(mat_one,[1])[None,:],matt_one[None,:]),0)
#                 lis.append(dem[ads[l]+1:ads[l+1]+1].sum())

#                 it=torch.ones(2,bs).cuda()
#                 for k in range(gs):
#                     it*=ite[:,:,k] 
#                     ite[:,:,k]=it
                    
                ite_ = torch.cumprod(ite, dim=2)

                dem_to_sum = torch.flip(ite_[0],[1])*ite_[1]*dem
                lis.append(dem_to_sum.sum(1))
                if l==0:
                    sta = torch.flip(ite_[0],[1])
                    lis.append(((sta==0).float()*dem).sum(1))
                if l==(i-1):
                    end = ite_[1]   
                    lis.append(((end==0).float()*dem).sum(1))
#                 print(time.time()-w)    
            if i==0:
#                 demc=dem.cpu()
                mat_one = torch.ones(bs,gs)
                mat_one[torch.arange(bs), ads[:,0].squeeze()]=0

                matt_one = torch.ones(bs,gs)
                matt_one[torch.arange(bs), ads[:,0].squeeze()+1]=0                
                ite = torch.cat((torch.flip(mat_one,[1])[None,:],matt_one[None,:]),0)

#                 it=torch.ones(2,bs).cuda()
#                 for j in range(gs):
#                     it*=ite[:,:,j] 
#                     ite[:,:,j]=it
                ite_ = torch.cumprod(ite, dim=2)
                sta = torch.flip(ite_[0],[1])

                end = ite_[1]   
                lis.append(((sta==0).float()*dem).sum(1))
                lis.append(((end==0).float()*dem).sum(1))                
#             print(lis)
            mat_seg_dem = torch.stack(lis,1)

            mat_ind = ((mat_seg_dem<1.).sum(1))==i+2
#             print(mat_ind)
            col = torch.nonzero(mat_ind==1)


#                 print(col)
            matrix_one[col.squeeze(),torch.zeros(len(col)).long()+i+1]=gs   
#             print((matrix_one==gs).sum())
            
            if all(matrix_one.sum(1)>gs):
                retrive = matrix_one.clone()
                for j in range(1,gs+1):
                    matrix_one[:,j-1]=retrive[:,0:j].sum(1)
                ret = torch.cat((saving_lis_i.cpu(),torch.zeros(bs,1).long()),1)   
                ret[matrix_one>gs]=gs
#                 print((matrix_one>gs).sum(1))
                return ret.sort(1)[0]


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        CAPACITY = 1.
        demands = torch.cat(
            (
                dataset['demand'][:, :1] * 0 - CAPACITY,  # Hacky
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = dataset['demand'][:, 0] * 0  #  torch.zeros(batch_size, 1, out=dataset['demand'].data.new())
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0))[:, None]] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) from each next location from its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000):
        super(VRPDataset, self).__init__()

        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'demand': torch.FloatTensor(demand) / capacity,  # Normalize so capacity = 1
                        'depot': torch.FloatTensor(depot),
                    }
                    for depot, loc, demand, capacity in data[:num_samples]
                ]
        else:
            vrpDatagen.gen(num_samples, size, CAPACITIES[size])
            with open(os.path.join('data', 'vrp', "vrp_20_30.json"), 'r') as f:
                 DT = json.load(f)
                    
            DataProcessor = data_utils.vrpDataProcessor()
            samples = DataProcessor.get_batch(DT, len(DT))
            
            self.data = [
                # {
                #     'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                #     'demand': torch.FloatTensor(size).uniform_(0, 1),
                #     'depot': torch.FloatTensor(2).uniform_(0, 1)
                # }
                {
                    'loc': torch.cat((torch.FloatTensor(samples[i].route)[:,0:2], torch.FloatTensor(samples[i].route)[0,0:2].repeat(100-len(samples[i].route), 1)), 0),                  
                    'demand': torch.cat((torch.FloatTensor(samples[i].route)[:,2].float(), torch.zeros(100-len(samples[i].route))),0),
                    'depot': torch.FloatTensor(samples[i].route)[0,0:2]
                }

                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

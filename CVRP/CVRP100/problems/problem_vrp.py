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
        
        batch_size, graph_size = dataset['demand'].size()
        loc_with_depot = dataset['loc']   ########## batch_size, graph+1, 2   
    
        comb_num, g2s = dic.size()

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

            rec_new = rec.cpu()
            exchange_sort = exchange.sort(1)[0].cpu()
            for i in range(batch_size):
                inter = rec_new[i][exchange_sort[i][0]:exchange_sort[i][1]+1].clone()
                rec_new[i][exchange_sort[i][0]:exchange_sort[i][1]+1] = torch.flip(inter,[0])
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
    def seq_tensor2(input, rec):
        
        bs, g2s = rec.size()
      
        inp_dem = input[:,None,:].expand(input.size()[0], bs//input.size()[0], g2s).reshape(-1,g2s)
        
        demand = inp_dem.clone()

        bb = demand.gather(1, rec.long()-1)
        b=bb.cumsum(1)
 
        dep_insert = rec.sort(1)[1][:,100:]
        dep_insert = dep_insert.sort(1)[0]    
      
        dep_insert_g = torch.cat((dep_insert, (g2s-1)*torch.ones(bs,1).long().cuda()),1)

        c=b.gather(1, dep_insert_g) #select relevant terms
        d=torch.cat((torch.zeros(bs,1).cuda(), b.gather(1, dep_insert)),1) #select relevant terms 
                        
        tr = c-d
        dem_per_rou = torch.cat(((tr[:,0]+tr[:,-1])[:,None],tr[:,1:-1]),1)                       
        proceed_depot = (dem_per_rou<=1.0 + 1e-5).all(dim=1)                         
        
        return proceed_depot

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
                {
                    'loc': torch.cat((torch.FloatTensor(samples[i].route)[:,0:2], torch.FloatTensor(samples[i].route)[0,0:2].repeat(128-len(samples[i].route), 1)), 0),                  
                    'demand': torch.cat((torch.FloatTensor(samples[i].route)[:,2].float(), torch.zeros(128-len(samples[i].route))),0),
                    'depot': torch.FloatTensor(samples[i].route)[0,0:2]
                }

                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
